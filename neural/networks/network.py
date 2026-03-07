import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ops import MultiHeadAttentionBlock, PositionalEncoding, DenseResidualBlock, CardEmbedding
from networks.card_abstraction import compute_card_features, CARD_FEATURE_DIM, CardAbstractionEncoder

NUM_ACTIONS = 10


class ContextUpdater(nn.Module):
    """GRU-style context updater: compresses history into fixed-size vector."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, features, prev_context):
        combined = torch.cat([features, prev_context], dim=1)
        z = self.update_gate(combined)
        r = self.reset_gate(combined)
        candidate = self.candidate(torch.cat([features, r * prev_context], dim=1))
        new_context = (1 - z) * prev_context + z * candidate
        return new_context


def _build_model(name, lr, embedding_dim, hidden_dim):
    """Shared model builder — returns dict of modules and optimizer."""
    modules = nn.ModuleDict()

    # Card encoding
    modules['card_embedding'] = CardEmbedding(embedding_dim)
    modules['card_pos'] = PositionalEncoding(embedding_dim, max_len=7)
    modules['card_attn'] = MultiHeadAttentionBlock(hidden_dim=embedding_dim, num_heads=8)
    modules['card_flat'] = nn.Flatten()

    # Card abstraction encoder
    modules['card_abs_encoder'] = CardAbstractionEncoder(hidden_dim // 2)

    # Categorical embeddings
    modules['stage_embedding'] = nn.Embedding(4, 32)
    modules['position_embedding'] = nn.Embedding(3, 32)

    # Feature encoder: card embeddings + card abstraction + numeric + categorical
    card_features_dim = 7 * embedding_dim  # 2 private + 5 public
    card_abs_dim = hidden_dim // 2
    numeric_dim = 9  # 3 stacks + 3 bets + 3 active mask
    categorical_dim = 32 + 32  # stage + position
    input_dim = card_features_dim + card_abs_dim + numeric_dim + categorical_dim

    modules['features_net'] = nn.Sequential(
        nn.Linear(input_dim, hidden_dim * 4),
        nn.LayerNorm(hidden_dim * 4),
        nn.GELU(),
        nn.Linear(hidden_dim * 4, hidden_dim * 2),
        nn.LayerNorm(hidden_dim * 2),
        nn.GELU(),
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
    )

    # Context updater
    modules['context_updater'] = ContextUpdater(hidden_dim)

    # Deeper trunk: 3 residual blocks
    modules['main_net'] = nn.Sequential(
        DenseResidualBlock(hidden_dim * 2, hidden_dim),
        nn.GELU(),
        DenseResidualBlock(hidden_dim, hidden_dim),
        nn.GELU(),
        DenseResidualBlock(hidden_dim, hidden_dim),
        nn.GELU(),
    )

    # Stage-specific action heads
    stage_heads = nn.ModuleList([
        nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS),
        )
        for _ in range(4)
    ])
    for head in stage_heads:
        nn.init.xavier_uniform_(head[2].weight, gain=0.1)
        nn.init.zeros_(head[2].bias)
    modules['stage_heads'] = stage_heads

    return modules


class DeepCFRModel(nn.Module):
    def __init__(self, name, lr=1e-3, embedding_dim=128, hidden_dim=128):
        super(DeepCFRModel, self).__init__()
        self.name = name
        self.step = 0
        self.hidden_dim = hidden_dim

        self.modules_dict = _build_model(name, lr, embedding_dim, hidden_dim)
        # Expose as attributes for easy access
        self.card_embedding = self.modules_dict['card_embedding']
        self.card_pos = self.modules_dict['card_pos']
        self.card_attn = self.modules_dict['card_attn']
        self.card_flat = self.modules_dict['card_flat']
        self.card_abs_encoder = self.modules_dict['card_abs_encoder']
        self.stage_embedding = self.modules_dict['stage_embedding']
        self.position_embedding = self.modules_dict['position_embedding']
        self.features_net = self.modules_dict['features_net']
        self.context_updater = self.modules_dict['context_updater']
        self.main_net = self.modules_dict['main_net']
        self.stage_heads = self.modules_dict['stage_heads']

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=3e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5000, T_mult=2, eta_min=1e-5
        )

    def encode_features(self, x):
        public_cards, private_cards, stacks, _, bets, active_players_mask, stage, current_player_pos = x
        device = private_cards.device

        # Card embeddings via attention
        cards_mask = torch.cat([(private_cards == -1), (public_cards == -1)], dim=1)
        cards_features = self.card_attn(torch.cat([
            self.card_embedding(private_cards),
            self.card_pos(self.card_embedding(public_cards))
        ], dim=1), key_padding_mask=cards_mask)

        # Card abstraction features (hand strength, draws, etc.)
        card_abs_features = compute_card_features(private_cards, public_cards, device)
        card_abs_encoded = self.card_abs_encoder(card_abs_features)

        raw_features = self.features_net(torch.cat([
            self.card_flat(cards_features),
            card_abs_encoded,
            stacks,
            bets,
            active_players_mask,
            self.stage_embedding(stage)[:, 0, :],
            self.position_embedding(current_player_pos)[:, 0, :]
        ], dim=1))
        return raw_features

    def get_action_logits(self, features, context, stages):
        combined = torch.cat([features, context], dim=1)
        shared = self.main_net(combined)
        logits = torch.zeros(features.size(0), NUM_ACTIONS, device=features.device)
        for stage_idx in range(4):
            mask = (stages == stage_idx)
            if mask.any():
                logits[mask] = self.stage_heads[stage_idx](shared[mask])
        return logits

    @torch.no_grad()
    def get_probs(self, x, actions_mask, prev_context=None):
        _, _, _, _, _, _, stage, _ = x
        raw_features = self.encode_features(x)

        if prev_context is None:
            prev_context = torch.zeros(raw_features.size(0), self.hidden_dim, device=raw_features.device)

        new_context = self.context_updater(raw_features, prev_context)

        stages = stage.squeeze(1)
        logits = self.get_action_logits(raw_features, new_context, stages)

        # Mask illegal actions before softmax
        logits = logits - logits.max(dim=1, keepdim=True).values
        logits = logits.masked_fill(actions_mask == 0, -1e9)
        probs = torch.softmax(logits, dim=1)

        bad_rows = probs.sum(dim=1) < 1e-8
        if bad_rows.any():
            probs[bad_rows] = actions_mask[bad_rows]
            probs[bad_rows] = probs[bad_rows] / probs[bad_rows].sum(dim=1, keepdim=True)

        return probs, new_context

    def save(self, name=None):
        path = f"./checkpoints/{self.name}-{name}.pth" if name is not None else f"./checkpoints/{self.name}.pth"
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save({
            'net': self.state_dict(),
            'opt': self.optimizer.state_dict(),
            'step': self.step,
        }, path)

    def load(self, name=None, map_location=None):
        path = f"./checkpoints/{self.name}-{name}.pth" if name is not None else f"./checkpoints/{self.name}.pth"
        dat = torch.load(path, weights_only=False, map_location=map_location)
        self.load_state_dict(dat['net'])
        self.optimizer.load_state_dict(dat['opt'])
        if 'step' in dat:
            self.step = dat['step']


class AvgStrategyModel(nn.Module):
    """Average strategy network — same architecture as DeepCFRModel."""

    def __init__(self, name, lr=1e-3, embedding_dim=128, hidden_dim=128):
        super(AvgStrategyModel, self).__init__()
        self.name = name
        self.step = 0
        self.hidden_dim = hidden_dim

        self.modules_dict = _build_model(name, lr, embedding_dim, hidden_dim)
        self.card_embedding = self.modules_dict['card_embedding']
        self.card_pos = self.modules_dict['card_pos']
        self.card_attn = self.modules_dict['card_attn']
        self.card_flat = self.modules_dict['card_flat']
        self.card_abs_encoder = self.modules_dict['card_abs_encoder']
        self.stage_embedding = self.modules_dict['stage_embedding']
        self.position_embedding = self.modules_dict['position_embedding']
        self.features_net = self.modules_dict['features_net']
        self.context_updater = self.modules_dict['context_updater']
        self.main_net = self.modules_dict['main_net']
        self.stage_heads = self.modules_dict['stage_heads']

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=3e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5000, T_mult=2, eta_min=1e-5
        )

    def encode_features(self, x):
        public_cards, private_cards, stacks, _, bets, active_players_mask, stage, current_player_pos = x
        device = private_cards.device

        cards_mask = torch.cat([(private_cards == -1), (public_cards == -1)], dim=1)
        cards_features = self.card_attn(torch.cat([
            self.card_embedding(private_cards),
            self.card_pos(self.card_embedding(public_cards))
        ], dim=1), key_padding_mask=cards_mask)

        card_abs_features = compute_card_features(private_cards, public_cards, device)
        card_abs_encoded = self.card_abs_encoder(card_abs_features)

        raw_features = self.features_net(torch.cat([
            self.card_flat(cards_features),
            card_abs_encoded,
            stacks,
            bets,
            active_players_mask,
            self.stage_embedding(stage)[:, 0, :],
            self.position_embedding(current_player_pos)[:, 0, :]
        ], dim=1))
        return raw_features

    def get_action_logits(self, features, context, stages):
        combined = torch.cat([features, context], dim=1)
        shared = self.main_net(combined)
        logits = torch.zeros(features.size(0), NUM_ACTIONS, device=features.device)
        for stage_idx in range(4):
            mask = (stages == stage_idx)
            if mask.any():
                logits[mask] = self.stage_heads[stage_idx](shared[mask])
        return logits

    def save(self, name=None):
        path = f"./checkpoints/{self.name}-{name}.pth" if name is not None else f"./checkpoints/{self.name}.pth"
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save({
            'net': self.state_dict(),
            'opt': self.optimizer.state_dict(),
            'step': self.step,
        }, path)

    def load(self, name=None, map_location=None):
        path = f"./checkpoints/{self.name}-{name}.pth" if name is not None else f"./checkpoints/{self.name}.pth"
        dat = torch.load(path, weights_only=False, map_location=map_location)
        self.load_state_dict(dat['net'])
        self.optimizer.load_state_dict(dat['opt'])
        if 'step' in dat:
            self.step = dat['step']
