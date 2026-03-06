import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ops import MultiHeadAttentionBlock, PositionalEncoding, DenseResidualBlock, CardEmbedding

NUM_ACTIONS = 10


class ContextUpdater(nn.Module):
    """GRU-style context updater: compresses history into fixed-size vector.
    Takes current features + previous context -> outputs updated context."""

    def __init__(self, hidden_dim):
        super().__init__()
        # GRU gates operating on [features, prev_context]
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
        """
        features: [batch, hidden_dim] — current step features
        prev_context: [batch, hidden_dim] — previous context (zeros if first step)
        Returns: [batch, hidden_dim] — updated context
        """
        combined = torch.cat([features, prev_context], dim=1)
        z = self.update_gate(combined)
        r = self.reset_gate(combined)
        candidate = self.candidate(torch.cat([features, r * prev_context], dim=1))
        new_context = (1 - z) * prev_context + z * candidate
        return new_context


class DeepCFRModel(nn.Module):
    def __init__(self, name, lr=1e-3, embedding_dim=128, hidden_dim=128):
        super(DeepCFRModel, self).__init__()
        self.name = name
        self.step = 0
        self.hidden_dim = hidden_dim

        # Card encoding
        self.card_embedding = CardEmbedding(embedding_dim)
        self.card_pos = PositionalEncoding(embedding_dim, max_len=7)
        self.card_attn = MultiHeadAttentionBlock(hidden_dim=embedding_dim, num_heads=8)
        self.card_flat = nn.Flatten()

        # Categorical embeddings
        self.stage_embedding = nn.Embedding(4, 32)
        self.position_embedding = nn.Embedding(3, 32)

        # Feature encoder
        card_features_dim = 7 * embedding_dim  # 2 private + 5 public
        numeric_dim = 9  # 3 stacks + 3 bets + 3 active mask
        categorical_dim = 32 + 32  # stage + position
        input_dim = card_features_dim + numeric_dim + categorical_dim

        self.features_net = nn.Sequential(
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

        # Context updater (replaces transformer history)
        self.context_updater = ContextUpdater(hidden_dim)

        # Shared trunk — takes features + context
        self.main_net = nn.Sequential(
            DenseResidualBlock(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            DenseResidualBlock(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Stage-specific action heads (preflop, flop, turn, river)
        self.stage_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, NUM_ACTIONS),
            )
            for _ in range(4)
        ])

        for head in self.stage_heads:
            nn.init.xavier_uniform_(head[2].weight, gain=0.1)
            nn.init.zeros_(head[2].bias)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=3e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5000, T_mult=2, eta_min=1e-5
        )

    def encode_features(self, x):
        public_cards, private_cards, stacks, _, bets, active_players_mask, stage, current_player_pos = x
        cards_mask = torch.cat([(private_cards == -1), (public_cards == -1)], dim=1)

        cards_features = self.card_attn(torch.cat([
            self.card_embedding(private_cards),
            self.card_pos(self.card_embedding(public_cards))
        ], dim=1), key_padding_mask=cards_mask)

        raw_features = self.features_net(torch.cat([
            self.card_flat(cards_features),
            stacks,
            bets,
            active_players_mask,
            self.stage_embedding(stage)[:, 0, :],
            self.position_embedding(current_player_pos)[:, 0, :]
        ], dim=1))
        return raw_features

    def get_action_logits(self, features, context, stages):
        """
        features: [batch, hidden_dim]
        context: [batch, hidden_dim]
        stages: [batch] int tensor (0-3)
        """
        combined = torch.cat([features, context], dim=1)  # [batch, hidden_dim*2]
        shared = self.main_net(combined)
        logits = torch.zeros(features.size(0), NUM_ACTIONS, device=features.device)
        for stage_idx in range(4):
            mask = (stages == stage_idx)
            if mask.any():
                logits[mask] = self.stage_heads[stage_idx](shared[mask])
        return logits

    @torch.no_grad()
    def get_probs(self, x, actions_mask, prev_context=None):
        """
        Inference: get action probabilities.
        prev_context: [batch, hidden_dim] or None — fixed-size context from previous step
        Returns: (probs, updated_context [batch, hidden_dim])
        """
        _, _, _, _, _, _, stage, _ = x
        raw_features = self.encode_features(x)

        if prev_context is None:
            prev_context = torch.zeros(raw_features.size(0), self.hidden_dim, device=raw_features.device)

        new_context = self.context_updater(raw_features, prev_context)

        stages = stage.squeeze(1)
        logits = self.get_action_logits(raw_features, new_context, stages)

        probs = torch.softmax(logits, dim=1)
        probs = probs * actions_mask

        if probs.sum().item() < 1e-8:
            probs = probs + actions_mask

        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs, new_context

    def save(self, name=None):
        if name is not None:
            path = f"./checkpoints/{self.name}-{name}.pth"
        else:
            path = f"./checkpoints/{self.name}.pth"
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save({
            'net': self.state_dict(),
            'opt': self.optimizer.state_dict(),
            'step': self.step,
        }, path)

    def load(self, name=None):
        if name is not None:
            dat = torch.load(f"./checkpoints/{self.name}-{name}.pth", weights_only=False)
        else:
            dat = torch.load(f"./checkpoints/{self.name}.pth", weights_only=False)
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

        self.card_embedding = CardEmbedding(embedding_dim)
        self.card_pos = PositionalEncoding(embedding_dim, max_len=7)
        self.card_attn = MultiHeadAttentionBlock(hidden_dim=embedding_dim, num_heads=8)
        self.card_flat = nn.Flatten()

        self.stage_embedding = nn.Embedding(4, 32)
        self.position_embedding = nn.Embedding(3, 32)

        card_features_dim = 7 * embedding_dim
        numeric_dim = 9
        categorical_dim = 32 + 32
        input_dim = card_features_dim + numeric_dim + categorical_dim

        self.features_net = nn.Sequential(
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

        self.context_updater = ContextUpdater(hidden_dim)

        self.main_net = nn.Sequential(
            DenseResidualBlock(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            DenseResidualBlock(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.stage_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, NUM_ACTIONS),
            )
            for _ in range(4)
        ])

        for head in self.stage_heads:
            nn.init.xavier_uniform_(head[2].weight, gain=0.1)
            nn.init.zeros_(head[2].bias)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=3e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5000, T_mult=2, eta_min=1e-5
        )

    def encode_features(self, x):
        public_cards, private_cards, stacks, _, bets, active_players_mask, stage, current_player_pos = x
        cards_mask = torch.cat([(private_cards == -1), (public_cards == -1)], dim=1)

        cards_features = self.card_attn(torch.cat([
            self.card_embedding(private_cards),
            self.card_pos(self.card_embedding(public_cards))
        ], dim=1), key_padding_mask=cards_mask)

        raw_features = self.features_net(torch.cat([
            self.card_flat(cards_features),
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
        if name is not None:
            path = f"./checkpoints/{self.name}-{name}.pth"
        else:
            path = f"./checkpoints/{self.name}.pth"
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save({
            'net': self.state_dict(),
            'opt': self.optimizer.state_dict(),
            'step': self.step,
        }, path)

    def load(self, name=None):
        if name is not None:
            dat = torch.load(f"./checkpoints/{self.name}-{name}.pth", weights_only=False)
        else:
            dat = torch.load(f"./checkpoints/{self.name}.pth", weights_only=False)
        self.load_state_dict(dat['net'])
        self.optimizer.load_state_dict(dat['opt'])
        if 'step' in dat:
            self.step = dat['step']
