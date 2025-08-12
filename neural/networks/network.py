import os
from random import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ops import MultiHeadAttentionBlock, PositionalEncoding, DenseResidualBlock, CardEmbedding, ScaledLinear


class DeepCFRModel(nn.Module):
    def __init__(self, name, lr=1e-3, embedding_dim=128, hidden_dim=128):
        super(DeepCFRModel, self).__init__()
        self.name = name
        self.step = 0

        self.card_embedding = CardEmbedding(embedding_dim)  # For cards
        self.card_pos = PositionalEncoding(embedding_dim, max_len=5)

        self.card_attn = MultiHeadAttentionBlock(hidden_dim=embedding_dim, num_heads=8)
        self.card_flat = nn.Flatten()

        self.stage_embedding = nn.Embedding(4, 32)
        self.position_embedding = nn.Embedding(3, 32)

        self.features_net = nn.Sequential(
            nn.Linear(7 * embedding_dim + 9 + 32 + 32, 6 * embedding_dim),
            nn.LayerNorm(6 * embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(6 * embedding_dim, 6 * embedding_dim),
            nn.LayerNorm(6 * embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(6 * embedding_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2)
        )

        self.main_net = nn.Sequential(
            DenseResidualBlock(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            DenseResidualBlock(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            DenseResidualBlock(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.action_head = nn.Linear(hidden_dim, 5)

        nn.init.xavier_uniform_(self.action_head.weight, gain=1/math.sqrt(hidden_dim))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=3e-5)

    def forward(self, x):
        public_cards, private_cards, stacks, _, bets, active_players_mask, stage, current_player_pos = x
        """
        Args:
            public_cards: (batch_size, 5) - индексы карт (padding = num_cards)
            private_cards: (batch_size, 2) - индексы карт
            stacks: (batch_size, 3) - стеки игроков
            bets: (batch_size, 3) - текущие ставки игроков
            active_players_mask: (batch_size, 3) - маска активных игроков
            stage: (batch_size,1) - стадия игры (0-3)
            current_player_pos: (batch_size,1) - позиция текущего игрока (0-2)
        """
        private_features = self.card_embedding(private_cards)
        public_features = self.card_pos(self.card_embedding(public_cards))

        cards_mask = torch.cat([(private_cards == -1),(public_cards == -1)], dim=1)

        cards_features = self.card_attn(torch.cat([
            private_features,
            public_features
        ], dim=1), key_padding_mask=cards_mask)

        cards_features = self.card_flat(cards_features)

        stage_features = self.stage_embedding(stage)[:, 0, :]
        position_features = self.position_embedding(current_player_pos)[:, 0, :]

        raw_features = torch.cat([
            cards_features,
            stacks,
            bets,
            active_players_mask,
            stage_features,
            position_features
        ], dim=1)

        raw_features = self.features_net(raw_features)

        features = self.main_net(raw_features)

        return self.action_head(features)

    @torch.no_grad()
    def get_probs(self, x, actions_mask):
        # Epsilon-greedy
        logits = self.forward(x)

        # Выбираем действие
        probs = torch.softmax(logits, dim=1)
        probs = probs * actions_mask

        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def save(self, name=None):
        if name is not None:
            path = f"./checkpoints/{self.name}-{name}.pth"
        else:
            path = f"./checkpoints/{self.name}.pth"

        os.makedirs(f"./checkpoints", exist_ok=True)
        torch.save({
            'net': self.state_dict(),
            'opt': self.optimizer.state_dict(),
        }, path)

    def load(self, name=None):
        if name is not None:
            dat = torch.load(f"./checkpoints/{self.name}-{name}.pth")
        else:
            dat = torch.load(f"./checkpoints/{self.name}.pth")
        self.load_state_dict(dat['net'])
        self.optimizer.load_state_dict(dat['opt'])
