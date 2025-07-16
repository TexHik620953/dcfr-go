import os
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ops import MultiHeadAttentionBlock, PositionalEncoding, DenseResidualBlock, CardEmbedding


class DeepCFRModel(nn.Module):
    def __init__(self, name, lr=1e-3, embedding_dim=64, hidden_dim=128):
        super(DeepCFRModel, self).__init__()
        self.name = name
        self.step = 0

        self.hole_embedding = CardEmbedding(embedding_dim)  # For hole cards
        self.public_embedding = CardEmbedding(embedding_dim)  # For public cards
        self.stage_embedding = nn.Embedding(4, 4)

        self.position_embedding = nn.Embedding(3, 3)

        self.card_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.features_net = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.main_net = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.action_head = nn.Linear(hidden_dim, 5)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=2e-5)

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

        cards_features = self.card_net(torch.cat([
            self.public_embedding(public_cards),
            self.hole_embedding(private_cards)
        ], dim=1))

        raw_features = self.features_net(torch.cat([
            stacks,
            bets,
            active_players_mask,
            self.stage_embedding(stage)[:,0,:],
            self.position_embedding(current_player_pos)[:,0,:]
        ], dim=1))

        features = self.main_net(torch.cat([cards_features, raw_features], dim=1))

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

    def save(self):
        os.makedirs(f"./checkpoints", exist_ok=True)
        torch.save({
            'net': self.state_dict(),
            'opt': self.optimizer.state_dict(),
        }, f"./checkpoints/{self.name}.pth")

    def load(self):
        dat = torch.load(f"./checkpoints/{self.name}.pth")
        self.load_state_dict(dat['net'])
        self.optimizer.load_state_dict(dat['opt'])
