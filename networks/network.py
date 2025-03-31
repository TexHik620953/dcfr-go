import os
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ops import MultiHeadAttentionBlock, PositionalEncoding, DenseResidualBlock



class PokerStrategyNet(nn.Module):
    def __init__(self, name, num_cards=52, embedding_dim=64, hidden_dim=256):
        super().__init__()
        self.name = name

        # Embeddings
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=7)
        self.card_embedding = nn.Embedding(num_cards + 1, embedding_dim, padding_idx=num_cards)
        self.stage_embedding = nn.Embedding(4, embedding_dim)
        self.position_embedding = nn.Embedding(3, embedding_dim)

        # Attention для карт
        self.card_attention = nn.Sequential(
            MultiHeadAttentionBlock(embedding_dim, num_heads=4),
            MultiHeadAttentionBlock(embedding_dim, num_heads=4),
            MultiHeadAttentionBlock(embedding_dim, num_heads=4),
            MultiHeadAttentionBlock(embedding_dim, num_heads=4),
        )


        # Обработка стеков
        self.stacks_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.LeakyReLU()
        )
        # Обработка ставок
        self.bets_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.LeakyReLU()
        )
        # Обработка ставок
        self.ply_mask_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.LeakyReLU()
        )

        # Основная сеть
        self.main_net = nn.Sequential(
            DenseResidualBlock(embedding_dim * 4 + hidden_dim // 8 * 3, hidden_dim),
            DenseResidualBlock(hidden_dim, hidden_dim),
            DenseResidualBlock(hidden_dim, hidden_dim),
            DenseResidualBlock(hidden_dim, hidden_dim),
            DenseResidualBlock(hidden_dim, hidden_dim),
            DenseResidualBlock(hidden_dim, hidden_dim),
            DenseResidualBlock(hidden_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 5)
        )

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        for module in [self.card_embedding, self.stage_embedding, self.position_embedding]:
            nn.init.normal_(module.weight, mean=0, std=0.05)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

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
        public_emb = self.card_embedding(public_cards)  # [Batch, 5, Embedding]
        private_emb = self.card_embedding(private_cards)  # [Batch, 2, Embedding]
        card_emb = torch.cat((private_emb, public_emb), dim=1)  # [Batch, 7, Embedding]
        card_emb = self.pos_encoder(card_emb)  # Добавляем позиционные кодировки для публичных карт

        card_attns = self.card_attention(card_emb)
        card_features = torch.cat([
            card_attns.mean(dim=1),
            card_attns.max(dim=1).values
        ], dim=1)

        meta_emb = torch.cat((
            self.stacks_net(stacks),
            self.bets_net(bets),
            self.ply_mask_net(active_players_mask),
            self.stage_embedding(stage.squeeze(1)),
            self.position_embedding(current_player_pos.squeeze(1))
        ), dim=1)

        # Объединение
        combined = torch.cat((
            card_features,
            meta_emb
        ), dim=1)

        # Основная сеть
        logits = self.main_net(combined)

        return logits

    @torch.no_grad()
    def get_probs(self, x, actions_mask):
        # Epsilon-greedy
        logits = self.forward(x)
        # Выбираем действие
        probs = torch.softmax(logits, dim=1)
        # Зануляем недопустимые действия
        probs = probs * actions_mask
        # Если действий не осталось, выбираем случайное
        if probs.sum() == 0:
            probs = actions_mask / actions_mask.sum(dim=1, keepdim=True)
        # Нормализуем
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def save(self):
        os.makedirs(f"./checkpoints", exist_ok=True)
        torch.save({'net': self.state_dict()}, f"./checkpoints/{self.name}.pth")

    def load(self):
        dat = torch.load(f"./checkpoints/{self.name}.pth")
        self.load_state_dict(dat['net'])