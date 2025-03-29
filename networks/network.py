import os
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ops import MultiHeadAttentionBlock



class PokerStrategyNet(nn.Module):
    def __init__(self, name, num_cards=52, embedding_dim=32, hidden_dim=128):
        super().__init__()
        self.name = name
        # Embedding слои
        self.card_embedding = nn.Embedding(num_cards + 1, embedding_dim, padding_idx=num_cards)
        self.stage_embedding = nn.Embedding(4, embedding_dim)  # PREFLOP/FLOP/TURN/RIVER
        self.position_embedding = nn.Embedding(3, embedding_dim)  # 0,1,2

        # Модули для обработки игровой информации
        self.stacks_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.LeakyReLU()
        )
        self.bets_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.LeakyReLU()
        )
        # Модуль для обработки активных игроков
        self.active_players_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.LeakyReLU()
        )

        self.embedding_attention = nn.Sequential(
            MultiHeadAttentionBlock(embedding_dim, 4),
            MultiHeadAttentionBlock(embedding_dim, 4),
            MultiHeadAttentionBlock(embedding_dim, 4),

        )

        # Основная сеть
        self.main_net = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim // 4 * 3, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 5)  # 4 действия: FOLD, CHECK/CALL, BET/RAISE, ALL-IN
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
        public_cards, private_cards, stacks, actions_mask, bets, active_players_mask, stage, current_player_pos = x
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

        public_emb = self.card_embedding(public_cards)  # (batch_size, 5, embedding_dim)
        private_emb = self.card_embedding(private_cards)
        stage_emb = self.stage_embedding(stage)
        position_emb = self.position_embedding(current_player_pos)


        emb = torch.cat((public_emb, private_emb, stage_emb, position_emb), dim=1)

        features  = self.embedding_attention(emb)
        features = features[:,0,:]


        # Игровая информация
        stacks_features = self.stacks_net(stacks)
        bets_features = self.bets_net(bets)
        active_players_features = self.active_players_net(active_players_mask.float())


        # Объединяем все фичи
        combined = torch.cat([
            features,
            stacks_features,
            bets_features,
            active_players_features,
        ], dim=1)

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