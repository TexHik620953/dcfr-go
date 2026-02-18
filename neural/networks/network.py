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

        # Задача - закодировать состояние игры в латентное представление.
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

        # Задача - подмешивание контекста истории в латентное представление
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # размер выхода features_net
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        # Init context for lstm
        self.init_lstm = nn.Parameter(torch.ones(2, 1, hidden_dim))

        # Задача - предсказать действия из латентного представления.
        self.main_net = nn.Sequential(
            DenseResidualBlock(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            DenseResidualBlock(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            DenseResidualBlock(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )

        self.action_head = nn.Linear(hidden_dim, 5)

        nn.init.xavier_uniform_(self.action_head.weight, gain=1/math.sqrt(hidden_dim))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=3e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5000, T_mult=2, eta_min=1e-5
        )


    def encode_features(self, x):
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
        public_cards, private_cards, stacks, _, bets, active_players_mask, stage, current_player_pos = x
        cards_mask = torch.cat([(private_cards == -1),(public_cards == -1)], dim=1)

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
    def process_features(self, features, lstm_context=None):
        # Подготавливаем контекст lstm
        lstm_h = None
        lstm_c = None
        if lstm_context is None:
            # Here code if all context is none
            batch_size = features.size(0)
            lstm_h = self.init_lstm.expand(2, batch_size, -1).contiguous()  # (2, batch, hidden_dim)
            lstm_c = torch.zeros_like(lstm_h)
        else:
            (lstm_h, lstm_c) = lstm_context
            lstm_h = lstm_h.contiguous()
            lstm_c = lstm_c.contiguous()

        raw_features = features.unsqueeze(1)
        # Пропускаем через LSTM
        lstm_out, new_context = self.lstm(raw_features, (lstm_h, lstm_c))
        lstm_out = lstm_out.squeeze(1)  # [batch, hidden_dim]

        features = self.main_net(lstm_out)
        action_logits = self.action_head(features)
        return action_logits, new_context

    def forward(self, x, lstm_context):
        raw_features = self.encode_features(x)
        return self.process_features(raw_features, lstm_context)

    @torch.no_grad()
    def get_probs(self, x, actions_mask, lstm_context=None):
        device = next(self.parameters()).device
        if lstm_context is not None:
            raw_lstm_h, raw_lstm_c = lstm_context
            lstm_h = []
            lstm_c = []
            for i in range(len(raw_lstm_h)):
                if raw_lstm_h[i] is None or raw_lstm_c[i] is None:
                    h = self.init_lstm.data.expand(2, 1, -1).contiguous()
                    c = torch.zeros_like(h)
                else:
                    # Восстанавливаем из плоского массива
                    h_flat = raw_lstm_h[i]  # размер: (2*hidden_dim,)
                    c_flat = raw_lstm_c[i]  # размер: (2*hidden_dim,)

                    # Преобразуем в тензор и разделяем на 2 слоя
                    h_tensor = torch.tensor(h_flat, device=device)  # (2*hidden_dim,)
                    c_tensor = torch.tensor(c_flat, device=device)
                    # Reshape: (2, hidden_dim) -> (2, 1, hidden_dim) для batch_size=1
                    h = h_tensor.view(2, -1).unsqueeze(1)  # (2, hidden_dim) -> (2, 1, hidden_dim)
                    c = c_tensor.view(2, -1).unsqueeze(1)  # (2, hidden_dim) -> (2, 1, hidden_dim)
                lstm_h.append(h)
                lstm_c.append(c)
                # Стек: (batch, num_layers, hidden_dim) -> (num_layers, batch, hidden_dim)
            lstm_c = torch.hstack(lstm_c)  # (num_layers, batch, hidden_dim)
            lstm_h = torch.hstack(lstm_h)  # (num_layers, batch, hidden_dim)
            lstm_context = (lstm_h, lstm_c)

        raw_features = self.encode_features(x)

        # Epsilon-greedy
        logits, new_context = self.process_features(raw_features, lstm_context)

        # Выбираем действие
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
