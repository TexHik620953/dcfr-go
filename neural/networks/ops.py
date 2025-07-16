import torch.nn as nn
import torch
import math


class MultiHeadAttentionBlock(nn.Module):
    """Один блок Self-Attention с LayerNorm и Residual connection"""

    def __init__(self, hidden_dim=64, num_heads=8, dropout=0.05):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Time, Features]
        x = x.transpose(0, 1)  # MultiheadAttention ожидает [Time, Batch, Features]

        # Self-Attention
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        x = self.norm1(x + attn_out)  # Residual + Norm

        # Feed Forward
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)  # Residual + Norm

        return x.transpose(0, 1)  # Возвращаем к [Batch, Time, Features]


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Тензор формы [Batch, SeqLen, EmbeddingDim]
        Returns:
            Тензор с добавленными позиционными кодировками
        """
        return x + self.pe[:x.size(1)]


class DenseResidualBlock(nn.Module):
    def __init__(self, input_dim, units):
        super(DenseResidualBlock, self).__init__()
        self.dense1 = nn.Linear(input_dim, units)
        self.dense2 = nn.Linear(units, units)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(units)

        if input_dim != units:
            self.shortcut = nn.Linear(input_dim, units)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.dense1(x)
        x = self.leaky_relu(x)

        x = self.dense2(x)
        x = self.leaky_relu(x)

        x = x + shortcut
        x = self.leaky_relu(x)
        x = self.layer_norm(x)

        return x


class CardEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(CardEmbedding, self).__init__()
        self.rank = nn.Embedding(13, embedding_dim)
        self.suit = nn.Embedding(4, embedding_dim)
        self.card = nn.Embedding(52, embedding_dim)
    def forward(self, x):
        B, num_cards = x.shape
        x = x.view(-1)

        valid = x.ge(0).float()
        x = x.clamp(min=0)

        embs = self.card(x) + self.rank(x % 13) + self.suit(x // 13)
        embs = embs * valid.unsqueeze(1)
        return embs.view(B, num_cards, -1).sum(1)
