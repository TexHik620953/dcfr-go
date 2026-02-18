import torch.nn as nn
import torch
import math
import torch.nn.functional as F

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

    def forward(self, x, key_padding_mask=None):
        # x: [Batch, Time, Features]
        x = x.transpose(0, 1)  # MultiheadAttention ожидает [Time, Batch, Features]

        # Self-Attention
        attn_out, _ = self.attention(x, x, x,key_padding_mask=key_padding_mask)
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
    def __init__(self, input_dim, units, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, units)
        self.norm2 = nn.LayerNorm(units)
        self.linear2 = nn.Linear(units, units)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(0.2)

        if input_dim != units:
            self.shortcut = nn.Linear(input_dim, units)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        # Pre-activation: norm -> activation -> linear
        h = self.norm1(x)
        h = self.activation(h)
        h = self.linear1(h)

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)

        # Сложение с shortcut
        return h + shortcut


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
        embs = F.layer_norm(embs, [embs.size(-1)])
        embs = embs * valid.unsqueeze(1)
        return embs.view(B, num_cards, -1)


class ScaledLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (in_features ** -0.5)  # 1/sqrt(hidden_dim)

    def forward(self, x):
        return self.linear(x) * self.scale
