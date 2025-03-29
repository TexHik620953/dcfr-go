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

