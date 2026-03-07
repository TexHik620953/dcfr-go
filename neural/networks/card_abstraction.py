"""
Card abstraction: computes hand strength features from raw cards.
These features help the network understand card values without
learning everything from scratch via embeddings.
"""

import torch
import torch.nn as nn


def compute_card_features(private_cards, public_cards, device):
    """
    Compute hand abstraction features from cards.
    private_cards: [batch, 2] int tensor (card indices 0-51, -1 for missing)
    public_cards: [batch, 5] int tensor (card indices 0-51, -1 for missing)
    Returns: [batch, num_features] float tensor
    """
    batch = private_cards.size(0)
    features = []

    priv = private_cards.clamp(min=0)
    pub = public_cards.clamp(min=0)
    pub_mask = (public_cards >= 0)  # [batch, 5]

    priv_ranks = priv % 13  # [batch, 2]
    priv_suits = priv // 13  # [batch, 2]
    pub_ranks = pub % 13  # [batch, 5]
    pub_suits = pub // 13  # [batch, 5]

    # --- Preflop features (always available) ---

    # 1. Pocket pair (both cards same rank)
    is_pair = (priv_ranks[:, 0] == priv_ranks[:, 1]).float().unsqueeze(1)
    features.append(is_pair)

    # 2. Suited (both cards same suit)
    is_suited = (priv_suits[:, 0] == priv_suits[:, 1]).float().unsqueeze(1)
    features.append(is_suited)

    # 3. High card rank (normalized 0-1)
    high_rank = priv_ranks.max(dim=1).values.float().unsqueeze(1) / 12.0
    features.append(high_rank)

    # 4. Low card rank (normalized 0-1)
    low_rank = priv_ranks.min(dim=1).values.float().unsqueeze(1) / 12.0
    features.append(low_rank)

    # 5. Gap between cards (connectivity for straights)
    gap = (priv_ranks.max(dim=1).values - priv_ranks.min(dim=1).values).float().unsqueeze(1) / 12.0
    features.append(gap)

    # --- Postflop features (when public cards available) ---

    # All 7 cards combined
    all_ranks = torch.cat([priv_ranks, pub_ranks], dim=1)  # [batch, 7]
    all_suits = torch.cat([priv_suits, pub_suits], dim=1)  # [batch, 7]
    all_mask = torch.cat([torch.ones(batch, 2, device=device, dtype=torch.bool), pub_mask], dim=1)  # [batch, 7]

    # 6. Number of board cards (0, 3, 4, 5 normalized)
    num_public = pub_mask.sum(dim=1).float().unsqueeze(1) / 5.0
    features.append(num_public)

    # 7-10. Rank counts — how many of each rank appear (detect pairs, trips, quads)
    # Count occurrences of each rank
    rank_counts = torch.zeros(batch, 13, device=device)
    for i in range(7):
        valid = all_mask[:, i]
        rank_idx = all_ranks[:, i]
        rank_counts.scatter_add_(1, rank_idx.unsqueeze(1), valid.float().unsqueeze(1))

    has_pair = (rank_counts >= 2).any(dim=1).float().unsqueeze(1)
    has_two_pair = ((rank_counts >= 2).sum(dim=1) >= 2).float().unsqueeze(1)
    has_trips = (rank_counts >= 3).any(dim=1).float().unsqueeze(1)
    has_quads = (rank_counts >= 4).any(dim=1).float().unsqueeze(1)
    features.extend([has_pair, has_two_pair, has_trips, has_quads])

    # 11. Full house
    has_full_house = (has_trips.bool() & ((rank_counts >= 2).sum(dim=1, keepdim=True) >= 2)).float()
    features.append(has_full_house)

    # 12-13. Flush draw / flush made
    suit_counts = torch.zeros(batch, 4, device=device)
    for i in range(7):
        valid = all_mask[:, i]
        suit_idx = all_suits[:, i]
        suit_counts.scatter_add_(1, suit_idx.unsqueeze(1), valid.float().unsqueeze(1))

    max_suit_count = suit_counts.max(dim=1).values
    has_flush = (max_suit_count >= 5).float().unsqueeze(1)
    # Flush draw: 4 of same suit and not yet river (fewer than 5 public cards)
    has_flush_draw = ((max_suit_count == 4) & (pub_mask.sum(dim=1) < 5)).float().unsqueeze(1)
    features.extend([has_flush, has_flush_draw])

    # 14. Straight potential (simplified: count unique ranks in consecutive windows of 5)
    rank_present = (rank_counts > 0).float()  # [batch, 13]
    # Add ace as low (rank 12 also counts as below rank 0)
    rank_extended = torch.cat([rank_present[:, 12:13], rank_present], dim=1)  # [batch, 14]
    max_consec = torch.zeros(batch, 1, device=device)
    for start in range(10):  # windows for straights: A-5 through T-A
        window = rank_extended[:, start:start + 5]
        consec = window.sum(dim=1, keepdim=True)
        max_consec = torch.max(max_consec, consec)
    has_straight = (max_consec >= 5).float()
    straight_draw = ((max_consec == 4) & (has_straight == 0)).float()
    features.extend([has_straight, straight_draw])

    # 15. Board pairing (does a public card pair with our hole cards)
    board_pairs_hand = torch.zeros(batch, 1, device=device)
    for i in range(5):
        valid = pub_mask[:, i]
        matches = ((pub_ranks[:, i:i+1] == priv_ranks).any(dim=1) & valid).float().unsqueeze(1)
        board_pairs_hand = torch.max(board_pairs_hand, matches)
    features.append(board_pairs_hand)

    # 16. Overcards (private cards higher than all board cards)
    has_board = pub_mask.any(dim=1)  # [batch] — per-sample check
    # For preflop (no board), set max_board_rank very high so overcards = 0
    max_board_rank = torch.where(
        has_board,
        (pub_ranks * pub_mask.long()).max(dim=1).values,
        torch.tensor(12, device=device)  # higher than any rank
    )
    overcards = (priv_ranks > max_board_rank.unsqueeze(1)).sum(dim=1).float().unsqueeze(1) / 2.0
    features.append(overcards)

    return torch.cat(features, dim=1)  # [batch, 18]


CARD_FEATURE_DIM = 17


class CardAbstractionEncoder(nn.Module):
    """Lightweight MLP that processes card abstraction features."""
    def __init__(self, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CARD_FEATURE_DIM, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, card_features):
        return self.net(card_features)
