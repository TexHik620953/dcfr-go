
import torch
import numpy as np

NUM_ACTIONS = 10


def convert_pbstate_to_tensor(states, device):
    n = len(states)

    # Pre-allocate numpy arrays
    active_players_mask = np.empty((n, 3), dtype=np.float32)
    player_pots = np.empty((n, 3), dtype=np.float32)
    stakes = np.empty((n, 3), dtype=np.float32)
    actions_mask = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
    stage = np.empty((n, 1), dtype=np.int32)
    current_player = np.empty((n, 1), dtype=np.int32)
    public_cards = np.full((n, 5), -1, dtype=np.int32)
    private_cards = np.empty((n, 2), dtype=np.int32)

    history_h = [None] * n

    for i, s in enumerate(states):
        gs = s.game_state
        active_players_mask[i] = gs.active_players_mask
        player_pots[i] = gs.players_pots
        stakes[i] = gs.stakes
        stage[i, 0] = gs.stage
        current_player[i, 0] = gs.current_player

        for a in gs.legal_actions:
            actions_mask[i, a] = 1.0

        pc = gs.public_cards
        for j in range(len(pc)):
            public_cards[i, j] = pc[j]

        private_cards[i, 0] = gs.private_cards[0]
        private_cards[i, 1] = gs.private_cards[1]

        if len(s.lstm_context_h) > 0:
            history_h[i] = s.lstm_context_h

    bank = stakes.sum(axis=1, keepdims=True) + player_pots.sum(axis=1, keepdims=True)
    bank = np.maximum(bank, 1e-8)
    stakes /= bank
    player_pots /= bank

    public_cards = torch.as_tensor(public_cards, device=device)
    private_cards = torch.as_tensor(private_cards, device=device)
    stakes = torch.as_tensor(stakes, device=device)
    actions_mask_t = torch.as_tensor(actions_mask, device=device)
    player_pots = torch.as_tensor(player_pots, device=device)
    active_players_mask = torch.as_tensor(active_players_mask, device=device)
    stage = torch.as_tensor(stage, device=device)
    current_player = torch.as_tensor(current_player, device=device)

    return (public_cards,
            private_cards,
            stakes,
            actions_mask_t,
            player_pots,
            active_players_mask,
            stage,
            current_player
            ), actions_mask_t, history_h


def _convert_common(samples, device):
    """Shared conversion logic for game state fields."""
    n = len(samples)

    active_players_mask = np.empty((n, 3), dtype=np.float32)
    player_pots = np.empty((n, 3), dtype=np.float32)
    stakes = np.empty((n, 3), dtype=np.float32)
    actions_mask = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
    stage = np.empty((n, 1), dtype=np.int32)
    current_player = np.empty((n, 1), dtype=np.int32)
    public_cards = np.full((n, 5), -1, dtype=np.int32)
    private_cards = np.empty((n, 2), dtype=np.int32)

    for i, s in enumerate(samples):
        gs = s.game_state
        active_players_mask[i] = gs.active_players_mask
        player_pots[i] = gs.players_pots
        stakes[i] = gs.stakes
        stage[i, 0] = gs.stage
        current_player[i, 0] = gs.current_player

        for a in gs.legal_actions:
            actions_mask[i, a] = 1.0

        pc = gs.public_cards
        for j in range(len(pc)):
            public_cards[i, j] = pc[j]

        private_cards[i, 0] = gs.private_cards[0]
        private_cards[i, 1] = gs.private_cards[1]

    bank = stakes.sum(axis=1, keepdims=True) + player_pots.sum(axis=1, keepdims=True)
    bank = np.maximum(bank, 1e-8)
    stakes /= bank
    player_pots /= bank

    public_cards = torch.as_tensor(public_cards, device=device)
    private_cards = torch.as_tensor(private_cards, device=device)
    stakes = torch.as_tensor(stakes, device=device)
    actions_mask = torch.as_tensor(actions_mask, device=device)
    player_pots = torch.as_tensor(player_pots, device=device)
    active_players_mask = torch.as_tensor(active_players_mask, device=device)
    stage = torch.as_tensor(stage, device=device)
    current_player = torch.as_tensor(current_player, device=device)

    return (public_cards, private_cards, stakes, actions_mask,
            player_pots, active_players_mask, stage, current_player)


def convert_states_to_batch(samples, device):
    n = len(samples)

    regrets = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
    iterations = np.empty(n, dtype=np.float32)

    for i, s in enumerate(samples):
        iterations[i] = s.iteration
        for k, v in s.regrets.items():
            regrets[i, k] = v

    state_tensors = _convert_common(samples, device)

    iterations = torch.as_tensor(iterations, device=device)
    regrets = torch.as_tensor(regrets, device=device)

    return state_tensors, (iterations, regrets)


def convert_strategy_states_to_batch(samples, device):
    n = len(samples)

    strategies = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
    iterations = np.empty(n, dtype=np.float32)

    for i, s in enumerate(samples):
        iterations[i] = s.iteration
        for k, v in s.strategy.items():
            strategies[i, k] = v

    state_tensors = _convert_common(samples, device)

    iterations = torch.as_tensor(iterations, device=device)
    strategies = torch.as_tensor(strategies, device=device)

    return state_tensors, (iterations, strategies)
