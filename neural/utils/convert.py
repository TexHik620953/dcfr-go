
import torch
import numpy as np

NUM_ACTIONS = 10


def convert_pbstate_to_tensor(states, device):
    active_players_mask = np.array([s.game_state.active_players_mask for s in states])
    player_pots = np.array([s.game_state.players_pots for s in states])
    stakes = np.array([s.game_state.stakes for s in states])
    actions_mask = [list(s.game_state.legal_actions) for s in states]
    stage = np.array([s.game_state.stage for s in states])
    current_player = np.array([s.game_state.current_player for s in states])

    public_cards = [list(s.game_state.public_cards) for s in states]
    private_cards = np.array([s.game_state.private_cards for s in states])

    for i, act in enumerate(actions_mask):
        r = np.zeros(NUM_ACTIONS)
        r[act] = 1.0
        actions_mask[i] = r
    actions_mask = np.array(actions_mask)

    for i, pc in enumerate(public_cards):
        public_cards[i] = pc + [-1 for _ in range(5 - len(pc))]
    public_cards = np.array(public_cards)

    bank = stakes.sum(axis=1, keepdims=True) + player_pots.sum(axis=1, keepdims=True)
    bank = np.maximum(bank, 1e-8)
    stakes = stakes / bank
    player_pots = player_pots / bank

    # History features for transformer (list of flat float arrays or None)
    history_h = [s.lstm_context_h if len(s.lstm_context_h) > 0 else None for s in states]

    public_cards = torch.tensor(public_cards, device=device, dtype=torch.int)
    private_cards = torch.tensor(private_cards, device=device, dtype=torch.int)
    stakes = torch.tensor(stakes, device=device, dtype=torch.float32)
    actions_mask = torch.tensor(actions_mask, device=device, dtype=torch.float32)
    player_pots = torch.tensor(player_pots, device=device, dtype=torch.float32)
    active_players_mask = torch.tensor(active_players_mask, device=device, dtype=torch.float32)
    stage = torch.tensor(stage, device=device, dtype=torch.int).unsqueeze(1)
    current_player = torch.tensor(current_player, device=device, dtype=torch.int).unsqueeze(1)

    return (public_cards,
            private_cards,
            stakes,
            actions_mask,
            player_pots,
            active_players_mask,
            stage,
            current_player
            ), actions_mask, history_h


def convert_states_to_batch(samples, device):
    regrets = [s.regrets for s in samples]
    iterations = np.array([s.iteration for s in samples])

    for i, regret in enumerate(regrets):
        r = np.zeros(NUM_ACTIONS)
        for k in regret:
            r[k] = regret[k]
        regrets[i] = r
    regrets = np.array(regrets)

    active_players_mask = np.array([s.game_state.active_players_mask for s in samples])
    player_pots = np.array([s.game_state.players_pots for s in samples])
    stakes = np.array([s.game_state.stakes for s in samples])
    actions_mask = [list(s.game_state.legal_actions) for s in samples]
    stage = np.array([s.game_state.stage for s in samples])
    current_player = np.array([s.game_state.current_player for s in samples])

    public_cards = [list(s.game_state.public_cards) for s in samples]
    private_cards = np.array([s.game_state.private_cards for s in samples])

    for i, act in enumerate(actions_mask):
        r = np.zeros(NUM_ACTIONS)
        r[act] = 1.0
        actions_mask[i] = r
    actions_mask = np.array(actions_mask)

    for i, pc in enumerate(public_cards):
        public_cards[i] = pc + [-1 for _ in range(5 - len(pc))]
    public_cards = np.array(public_cards)

    bank = stakes.sum(axis=1, keepdims=True) + player_pots.sum(axis=1, keepdims=True)
    bank = np.maximum(bank, 1e-8)
    stakes = stakes / bank
    player_pots = player_pots / bank

    public_cards = torch.tensor(public_cards, device=device, dtype=torch.int)
    private_cards = torch.tensor(private_cards, device=device, dtype=torch.int)
    stakes = torch.tensor(stakes, device=device, dtype=torch.float32)
    actions_mask = torch.tensor(actions_mask, device=device, dtype=torch.float32)
    player_pots = torch.tensor(player_pots, device=device, dtype=torch.float32)
    active_players_mask = torch.tensor(active_players_mask, device=device, dtype=torch.float32)
    stage = torch.tensor(stage, device=device, dtype=torch.int).unsqueeze(1)
    current_player = torch.tensor(current_player, device=device, dtype=torch.int).unsqueeze(1)

    iterations = torch.tensor(iterations, device=device, dtype=torch.float32)
    regrets = torch.tensor(regrets, device=device, dtype=torch.float32)

    return (public_cards,
            private_cards,
            stakes,
            actions_mask,
            player_pots,
            active_players_mask,
            stage,
            current_player
            ), (iterations, regrets)


def convert_strategy_states_to_batch(samples, device):
    """Convert strategy samples (for average strategy network training).
    Returns: (state_tensors, (iterations, strategy_targets))
    """
    strategies = [s.strategy for s in samples]
    iterations = np.array([s.iteration for s in samples])

    for i, strat in enumerate(strategies):
        r = np.zeros(NUM_ACTIONS)
        for k in strat:
            r[k] = strat[k]
        strategies[i] = r
    strategies = np.array(strategies)

    active_players_mask = np.array([s.game_state.active_players_mask for s in samples])
    player_pots = np.array([s.game_state.players_pots for s in samples])
    stakes = np.array([s.game_state.stakes for s in samples])
    actions_mask = [list(s.game_state.legal_actions) for s in samples]
    stage = np.array([s.game_state.stage for s in samples])
    current_player = np.array([s.game_state.current_player for s in samples])

    public_cards = [list(s.game_state.public_cards) for s in samples]
    private_cards = np.array([s.game_state.private_cards for s in samples])

    for i, act in enumerate(actions_mask):
        r = np.zeros(NUM_ACTIONS)
        r[act] = 1.0
        actions_mask[i] = r
    actions_mask = np.array(actions_mask)

    for i, pc in enumerate(public_cards):
        public_cards[i] = pc + [-1 for _ in range(5 - len(pc))]
    public_cards = np.array(public_cards)

    bank = stakes.sum(axis=1, keepdims=True) + player_pots.sum(axis=1, keepdims=True)
    bank = np.maximum(bank, 1e-8)
    stakes = stakes / bank
    player_pots = player_pots / bank

    public_cards = torch.tensor(public_cards, device=device, dtype=torch.int)
    private_cards = torch.tensor(private_cards, device=device, dtype=torch.int)
    stakes = torch.tensor(stakes, device=device, dtype=torch.float32)
    actions_mask = torch.tensor(actions_mask, device=device, dtype=torch.float32)
    player_pots = torch.tensor(player_pots, device=device, dtype=torch.float32)
    active_players_mask = torch.tensor(active_players_mask, device=device, dtype=torch.float32)
    stage = torch.tensor(stage, device=device, dtype=torch.int).unsqueeze(1)
    current_player = torch.tensor(current_player, device=device, dtype=torch.int).unsqueeze(1)

    iterations = torch.tensor(iterations, device=device, dtype=torch.float32)
    strategies = torch.tensor(strategies, device=device, dtype=torch.float32)

    return (public_cards,
            private_cards,
            stakes,
            actions_mask,
            player_pots,
            active_players_mask,
            stage,
            current_player
            ), (iterations, strategies)
