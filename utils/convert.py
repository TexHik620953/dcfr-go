
import torch
import numpy as np


def convert_pbstate_to_tensor(states, device):
    active_players_mask = np.array([s.active_players_mask for s in states])
    player_pots = np.array([s.players_pots for s in states])
    stakes = np.array([s.stakes for s in states])
    actions_mask = [list(s.legal_actions) for s in states]
    stage = np.array([s.stage for s in states])
    current_player = np.array([s.current_player for s in states])

    public_cards = [list(s.public_cards) for s in states]
    private_cards = np.array([s.private_cards for s in states])

    for i, act in enumerate(actions_mask):
        r = np.zeros(5)
        r[act] = 1.0
        actions_mask[i] = r
    actions_mask = np.array(actions_mask)

    for i, pc in enumerate(public_cards):
        public_cards[i] = pc + [-1 for i in range(5 - len(pc))]
    public_cards = np.array(public_cards)

    bank = stakes.sum(axis=1, keepdims=True) + player_pots.sum(axis=1, keepdims=True)
    stakes = stakes/bank
    player_pots = player_pots/bank


    # Формируем тензоры
    public_cards = torch.tensor(public_cards, device=device, dtype=torch.int)
    private_cards = torch.tensor(private_cards, device=device, dtype=torch.int)
    stakes = torch.tensor(stakes, device=device, dtype=torch.float32)
    actions_mask = torch.tensor(actions_mask, device=device, dtype=torch.float32)
    player_pots = torch.tensor(player_pots, device=device, dtype=torch.float32)
    active_players_mask = torch.tensor(active_players_mask, device=device, dtype=torch.float32)
    stage = torch.tensor(stage, device=device, dtype=torch.int).unsqueeze(1)
    current_player = torch.tensor(current_player, device=device, dtype=torch.int).unsqueeze(1)

    '''
            public_cards: (batch_size, 5) - индексы карт (padding = num_cards)
            private_cards: (batch_size, 2) - индексы карт
            stacks: (batch_size, 3) - стеки игроков
            bets: (batch_size, 3) - текущие ставки игроков
            active_players_mask: (batch_size, 3) - маска активных игроков
            stage: (batch_size,) - стадия игры (0-3)
            current_player_pos: (batch_size,) - позиция текущего игрока (0-2)
            current_player_pos: (batch_size,5) - маска действий 
    '''

    return (public_cards,
            private_cards,
            stakes,
            actions_mask,
            player_pots,
            active_players_mask,
            stage,
            current_player
            ), actions_mask



def convert_states_to_batch(samples, device):
    reach_prob = np.array([s.reach_prob for s in samples])
    regrets = [s.regrets for s in samples]
    strategy = [s.strategy for s in samples]
    iterations = np.array([s.iteration for s in samples])

    for i, regret in enumerate(regrets):
        r = np.zeros(5)
        for k in regret:
            r[k] = regret[k]
        regrets[i] = r
    regrets = np.array(regrets)

    for i, regret in enumerate(strategy):
        r = np.zeros(5)
        for k in regret:
            r[k] = regret[k]
        strategy[i] = r
    strategy = np.array(strategy)

    active_players_mask = np.array([s.state.active_players_mask for s in samples])
    player_pots = np.array([s.state.players_pots for s in samples])
    stakes = np.array([s.state.stakes for s in samples])
    actions_mask = [list(s.state.legal_actions) for s in samples]
    stage = np.array([s.state.stage for s in samples])
    current_player = np.array([s.state.current_player for s in samples])

    public_cards = [list(s.state.public_cards) for s in samples]
    private_cards = np.array([s.state.private_cards for s in samples])


    for i, act in enumerate(actions_mask):
        r = np.zeros(5)
        r[act] = 1.0
        actions_mask[i] = r
    actions_mask = np.array(actions_mask)

    for i, pc in enumerate(public_cards):
        public_cards[i] = pc + [-1 for i in range(5 - len(pc))]
    public_cards = np.array(public_cards)

    bank = stakes.sum(axis=1, keepdims=True) + player_pots.sum(axis=1, keepdims=True)
    stakes = stakes/bank
    player_pots = player_pots/bank



    public_cards = torch.tensor(public_cards, device=device, dtype=torch.int)
    private_cards = torch.tensor(private_cards, device=device, dtype=torch.int)
    stakes = torch.tensor(stakes, device=device, dtype=torch.float32)
    actions_mask = torch.tensor(actions_mask, device=device, dtype=torch.float32)
    player_pots = torch.tensor(player_pots, device=device, dtype=torch.float32)
    active_players_mask = torch.tensor(active_players_mask, device=device, dtype=torch.float32)
    stage = torch.tensor(stage, device=device, dtype=torch.int).unsqueeze(1)
    current_player = torch.tensor(current_player, device=device, dtype=torch.int).unsqueeze(1)


    reach_prob = torch.tensor(reach_prob, device=device, dtype=torch.float32)
    iterations = torch.tensor(iterations, device=device, dtype=torch.float32)
    regrets = torch.tensor(regrets, device=device, dtype=torch.float32)
    strategy = torch.tensor(strategy, device=device, dtype=torch.float32)


    return (public_cards,
            private_cards,
            stakes,
            actions_mask,
            player_pots,
            active_players_mask,
            stage,
            current_player
            ), (reach_prob, iterations, regrets, strategy)