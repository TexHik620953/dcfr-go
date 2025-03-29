
import torch
import numpy as np


def convert_pbstate_to_tensor(state, device):
    active_players_mask = list(state.active_players_mask)
    player_pots = list(state.players_pots)
    stakes = list(state.stakes)
    legal_actions = list(state.legal_actions)
    stage = state.stage
    current_player = state.current_player

    public_cards = list(state.public_cards)
    private_cards = list(state.private_cards)

    # Создаем маску для действий
    actions_mask = np.zeros(5)
    actions_mask[legal_actions] = 1.0

    # Нормализуем стеки и ставки
    total_bank = np.sum(stakes) + np.sum(player_pots)
    stakes = np.array(stakes) / total_bank
    player_pots = np.array(player_pots) / total_bank

    # Добавляем признак пустоты в общие карты
    public_cards = public_cards + [52 for i in range(5 - len(public_cards))]

    # Формируем тензоры
    public_cards = torch.tensor(public_cards, device=device, dtype=torch.int).unsqueeze(0)
    private_cards = torch.tensor(private_cards, device=device, dtype=torch.int).unsqueeze(0)
    stakes = torch.tensor(stakes, device=device, dtype=torch.float32).unsqueeze(0)
    actions_mask = torch.tensor(actions_mask, device=device, dtype=torch.float32).unsqueeze(0)
    player_pots = torch.tensor(player_pots, device=device, dtype=torch.float32).unsqueeze(0)
    active_players_mask = torch.tensor(active_players_mask, device=device, dtype=torch.int).unsqueeze(0)
    stage = torch.tensor(stage, device=device, dtype=torch.int).unsqueeze(0).unsqueeze(1)
    current_player = torch.tensor(current_player, device=device, dtype=torch.int).unsqueeze(0).unsqueeze(1)

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
    weights = np.array([s.weight for s in samples])
    regrets = [s.regrets for s in samples]
    iterations = np.array([s.iteration for s in samples])

    for i, regret in enumerate(regrets):
        r = np.zeros(5)
        for k in regret:
            r[k] = regret[k]
        regrets[i] = r
    regrets = np.array(regrets)

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
        public_cards[i] = pc + [52 for i in range(5 - len(pc))]
    public_cards = np.array(public_cards)

    bank = stakes.sum(axis=1, keepdims=True) + player_pots.sum(axis=1, keepdims=True)
    stakes = stakes/bank
    player_pots = player_pots/bank



    public_cards = torch.tensor(public_cards, device=device, dtype=torch.int)
    private_cards = torch.tensor(private_cards, device=device, dtype=torch.int)
    stakes = torch.tensor(stakes, device=device, dtype=torch.float32)
    actions_mask = torch.tensor(actions_mask, device=device, dtype=torch.float32)
    player_pots = torch.tensor(player_pots, device=device, dtype=torch.float32)
    active_players_mask = torch.tensor(active_players_mask, device=device, dtype=torch.int)
    stage = torch.tensor(stage, device=device, dtype=torch.int).unsqueeze(1)
    current_player = torch.tensor(current_player, device=device, dtype=torch.int).unsqueeze(1)


    weights = torch.tensor(weights, device=device, dtype=torch.float32)
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
            ), (weights, iterations, regrets)