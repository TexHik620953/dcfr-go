package nolimitholdem

import "slices"

type Round struct {
	numPlayers int
	bigBlind   int
	deallerId  int

	gamePointer     int
	notPlayingCount int
	notRaiseCount   int
	round_raised    []int
}

func (r *Round) DeepCopy() *Round {
	cp := &Round{
		numPlayers:      r.numPlayers,
		bigBlind:        r.bigBlind,
		deallerId:       r.deallerId,
		gamePointer:     r.gamePointer,
		notPlayingCount: r.notPlayingCount,
		notRaiseCount:   r.notRaiseCount,
	}

	if r.round_raised != nil {
		cp.round_raised = make([]int, len(r.round_raised))
		copy(cp.round_raised, r.round_raised)
	}

	return cp
}

type roundConfig struct {
	numPlayers int
	bigBlind   int
	deallerId  int
}

func newRound(cfg roundConfig) *Round {
	h := &Round{
		numPlayers: cfg.numPlayers,
		bigBlind:   cfg.bigBlind,
		deallerId:  cfg.deallerId,

		notPlayingCount: 0,
		notRaiseCount:   0,
		round_raised:    make([]int, cfg.numPlayers),
		gamePointer:     -1,
	}
	return h
}

func (h *Round) StartNewRound(gamePointer int, players []*Player) {
	h.notRaiseCount = 0
	h.gamePointer = gamePointer

	for i := range h.round_raised {
		h.round_raised[i] = players[i].InChips
	}
}

func (h *Round) ProceedRound(players []*Player, action Action) int {
	player := players[h.gamePointer]

	max_raised := slices.Max(h.round_raised)

	if action == ACTION_CHECK_CALL {
		diff := max_raised - h.round_raised[h.gamePointer]

		h.round_raised[h.gamePointer] = max_raised
		player.Bet(diff)
		h.notRaiseCount++
	} else if action == ACTION_ALL_IN {
		all_in_quantity := player.RemainedChips

		h.round_raised[h.gamePointer] += all_in_quantity
		player.Bet(all_in_quantity)
		h.notRaiseCount = 1
	} else if action == ACTION_RAISE_POT {
		total_pot := 0
		for _, p := range players {
			total_pot += p.InChips
		}

		h.round_raised[h.gamePointer] += total_pot
		player.Bet(total_pot)
		h.notRaiseCount = 1
	} else if action == ACTION_RAISE_HALFPOT {
		half_total_pot := 0
		for _, p := range players {
			half_total_pot += p.InChips
		}
		half_total_pot = half_total_pot / 2

		h.round_raised[h.gamePointer] += half_total_pot
		player.Bet(half_total_pot)
		h.notRaiseCount = 1
	} else if action == ACTION_FOLD {
		player.Status = PLAYERSTATUS_FOLDED
	}

	if player.RemainedChips == 0 && player.Status != PLAYERSTATUS_FOLDED {
		player.Status = PLAYERSTATUS_ALLIN
	}
	if player.Status == PLAYERSTATUS_ALLIN {
		h.notPlayingCount += 1
		h.notRaiseCount -= 1
	}
	if player.Status == PLAYERSTATUS_FOLDED {
		h.notPlayingCount += 1
	}

	h.gamePointer = (h.gamePointer + 1) % h.numPlayers
	for players[h.gamePointer].Status == PLAYERSTATUS_FOLDED {
		h.gamePointer = (h.gamePointer + 1) % h.numPlayers
	}
	return h.gamePointer
}

func (h *Round) LegalActions(players []*Player) map[Action]struct{} {
	// you always can fold
	avialable_actions := map[Action]struct{}{
		ACTION_FOLD:          {},
		ACTION_CHECK_CALL:    {},
		ACTION_RAISE_HALFPOT: {},
		ACTION_RAISE_POT:     {},
		ACTION_ALL_IN:        {},
	}

	player := players[h.gamePointer]

	diff := slices.Max(h.round_raised) - h.round_raised[h.gamePointer]

	// If the current player has no more chips after call, we cannot raise
	if diff > 0 && diff >= player.RemainedChips {
		delete(avialable_actions, ACTION_RAISE_HALFPOT)
		delete(avialable_actions, ACTION_RAISE_POT)
		delete(avialable_actions, ACTION_ALL_IN)
	} else {
		// Even if we can raise, we have to check remained chips
		total_pot := 0
		for _, p := range players {
			total_pot += p.InChips
		}

		if total_pot > player.RemainedChips {
			delete(avialable_actions, ACTION_RAISE_POT)
		}
		if total_pot/2 > player.RemainedChips {
			delete(avialable_actions, ACTION_RAISE_HALFPOT)
		}
		if _, ex := avialable_actions[ACTION_RAISE_HALFPOT]; ex {
			if total_pot/2+h.round_raised[h.gamePointer] <= slices.Max(h.round_raised) {
				delete(avialable_actions, ACTION_RAISE_HALFPOT)
			}
		}
	}

	return avialable_actions
}

func (h *Round) IsOver() bool {
	return h.notRaiseCount+h.notPlayingCount >= h.numPlayers
}
