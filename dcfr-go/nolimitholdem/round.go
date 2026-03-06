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

// calcPotAfterCall returns (callAmount, totalPotAfterCall) for current player
func (h *Round) calcPotAfterCall(players []*Player) (int, int) {
	maxRaised := slices.Max(h.round_raised)
	callAmount := maxRaised - h.round_raised[h.gamePointer]
	totalPotAfterCall := callAmount
	for _, p := range players {
		totalPotAfterCall += p.InChips
	}
	return callAmount, totalPotAfterCall
}

func (h *Round) ProceedRound(players []*Player, action Action) int {
	player := players[h.gamePointer]
	maxRaised := slices.Max(h.round_raised)

	switch {
	case action == ACTION_CHECK_CALL:
		diff := maxRaised - h.round_raised[h.gamePointer]
		actualBet := min(diff, player.RemainedChips)
		h.round_raised[h.gamePointer] += actualBet
		player.Bet(actualBet)
		h.notRaiseCount++

	case action == ACTION_ALL_IN:
		allInAmount := player.RemainedChips
		h.round_raised[h.gamePointer] += allInAmount
		player.Bet(allInAmount)
		if h.round_raised[h.gamePointer] > maxRaised {
			h.notRaiseCount = 1
		} else {
			h.notRaiseCount++
		}

	case action == ACTION_FOLD:
		player.Status = PLAYERSTATUS_FOLDED

	default:
		// All raise actions use RaiseMultipliers
		mult, ok := RaiseMultipliers[action]
		if !ok {
			panic("unknown raise action")
		}
		callAmount, totalPotAfterCall := h.calcPotAfterCall(players)
		raiseAmount := totalPotAfterCall * mult[0] / mult[1]
		totalBet := callAmount + raiseAmount
		h.round_raised[h.gamePointer] += totalBet
		player.Bet(totalBet)
		h.notRaiseCount = 1
	}

	if player.RemainedChips == 0 && player.Status != PLAYERSTATUS_FOLDED {
		player.Status = PLAYERSTATUS_ALLIN
	}
	if player.Status == PLAYERSTATUS_ALLIN {
		h.notPlayingCount++
		h.notRaiseCount--
	}
	if player.Status == PLAYERSTATUS_FOLDED {
		h.notPlayingCount++
	}

	// Advance to next ACTIVE player (skip FOLDED and ALLIN)
	h.gamePointer = (h.gamePointer + 1) % h.numPlayers
	for i := 0; i < h.numPlayers; i++ {
		if players[h.gamePointer].Status == PLAYERSTATUS_ACTIVE {
			break
		}
		h.gamePointer = (h.gamePointer + 1) % h.numPlayers
	}
	return h.gamePointer
}

func (h *Round) LegalActions(players []*Player) map[Action]struct{} {
	actions := map[Action]struct{}{
		ACTION_FOLD:       {},
		ACTION_CHECK_CALL: {},
	}

	player := players[h.gamePointer]
	callAmount, totalPotAfterCall := h.calcPotAfterCall(players)

	// If call alone consumes all chips, no raises possible
	if callAmount >= player.RemainedChips {
		return actions
	}

	// All-in is always available if player has chips beyond the call
	actions[ACTION_ALL_IN] = struct{}{}

	// Add each raise size if affordable and meaningful
	for raiseAction, mult := range RaiseMultipliers {
		raiseAmount := totalPotAfterCall * mult[0] / mult[1]
		totalBet := callAmount + raiseAmount
		if totalBet <= player.RemainedChips && raiseAmount > 0 {
			actions[raiseAction] = struct{}{}
		}
	}

	return actions
}

func (h *Round) IsOver() bool {
	return h.notRaiseCount+h.notPlayingCount >= h.numPlayers
}
