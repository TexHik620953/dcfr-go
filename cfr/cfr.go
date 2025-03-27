package cfr

import (
	"dcfr-go/common/linq"
	"dcfr-go/nolimitholdem"
	"sync/atomic"
)

type CFR struct {
	coreGame *nolimitholdem.Game
	actor    nolimitholdem.Actor
}

func New(game *nolimitholdem.Game, actor nolimitholdem.Actor) *CFR {
	h := &CFR{
		coreGame: game,
		actor:    actor,
	}

	h.coreGame.Reset()

	return h
}

func (h *CFR) TraverseTree(playerId int) []float32 {
	players_probs := map[int]float32{}
	for i := range h.coreGame.PlayersCount() {
		players_probs[i] = 1
	}

	var a atomic.Int32

	payoffs := h.traverser(players_probs, playerId, &a)
	return payoffs
}

func (h *CFR) traverser(players_probs map[int]float32, playerId int, nodes_visited *atomic.Int32) []float32 {
	nodes_visited.Add(1)
	if h.coreGame.IsOver() {
		return h.coreGame.GetPayoffs()
	}

	current_player := h.coreGame.CurrentPlayer()
	state := h.coreGame.GetState(current_player)
	action_probs := h.actor.GetProbs(state)

	total_payoffs := make([]float32, h.coreGame.PlayersCount())
	action_payoffs := make(map[nolimitholdem.Action][]float32)

	// Iterate over all possible actions
	for action, action_probability := range action_probs {
		//Make a copy of original probabilities
		players_probs_copy := linq.CopyMap(players_probs)
		players_probs_copy[current_player] *= action_probability

		h.coreGame.Step(action)
		// Get result and apply it to cumulative payoffs
		internal_payoffs := h.traverser(players_probs_copy, playerId, nodes_visited)
		for i, payoff := range internal_payoffs {
			total_payoffs[i] = float32(payoff) * action_probability
		}
		action_payoffs[action] = make([]float32, len(internal_payoffs))
		copy(action_payoffs[action], internal_payoffs)

		h.coreGame.StepBack()
	}
	if current_player != playerId {
		return total_payoffs
	}

	// CFR HERE

	return total_payoffs
}
