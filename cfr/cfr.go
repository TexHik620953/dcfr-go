package cfr

import (
	"dcfr-go/nolimitholdem"
	"sync/atomic"
)

type CFR struct {
	coreGame  *nolimitholdem.Game
	actor     nolimitholdem.Actor
	memory    *MemoryBuffer
	iteration int
}

func New(game *nolimitholdem.Game, actor nolimitholdem.Actor, memory *MemoryBuffer) *CFR {
	h := &CFR{
		coreGame:  game,
		actor:     actor,
		memory:    memory,
		iteration: 0,
	}

	h.coreGame.Reset()

	return h
}

func (h *CFR) TraverseTree(playerId int) []float32 {
	// Инициализация вероятностей достижения состояния
	reachProbs := make([]float32, h.coreGame.PlayersCount())
	for i := range reachProbs {
		reachProbs[i] = 1.0
	}

	var nodesVisited atomic.Int32

	payoffs := h.traverser(reachProbs, playerId, &nodesVisited)

	h.iteration++

	return payoffs
}

func (h *CFR) traverser(reachProbs []float32, playerId int, nodesVisited *atomic.Int32) []float32 {
	nodesVisited.Add(1)
	if h.coreGame.IsOver() {
		return h.coreGame.GetPayoffs()
	}

	currentPlayer := h.coreGame.CurrentPlayer()
	state := h.coreGame.GetState(currentPlayer)
	actionProbs := h.actor.GetProbs(state)

	totalPayoffs := make([]float32, h.coreGame.PlayersCount())
	actionPayoffs := make(map[nolimitholdem.Action][]float32)

	// Iterate over all possible actions
	for action, action_probability := range actionProbs {
		//Make a copy of original probabilities
		newReachProbs := make([]float32, len(reachProbs))
		copy(newReachProbs, reachProbs)
		newReachProbs[currentPlayer] *= action_probability

		h.coreGame.Step(action)
		childPayoffs := h.traverser(newReachProbs, playerId, nodesVisited)
		h.coreGame.StepBack()

		// Сохраняем результаты
		actionPayoffs[action] = childPayoffs
		for i, payoff := range childPayoffs {
			totalPayoffs[i] = float32(payoff) * action_probability
		}
	}

	if currentPlayer != playerId {
		return totalPayoffs
	}

	// CFR HERE
	regrets := make(map[nolimitholdem.Action]float32)
	for action, payoffs := range actionPayoffs {
		regret := payoffs[playerId] - totalPayoffs[playerId]
		// CFR+: только положительные сожаления
		if regret > 0 {
			// Учитываем вероятность достижения оппонентами (product всех reachProbs кроме текущего игрока)
			oppReach := float32(1.0)
			for i, prob := range reachProbs {
				if i != playerId {
					oppReach *= prob
				}
			}
			regrets[action] = regret * oppReach
		}
	}
	h.memory.AddSample(playerId, state, regrets, reachProbs[playerId], h.iteration)

	return totalPayoffs
}
