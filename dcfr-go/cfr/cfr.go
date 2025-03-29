package cfr

import (
	"dcfr-go/common/random"
	"dcfr-go/nolimitholdem"
	"math/rand"
	"sync/atomic"
	"unsafe"
)

type CFR struct {
	coreGame *nolimitholdem.Game
	actor    CFRActor
	Memory   *MemoryBuffer

	stats *CFRStats

	rng *rand.Rand
}
type CFRStats struct {
	NodesVisited   atomic.Int32
	TreesTraversed atomic.Int32
}

func New(seed int64, game *nolimitholdem.Game, actor CFRActor, memory *MemoryBuffer, stats *CFRStats) *CFR {
	h := &CFR{
		coreGame: game,
		actor:    actor,
		Memory:   memory,
		stats:    stats,
		rng:      rand.New(rand.NewSource(seed)),
	}

	h.coreGame.Reset()

	return h
}

func (h *CFR) TraverseTree(playerId int) ([]float32, error) {
	// Инициализация вероятностей достижения состояния
	reachProbs := make([]float32, h.coreGame.PlayersCount())
	for i := range reachProbs {
		reachProbs[i] = 1.0
	}

	payoffs, err := h.traverser(reachProbs, playerId)
	if err != nil {
		return nil, err
	}
	h.stats.TreesTraversed.Add(1)
	return payoffs, nil
}

func (h *CFR) traverser(reachProbs []float32, learnerId int) ([]float32, error) {
	h.stats.NodesVisited.Add(1)

	if h.coreGame.IsOver() {
		return h.coreGame.GetPayoffs(), nil
	}

	currentPlayer := h.coreGame.CurrentPlayer()
	state := h.coreGame.GetState(currentPlayer)
	actionProbs, err := h.actor.GetProbs(learnerId, state)
	if err != nil {
		return nil, err
	}

	totalPayoffs := make([]float32, h.coreGame.PlayersCount())

	// For opponent, use only one action
	if currentPlayer != learnerId {
		action := nolimitholdem.Action(random.Sample(h.rng, *(*map[int32]float32)(unsafe.Pointer(&actionProbs))))
		newReachProbs := make([]float32, len(reachProbs))
		copy(newReachProbs, reachProbs)
		newReachProbs[currentPlayer] *= actionProbs[action]

		h.coreGame.Step(action)
		childPayoffs, err := h.traverser(newReachProbs, learnerId)
		if err != nil {
			return nil, err
		}
		h.coreGame.StepBack()
		for i, payoff := range childPayoffs {
			childPayoffs[i] = payoff * actionProbs[action]
		}
		return childPayoffs, nil // Возвращаем payoffs только для сэмплированного действия
	}

	// Iterate over all possible actions
	actionPayoffs := make(map[nolimitholdem.Action][]float32)
	for action, action_probability := range actionProbs {
		//Make a copy of original probabilities
		newReachProbs := make([]float32, len(reachProbs))
		copy(newReachProbs, reachProbs)
		newReachProbs[currentPlayer] *= action_probability

		h.coreGame.Step(action)
		childPayoffs, err := h.traverser(newReachProbs, learnerId)
		if err != nil {
			return nil, err
		}
		h.coreGame.StepBack()

		// Сохраняем результаты
		actionPayoffs[action] = childPayoffs
		for i, payoff := range childPayoffs {
			totalPayoffs[i] += float32(payoff) * action_probability
		}
	}

	if currentPlayer != learnerId {
		return totalPayoffs, nil
	}

	// CFR HERE
	regrets := make(nolimitholdem.Strategy)
	for action, payoffs := range actionPayoffs {
		regret := payoffs[learnerId] - totalPayoffs[learnerId]
		// CFR+: только положительные сожаления
		if regret > 0 {
			regrets[action] = regret
		}
	}
	if len(regrets) > 0 {
		h.Memory.AddSample(learnerId, state, regrets, reachProbs[learnerId], 0)
	}

	return totalPayoffs, nil
}
