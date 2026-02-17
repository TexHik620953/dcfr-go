package cfr

import (
	"dcfr-go/common/random"
	"dcfr-go/nolimitholdem"
	"fmt"
	"math/rand"
	"sync/atomic"
)

type CFR struct {
	coreGame *nolimitholdem.Game
	actor    CFRActor
	Memory   IMemoryBuffer

	stats *CFRStats

	rng *rand.Rand
}
type CFRStats struct {
	NodesVisited   atomic.Int32
	TreesTraversed atomic.Int32
}

func New(seed int64, game *nolimitholdem.Game, actor CFRActor, memory IMemoryBuffer, stats *CFRStats) *CFR {
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

func (h *CFR) TraverseTree(learnerId int, cfr_it int) ([]float32, error) {
	h.coreGame.Reset()
	payoffs, err := h.traverser(learnerId, cfr_it)
	if err != nil {
		return nil, err
	}
	h.stats.TreesTraversed.Add(1)
	return payoffs, nil
}

func (h *CFR) traverser(learnerId int, cfr_it int) ([]float32, error) {
	h.stats.NodesVisited.Add(1)

	if h.coreGame.IsOver() {
		return h.coreGame.GetPayoffs(), nil
	}

	currentPlayer := h.coreGame.CurrentPlayer()
	state := h.coreGame.GetState(currentPlayer)
	actionProbs, err := h.actor.GetProbs(state)
	if err != nil {
		return nil, err
	}

	// For opponent, use only one action
	if currentPlayer != learnerId {
		// Uncomment later
		// h.Memory.AddStrategySample(state, actionProbs, cfr_it)

		_action, err := random.Sample(h.rng, actionProbs)
		if err != nil {
			return nil, fmt.Errorf("Invalid probs sum")
		}
		action := nolimitholdem.Action(_action)

		h.coreGame.Step(action)
		childPayoffs, err := h.traverser(learnerId, cfr_it)
		if err != nil {
			return nil, err
		}
		h.coreGame.StepBack()
		return childPayoffs, nil // Возвращаем payoffs только для сэмплированного действия
	}

	totalPayoffs := make([]float32, h.coreGame.PlayersCount())
	// Payoffs for every action
	actionRawPayoffs := make(map[nolimitholdem.Action]float32)

	// Iterate over all possible actions
	for action, actProb := range actionProbs {
		h.coreGame.Step(action)
		childPayoffs, err := h.traverser(learnerId, cfr_it)
		if err != nil {
			return nil, err
		}
		h.coreGame.StepBack()

		// Сохраняем результаты
		actionRawPayoffs[action] = childPayoffs[learnerId]

		// Взвешенно добавляем в общий EV
		for i, val := range childPayoffs {
			totalPayoffs[i] += val * actProb
		}
	}

	// CFR HERE
	regrets := make(nolimitholdem.Strategy)
	for action, rawPayoff := range actionRawPayoffs {
		regrets[action] = rawPayoff - totalPayoffs[learnerId]
	}
	h.Memory.AddSample(learnerId, state, regrets, cfr_it)

	return totalPayoffs, nil
}
