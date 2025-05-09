package cfr

import (
	"dcfr-go/common/random"
	"dcfr-go/nolimitholdem"
	"log"
	"math"
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

func (h *CFR) TraverseTree(learnerId int, cfr_it int) ([]float32, error) {

	payoffs, err := h.traverser(0, learnerId, cfr_it)
	if err != nil {
		return nil, err
	}
	h.stats.TreesTraversed.Add(1)
	return payoffs, nil
}

func (h *CFR) traverser(logReachProb float32, learnerId int, cfr_it int) ([]float32, error) {
	h.stats.NodesVisited.Add(1)

	if h.coreGame.IsOver() {
		return h.coreGame.GetPayoffs(), nil
	}
	chanceNode := h.coreGame.GetCurrentNodeChance()

	logReachProb += float32(math.Log(chanceNode))

	currentPlayer := h.coreGame.CurrentPlayer()
	state := h.coreGame.GetState(currentPlayer)
	actionProbs, err := h.actor.GetProbs(learnerId, state)
	if err != nil {
		return nil, err
	}

	totalPayoffs := make([]float32, h.coreGame.PlayersCount())

	// For opponent, use only one action
	if currentPlayer != learnerId {
		h.Memory.AddStrategySample(state, actionProbs, cfr_it)

		_action, err := random.Sample(h.rng, *(*map[int32]float32)(unsafe.Pointer(&actionProbs)))
		if err != nil {
			log.Fatalf("Invalid probs sum")
		}
		action := nolimitholdem.Action(_action)
		logReachProb += float32(math.Log(float64(actionProbs[action])))

		h.coreGame.Step(action)
		childPayoffs, err := h.traverser(logReachProb, learnerId, cfr_it)
		if err != nil {
			return nil, err
		}
		h.coreGame.StepBack()
		for i, payoff := range childPayoffs {
			childPayoffs[i] = payoff * actionProbs[action]
		}
		return childPayoffs, nil // Возвращаем payoffs только для сэмплированного действия
	}

	// Payoffs for every action
	myActionsPayoff := make(map[nolimitholdem.Action]float32)
	// Iterate over all possible actions
	for action, action_probability := range actionProbs {
		logReachProb += float32(math.Log(float64(action_probability)))

		h.coreGame.Step(action)
		childPayoffs, err := h.traverser(logReachProb, learnerId, cfr_it)
		if err != nil {
			return nil, err
		}
		h.coreGame.StepBack()

		// Расчитываем ev
		for i, payoff := range childPayoffs {
			childPayoffs[i] = payoff * action_probability
			totalPayoffs[i] += childPayoffs[i]
		}

		// Сохраняем результаты
		myActionsPayoff[action] = childPayoffs[learnerId]
	}

	// CFR HERE
	regrets := make(nolimitholdem.Strategy)
	for action, payoffs := range myActionsPayoff {
		regret := payoffs - totalPayoffs[learnerId]
		// CFR+: только положительные сожаления
		if regret > 0 {
			regrets[action] = regret
		}
	}
	if len(regrets) > 0 {
		// Store state, regrets, cfr_iteration
		h.Memory.AddSample(learnerId, state, regrets, logReachProb, cfr_it)
	}

	return totalPayoffs, nil
}
