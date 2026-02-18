package cfr

import (
	"dcfr-go/common/random"
	"dcfr-go/nolimitholdem"
	"fmt"
	"math/rand"
	"sync/atomic"

	"github.com/google/uuid"
)

type ActorsContext struct {
	States map[int]*ActorState
}

func (h *ActorsContext) Clone() *ActorsContext {
	c := &ActorsContext{
		States: make(map[int]*ActorState),
	}
	for k, v := range h.States {
		s := &ActorState{
			LstmH: make([]float32, len(v.LstmH)),
			LstmC: make([]float32, len(v.LstmC)),
		}
		copy(s.LstmC, v.LstmC)
		copy(s.LstmH, v.LstmH)
		c.States[k] = s
	}
	return c
}
func (h *ActorsContext) At(player int) *ActorState {
	return h.States[player]
}

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

func (h *CFR) TraverseTree(learnerId int, iteration int) ([]float32, error) {
	// Reset game
	h.coreGame.Reset()

	// Reset players contexts
	playersContext := &ActorsContext{
		States: map[int]*ActorState{},
	}
	for i := range h.coreGame.PlayersCount() {
		playersContext.States[i] = &ActorState{
			LstmH: nil,
			LstmC: nil,
		}
	}

	gameID := uuid.New()

	payoffs, err := h.traverser(learnerId, iteration, gameID, playersContext)
	if err != nil {
		return nil, err
	}
	h.stats.TreesTraversed.Add(1)
	return payoffs, nil
}

func (h *CFR) traverser(learnerId int, cfr_it int, gameID uuid.UUID, playersState *ActorsContext) ([]float32, error) {
	h.stats.NodesVisited.Add(1)
	if h.coreGame.IsOver() {
		return h.coreGame.GetPayoffs(), nil
	}

	currentPlayer := h.coreGame.CurrentPlayer()
	state := h.coreGame.GetState(currentPlayer)

	// Get previous context for player
	actorState := playersState.At(currentPlayer)

	actionContext, err := h.actor.GetProbs(&CFRState{
		GameState:  state,
		ActorState: actorState,
	})
	if err != nil {
		return nil, err
	}
	// Store new context for player
	actorState.LstmC = actionContext.LstmC
	actorState.LstmH = actionContext.LstmH

	// For opponent, use only one action
	if currentPlayer != learnerId {
		// Uncomment later
		// h.Memory.AddStrategySample(state, actionProbs, cfr_it)

		_action, err := random.Sample(h.rng, actionContext.Strategy)
		if err != nil {
			return nil, fmt.Errorf("Invalid probs sum")
		}
		action := nolimitholdem.Action(_action)

		h.coreGame.Step(action)
		childPayoffs, err := h.traverser(learnerId, cfr_it, gameID, playersState.Clone())
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
	for action, actProb := range actionContext.Strategy {
		h.coreGame.Step(action)
		childPayoffs, err := h.traverser(learnerId, cfr_it, gameID, playersState.Clone())
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
	h.Memory.AddSample(learnerId, gameID, &CFRState{
		GameState:  state,
		ActorState: actorState,
	}, regrets, cfr_it)

	return totalPayoffs, nil
}
