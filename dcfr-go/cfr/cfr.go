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
		c.States[k] = v.Clone()
	}
	return c
}
func (h *ActorsContext) At(player int) *ActorState {
	return h.States[player]
}

// RegretMemory is the interface for regret sample storage.
type RegretMemory interface {
	AddSample(playerID int, gameID uuid.UUID, state *CFRState, regrets map[nolimitholdem.Action]float32, iteration int)
	FlushGame(playerID int, gameID uuid.UUID)
	GetSamples(playerID int, batchSize int) []*GameSample
	Count(playerID int) int
}

// StrategyMemory is the interface for strategy sample storage.
type StrategyMemoryI interface {
	AddSample(playerID int, gameID uuid.UUID, state *CFRState, strategy nolimitholdem.Strategy, iteration int)
	FlushGame(playerID int, gameID uuid.UUID)
	GetSamples(playerID int, batchSize int) []*StrategyGameSample
	Count(playerID int) int
}

type CFR struct {
	coreGame       *nolimitholdem.Game
	actor          CFRActor
	Memory         RegretMemory
	StrategyMemory StrategyMemoryI

	stats *CFRStats

	rng *rand.Rand

	// Pruning: skip actions with probability below this threshold for learner
	pruneThreshold float32
}
type CFRStats struct {
	NodesVisited   atomic.Int32
	TreesTraversed atomic.Int32
}

func New(seed int64, game *nolimitholdem.Game, actor CFRActor, memory RegretMemory, strategyMemory StrategyMemoryI, stats *CFRStats) *CFR {
	h := &CFR{
		coreGame:       game,
		actor:          actor,
		Memory:         memory,
		StrategyMemory: strategyMemory,
		stats:          stats,
		rng:            rand.New(rand.NewSource(seed)),
		pruneThreshold: 0.03,
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
		playersContext.States[i] = &ActorState{}
	}

	gameID := uuid.New()

	payoffs, err := h.traverser(learnerId, iteration, gameID, playersContext, 0)
	if err != nil {
		return nil, err
	}
	// Flush the completed game into the reservoir
	h.Memory.FlushGame(learnerId, gameID)
	// Flush strategy samples for all players (for average strategy network)
	for i := range h.coreGame.PlayersCount() {
		h.StrategyMemory.FlushGame(i, gameID)
	}
	h.stats.TreesTraversed.Add(1)
	return payoffs, nil
}

func (h *CFR) traverser(learnerId int, cfr_it int, gameID uuid.UUID, playersState *ActorsContext, depth int) ([]float32, error) {
	h.stats.NodesVisited.Add(1)
	if h.coreGame.IsOver() {
		return h.coreGame.GetPayoffs(), nil
	}
	if depth > 100 {
		return h.coreGame.GetPayoffs(), nil
	}

	currentPlayer := h.coreGame.CurrentPlayer()
	state := h.coreGame.GetState(currentPlayer)

	// Get previous context for player
	actorState := playersState.At(currentPlayer)

	// Save prev context BEFORE inference (for training samples)
	prevActorState := actorState.Clone()

	actionContext, err := h.actor.GetProbs(&CFRState{
		GameState:  state,
		ActorState: actorState,
	})
	if err != nil {
		return nil, err
	}
	// Store new context for player (for next step's inference)
	actorState.LstmH = actionContext.HistoryContext

	// Collect strategy sample with PREV context (so train can reproduce the context update)
	h.StrategyMemory.AddSample(currentPlayer, gameID, &CFRState{
		GameState:  state,
		ActorState: prevActorState,
	}, actionContext.Strategy, cfr_it)

	// For opponent, use only one action
	if currentPlayer != learnerId {
		_action, err := random.Sample(h.rng, actionContext.Strategy)
		if err != nil {
			return nil, fmt.Errorf("Invalid probs sum")
		}
		action := nolimitholdem.Action(_action)

		h.coreGame.Step(action)
		childPayoffs, err := h.traverser(learnerId, cfr_it, gameID, playersState.Clone(), depth+1)
		if err != nil {
			return nil, err
		}
		h.coreGame.StepBack()
		return childPayoffs, nil // Возвращаем payoffs только для сэмплированного действия
	}

	totalPayoffs := make([]float32, h.coreGame.PlayersCount())
	// Payoffs for every action
	actionRawPayoffs := make(map[nolimitholdem.Action]float32)

	// Iterate over actions, pruning low-probability ones
	// Count how many actions pass threshold — always explore at least 2
	aboveThreshold := 0
	for _, actProb := range actionContext.Strategy {
		if actProb >= h.pruneThreshold {
			aboveThreshold++
		}
	}
	canPrune := aboveThreshold >= 2

	for action, actProb := range actionContext.Strategy {
		if canPrune && actProb < h.pruneThreshold {
			continue
		}

		h.coreGame.Step(action)
		childPayoffs, err := h.traverser(learnerId, cfr_it, gameID, playersState.Clone(), depth+1)
		if err != nil {
			return nil, err
		}
		h.coreGame.StepBack()

		actionRawPayoffs[action] = childPayoffs[learnerId]

		for i, val := range childPayoffs {
			totalPayoffs[i] += val * actProb
		}
	}

	// CFR: compute regrets only for explored actions
	regrets := make(nolimitholdem.Strategy)
	for action := range actionContext.Strategy {
		if payoff, explored := actionRawPayoffs[action]; explored {
			regrets[action] = payoff - totalPayoffs[learnerId]
		} else {
			// Pruned action — assign zero regret
			regrets[action] = 0
		}
	}
	h.Memory.AddSample(learnerId, gameID, &CFRState{
		GameState:  state,
		ActorState: prevActorState,
	}, regrets, cfr_it)

	return totalPayoffs, nil
}
