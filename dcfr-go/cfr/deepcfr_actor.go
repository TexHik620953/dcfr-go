package cfr

import (
	"dcfr-go/nolimitholdem"
)

type CFRActor interface {
	GetProbs(learnerId int, state *nolimitholdem.GameState) (nolimitholdem.Strategy, error)
}

type DeepCFRActor struct {
	cache    *ActionsCache
	executor *GRPCBatchExecutor
}

func NewDeepCFRActor(cache *ActionsCache, executor *GRPCBatchExecutor) CFRActor {
	return &DeepCFRActor{
		cache:    cache,
		executor: executor,
	}
}

func (h *DeepCFRActor) GetProbs(learnerId int, state *nolimitholdem.GameState) (nolimitholdem.Strategy, error) {
	stateHash := state.Hash()
	// All players use their own current strategy (from their own network)
	strategy, ex := h.cache.GetRecord(int(state.CurrentPlayer), stateHash)
	if !ex {
		var err error
		strategy, err = h.getStrategy(state)
		if err != nil {
			return nil, err
		}
		h.cache.AddRecord(-1, stateHash, state, strategy)
	}
	return strategy, nil
}

func (h *DeepCFRActor) getStrategy(state *nolimitholdem.GameState) (nolimitholdem.Strategy, error) {
	ch := h.executor.EnqueueGetStrategy(state)
	return <-ch, nil
}
