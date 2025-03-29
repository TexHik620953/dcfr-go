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
	var err error
	var strategy nolimitholdem.Strategy
	var ex bool

	stateHash := state.Hash()

	if learnerId == int(state.CurrentPlayer) {
		// Get from cache
		strategy, ex = h.cache.GetRecord(int(state.CurrentPlayer), stateHash)
		if !ex {
			// Ask direct player network and store in cache
			strategy, err = h.getStrategy(state)
			h.cache.AddRecord(int(state.CurrentPlayer), stateHash, state, strategy)
		}
	} else {
		// Get from cache
		strategy, ex = h.cache.GetRecord(-1, stateHash)
		if !ex {
			// Ask direct player network and store in cache
			strategy, err = h.getAvgStrategy(state)
			h.cache.AddRecord(-1, stateHash, state, strategy)
		}
	}
	if err != nil {
		return nil, err
	}

	return strategy, nil
}

func (h *DeepCFRActor) getStrategy(state *nolimitholdem.GameState) (nolimitholdem.Strategy, error) {
	ch := h.executor.EnqueueGetStrategy(state)
	data := <-ch
	return data, nil
}

func (h *DeepCFRActor) getAvgStrategy(state *nolimitholdem.GameState) (nolimitholdem.Strategy, error) {
	ch := h.executor.EnqueueGetAvgStrategy(state)
	data := <-ch
	return data, nil
}
