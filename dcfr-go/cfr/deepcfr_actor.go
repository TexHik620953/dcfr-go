package cfr

import (
	"dcfr-go/nolimitholdem"
)

type CFRActor interface {
	GetProbs(learnerId int, state *nolimitholdem.GameState) (nolimitholdem.Strategy, error)
}

type DeepCFRActor struct {
	cache       *ActionsCache
	executor    *GRPCBatchExecutor
	players_cnt int
}

func NewDeepCFRActor(cache *ActionsCache, executor *GRPCBatchExecutor, players_cnt int) CFRActor {
	return &DeepCFRActor{
		cache:       cache,
		executor:    executor,
		players_cnt: players_cnt,
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
	chans := make([]chan nolimitholdem.Strategy, 0, h.players_cnt-1)

	for ply_id := range h.players_cnt {
		if ply_id != int(state.CurrentPlayer) {
			state_copy := state.Clone()
			state_copy.CurrentPlayer = int32(ply_id)
			chans = append(chans, h.executor.EnqueueGetStrategy(state_copy))
		}
	}

	data := make([]nolimitholdem.Strategy, 0, h.players_cnt-1)
	for _, ch := range chans {
		data = append(data, <-ch)
	}

	// Average opponents strategies
	avg_strategy := nolimitholdem.Strategy{}
	for _, s := range data {
		for action, prob := range s {
			_, ex := avg_strategy[action]
			if ex {
				avg_strategy[action] += prob / 2
			} else {
				avg_strategy[action] = prob / 2
			}
		}
	}

	// Normalize
	sum := float32(0)
	for _, prob := range avg_strategy {
		sum += prob
	}
	if sum <= 0.99 || sum >= 1.01 {
		for a, _ := range avg_strategy {
			avg_strategy[a] /= sum
		}
	}

	return avg_strategy, nil
}
