package cfr

import (
	"dcfr-go/nolimitholdem"
	"encoding/binary"
	"hash"
	"hash/fnv"
)

type CFRActorStateHash uint64

type StrategyWithContext struct {
	Strategy nolimitholdem.Strategy
	// Transformer history context (flattened sequence features)
	HistoryContext []float32
}

type CFRState struct {
	GameState  *nolimitholdem.GameState
	ActorState *ActorState
}

func (h *CFRState) Hash() uint64 {
	hasher := fnv.New64a()
	h.WriteHash(hasher)
	return hasher.Sum64()
}

func (h *CFRState) WriteHash(hasher hash.Hash64) {
	h.GameState.WriteHash(hasher)

	// Hash history context
	for _, val := range h.ActorState.LstmH {
		binary.Write(hasher, binary.LittleEndian, val)
	}
}

type CFRActor interface {
	GetProbs(state *CFRState) (*StrategyWithContext, error)
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

func (h *DeepCFRActor) GetProbs(state *CFRState) (*StrategyWithContext, error) {
	stateHash := state.Hash()
	strategy, ex := h.cache.GetRecord(int(state.GameState.CurrentPlayer), stateHash)
	if !ex {
		var err error
		strategy, err = h.getStrategy(state)
		if err != nil {
			return nil, err
		}
		h.cache.AddRecord(int(state.GameState.CurrentPlayer), stateHash, strategy)
	}
	return strategy, nil
}

func (h *DeepCFRActor) getStrategy(state *CFRState) (*StrategyWithContext, error) {
	ch := h.executor.EnqueueGetStrategy(state)
	r := <-ch
	return r, nil
}
