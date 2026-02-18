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
	LstmH    []float32
	LstmC    []float32
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
	// Хэшируем GameState
	h.GameState.WriteHash(hasher)

	// Хэшируем LstmH
	for _, val := range h.ActorState.LstmH {
		binary.Write(hasher, binary.LittleEndian, val)
	}

	// Хэшируем LstmC
	for _, val := range h.ActorState.LstmC {
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
	// All players use their own current strategy (from their own network)
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
