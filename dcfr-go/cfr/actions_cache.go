package cfr

import (
	"dcfr-go/common/defaultmap"
	"dcfr-go/common/safemap"
)

type Defaultmap[K comparable, V any] = defaultmap.DefaultSafemap[K, V]
type Safemap[K comparable, V any] = safemap.Safemap[K, V]

type ActionsCache struct {
	playerStrategyCache Defaultmap[int, Safemap[uint64, *StrategyWithContext]]
	cacheLimit          int
	cacheClearRate      float32
}

func NewActionsCache(cacheLimit int, cacheClearRate float32) *ActionsCache {
	return &ActionsCache{
		playerStrategyCache: defaultmap.New[int](func() Safemap[uint64, *StrategyWithContext] {
			return safemap.New[uint64, *StrategyWithContext]()
		}),
		cacheLimit:     cacheLimit,
		cacheClearRate: cacheClearRate,
	}
}
func (h *ActionsCache) ClearAll() {
	h.playerStrategyCache = defaultmap.New[int](func() Safemap[uint64, *StrategyWithContext] {
		return safemap.New[uint64, *StrategyWithContext]()
	})
}
func (h *ActionsCache) Clear() {
	keys := make([]int, 0)
	h.playerStrategyCache.Foreach(func(i int, s Safemap[uint64, *StrategyWithContext]) bool {
		keys = append(keys, i)
		return true
	})
	for _, playerId := range keys {
		h.playerStrategyCache.Delete(playerId)
	}
}

// -1 equals to Average network
func (h *ActionsCache) AddRecord(playerId int, stateHash uint64, strategy *StrategyWithContext) {
	plyCache := h.playerStrategyCache.Get(playerId)

	plyCache.Set(stateHash, strategy)
}

// -1 equals to Average network
func (h *ActionsCache) GetRecord(playerId int, stateHash uint64) (*StrategyWithContext, bool) {
	plyCache := h.playerStrategyCache.Get(playerId)
	record, ex := plyCache.Get(stateHash)
	if !ex {
		return nil, false
	}
	return record, true
}
