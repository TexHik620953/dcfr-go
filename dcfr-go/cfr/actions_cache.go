package cfr

import (
	"dcfr-go/common/defaultmap"
	"dcfr-go/common/safemap"
	"dcfr-go/nolimitholdem"
)

type Defaultmap[K comparable, V any] = defaultmap.DefaultSafemap[K, V]
type Safemap[K comparable, V any] = safemap.Safemap[K, V]

type ActionsCache struct {
	playerStrategyCache Defaultmap[int, Safemap[nolimitholdem.GameStateHash, nolimitholdem.Strategy]]
	cacheLimit          int
	cacheClearRate      float32
}

func NewActionsCache(cacheLimit int, cacheClearRate float32) *ActionsCache {
	return &ActionsCache{
		playerStrategyCache: defaultmap.New[int](func() Safemap[nolimitholdem.GameStateHash, nolimitholdem.Strategy] {
			return safemap.New[nolimitholdem.GameStateHash, nolimitholdem.Strategy]()
		}),
		cacheLimit:     cacheLimit,
		cacheClearRate: cacheClearRate,
	}
}
func (h *ActionsCache) ClearAll() {
	h.playerStrategyCache = defaultmap.New[int](func() Safemap[nolimitholdem.GameStateHash, nolimitholdem.Strategy] {
		return safemap.New[nolimitholdem.GameStateHash, nolimitholdem.Strategy]()
	})
}
func (h *ActionsCache) Clear() {
	keys := make([]int, 0)
	h.playerStrategyCache.Foreach(func(i int, s Safemap[nolimitholdem.GameStateHash, nolimitholdem.Strategy]) bool {
		keys = append(keys, i)
		return true
	})
	for _, playerId := range keys {
		h.playerStrategyCache.Delete(playerId)
	}
}

// -1 equals to Average network
func (h *ActionsCache) AddRecord(playerId int, stateHash nolimitholdem.GameStateHash, state *nolimitholdem.GameState, strategy nolimitholdem.Strategy) {
	plyCache := h.playerStrategyCache.Get(playerId)

	plyCache.Set(stateHash, strategy)
}

// -1 equals to Average network
func (h *ActionsCache) GetRecord(playerId int, stateHash nolimitholdem.GameStateHash) (nolimitholdem.Strategy, bool) {
	plyCache := h.playerStrategyCache.Get(playerId)
	record, ex := plyCache.Get(stateHash)
	if !ex {
		return nil, false
	}
	return record, true
}
