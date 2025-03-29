package cfr

import (
	"context"
	"dcfr-go/common/defaultmap"
	"dcfr-go/common/safemap"
	"dcfr-go/nolimitholdem"
	"dcfr-go/proto/infra"
	"slices"
	"time"
	"unsafe"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func state2proto(state *nolimitholdem.GameState) *infra.GameState {
	return &infra.GameState{
		ActivePlayersMask: state.ActivePlayersMask,
		PlayersPots:       state.PlayersPots,
		Stakes:            state.Stakes,
		LegalActions:      *(*map[int32]bool)(unsafe.Pointer(&state.LegalActions)),
		Stage:             infra.GameStage(state.Stage),
		CurrentPlayer:     state.CurrentPlayer,
		PublicCards:       *(*[]int32)(unsafe.Pointer(&state.PublicCards)),
		PrivateCards:      *(*[]int32)(unsafe.Pointer(&state.PrivateCards)),
	}
}
func sample2proto(sample *Sample) *infra.Sample {
	return &infra.Sample{
		State:     state2proto(sample.State),
		Regrets:   *(*map[int32]float32)(unsafe.Pointer(&sample.Regrets)),
		Weight:    sample.Weight,
		Iteration: int32(sample.Iteration),
	}
}

type Defaultmap[K comparable, V any] = defaultmap.DefaultSafemap[K, V]
type Safemap[K comparable, V any] = safemap.Safemap[K, V]
type CacheRecord struct {
	Strategy nolimitholdem.Strategy
	AddTime  time.Time
}

const PLAYER_CACHE_LIMIT = 20000
const PLAYER_CACHE_CLEAR = 0.2

type DeepCFRActor struct {
	playerStrategyCache Defaultmap[int, Safemap[nolimitholdem.GameStateHash, CacheRecord]]
	cacheHitCnt         int
	cacheMissCnt        int

	conn        *grpc.ClientConn
	actorClient infra.ActorClient
}

func NewDeepCFRActor(serverAddr string) (*DeepCFRActor, error) {

	h := &DeepCFRActor{
		playerStrategyCache: defaultmap.New[int](func() Safemap[nolimitholdem.GameStateHash, CacheRecord] {
			return safemap.New[nolimitholdem.GameStateHash, CacheRecord]()
		}),
		cacheHitCnt:  0,
		cacheMissCnt: 0,
	}

	var err error
	h.conn, err = grpc.NewClient(serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	h.actorClient = infra.NewActorClient(h.conn)
	return h, nil
}

func (h *DeepCFRActor) ClearCache() {
	h.playerStrategyCache = defaultmap.New[int](func() Safemap[nolimitholdem.GameStateHash, CacheRecord] {
		return safemap.New[nolimitholdem.GameStateHash, CacheRecord]()
	})
	h.cacheHitCnt = 0
	h.cacheMissCnt = 0
}

func (h *DeepCFRActor) GetProbs(learnerId int, state *nolimitholdem.GameState) (nolimitholdem.Strategy, error) {
	plyCache := h.playerStrategyCache.Get(int(state.CurrentPlayer))
	stateHash := state.Hash()
	// Get from cache
	cached, ex := plyCache.Get(stateHash)
	if ex {
		h.cacheHitCnt++
		return cached.Strategy, nil
	}
	h.cacheMissCnt++

	var err error
	var strategy nolimitholdem.Strategy
	if learnerId == int(state.CurrentPlayer) {
		// Ask direct player network
		strategy, err = h.getStrategy(state)
	} else {
		// Ask average network
		strategy, err = h.getAvgStrategy(state)
	}
	if err != nil {
		return nil, err
	}

	// Add to cache
	plyCache.Set(stateHash, CacheRecord{
		Strategy: strategy,
		AddTime:  time.Now(),
	})

	// Remove oldest keys
	if plyCache.Count() > PLAYER_CACHE_LIMIT {
		keys := make([]nolimitholdem.GameStateHash, plyCache.Count())
		i := 0
		plyCache.Foreach(func(gsh nolimitholdem.GameStateHash, cr CacheRecord) {
			keys[i] = gsh
			i++
		})
		slices.SortFunc(keys, func(a, b nolimitholdem.GameStateHash) int {
			a_v, _ := plyCache.Get(a)
			b_v, _ := plyCache.Get(b)

			return int(a_v.AddTime.Sub(b_v.AddTime))
		})
		for _, key := range keys[:int(float32(len(keys))*PLAYER_CACHE_CLEAR)] {
			plyCache.Delete(key)
		}
	}

	return strategy, nil
}

func (h *DeepCFRActor) getStrategy(state *nolimitholdem.GameState) (nolimitholdem.Strategy, error) {
	req := &infra.GameStateRequest{
		State: state2proto(state),
	}

	resp, err := h.actorClient.GetProbs(context.Background(), req)
	if err != nil {
		return nil, err
	}
	return *(*nolimitholdem.Strategy)(unsafe.Pointer(&resp.ActionProbs)), nil
}

func (h *DeepCFRActor) getAvgStrategy(state *nolimitholdem.GameState) (nolimitholdem.Strategy, error) {
	req := &infra.GameStateRequest{
		State: state2proto(state),
	}

	resp, err := h.actorClient.GetAvgProbs(context.Background(), req)
	if err != nil {
		return nil, err
	}
	return *(*nolimitholdem.Strategy)(unsafe.Pointer(&resp.ActionProbs)), nil
}

func (h *DeepCFRActor) Train(learnerId int, samples []*Sample) (float32, error) {
	req := &infra.TrainRequest{
		CurrentPlayer: int32(learnerId),
		Samples:       make([]*infra.Sample, len(samples)),
	}
	for i, sample := range samples {
		req.Samples[i] = sample2proto(sample)
	}

	resp, err := h.actorClient.Train(context.Background(), req)
	if err != nil {
		return 0, err
	}
	return resp.Loss, nil
}
func (h *DeepCFRActor) TrainAvg() (float32, error) {
	req := &infra.TrainAvgRequest{}

	resp, err := h.actorClient.TrainAvg(context.Background(), req)
	if err != nil {
		return 0, err
	}
	return resp.Loss, nil
}
