package cfr

import (
	"context"
	"dcfr-go/common/defaultmap"
	"dcfr-go/common/safemap"
	"dcfr-go/nolimitholdem"
	"dcfr-go/proto/infra"
	"log"
	"sync"
	"time"
	"unsafe"

	"github.com/google/uuid"
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
func proto2probs(probs *infra.ProbsResponse) nolimitholdem.Strategy {
	return probs.ActionProbs
}

func sample2proto(sample *Sample) *infra.Sample {
	return &infra.Sample{
		State:     state2proto(sample.State),
		Regrets:   *(*map[int32]float32)(unsafe.Pointer(&sample.Regrets)),
		ReachProb: 0,
		Iteration: int32(sample.Iteration),
	}
}

type execution_unit struct {
	rq     *infra.GameState
	respCh chan nolimitholdem.Strategy
}

type GRPCBatchExecutor struct {
	conn        *grpc.ClientConn
	actorClient infra.ActorClient

	batchSize    int
	maxBatchSize int

	requestsPool Defaultmap[int, Safemap[string, execution_unit]]

	lastExec time.Time

	executionLock sync.Mutex
}

func NewGrpcBatchExecutor(serverAddr string, batchSize int, maxBatchSize int) (*GRPCBatchExecutor, error) {
	h := &GRPCBatchExecutor{
		batchSize:    batchSize,
		maxBatchSize: batchSize,
		requestsPool: defaultmap.New[int](func() Safemap[string, execution_unit] {
			return safemap.New[string, execution_unit]()
		}),
		lastExec: time.Now(),
	}
	var err error
	h.conn, err = grpc.NewClient(serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	h.actorClient = infra.NewActorClient(h.conn)

	go h.watcher()

	return h, nil
}

func (h *GRPCBatchExecutor) watcher() {
	for {
		if time.Since(h.lastExec) > time.Millisecond*100 {
			keys := make([]int, h.requestsPool.Count())
			h.requestsPool.Foreach(func(i int, s Safemap[string, execution_unit]) bool {
				keys = append(keys, i)
				return true
			})
			for _, k := range keys {
				h.execute(k)
			}
		}
		<-time.After(time.Millisecond * 110)
	}
}

func (h *GRPCBatchExecutor) Save() error {
	_, err := h.actorClient.Save(context.Background(), &infra.Empty{})
	return err
}
func (h *GRPCBatchExecutor) Reset() error {
	_, err := h.actorClient.Reset(context.Background(), &infra.Empty{})
	return err
}

func (h *GRPCBatchExecutor) Train(learnerId int, samples []*Sample) (float32, error) {
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
func (h *GRPCBatchExecutor) execute(playerId int) {
	h.executionLock.Lock()
	defer h.executionLock.Unlock()
	h.lastExec = time.Now()

	rp := h.requestsPool.Get(playerId)

	targetSize := rp.Count()
	if targetSize > h.maxBatchSize {
		targetSize = h.maxBatchSize
	}
	if targetSize == 0 {
		return
	}

	req := &infra.GameStateRequest{
		State: make([]*infra.GameState, 0, targetSize),
	}

	keys := make([]string, 0, targetSize)
	c := 0
	rp.Foreach(func(k string, v execution_unit) bool {
		req.State = append(req.State, v.rq)
		keys = append(keys, k)
		c++
		return c < targetSize
	})

	resp, err := h.actorClient.GetProbs(context.Background(), req)
	if err != nil {
		log.Fatal(err)
	}

	for i, v := range resp.Responses {
		key := keys[i]
		unit, ex := rp.Get(key)
		if !ex {
			log.Fatal("this should not happens")
		}
		unit.respCh <- proto2probs(v)
		close(unit.respCh)
		rp.Delete(key)
	}
}

func (h *GRPCBatchExecutor) EnqueueGetStrategy(state *nolimitholdem.GameState) chan nolimitholdem.Strategy {
	h.executionLock.Lock()
	defer h.executionLock.Unlock()

	rp := h.requestsPool.Get(int(state.CurrentPlayer))

	req_id := uuid.NewString()
	for rp.Exists(req_id) {
		req_id = uuid.NewString()
	}

	ch := make(chan nolimitholdem.Strategy, 1)

	rp.Set(req_id, execution_unit{
		rq:     state2proto(state),
		respCh: ch,
	})
	if rp.Count() >= h.batchSize {
		go func() {
			h.execute(int(state.CurrentPlayer))
		}()
	}
	return ch
}
