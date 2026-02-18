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

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func cfrstate2proto(state *CFRState) *infra.CFRState {
	s := &infra.CFRState{
		GameState:    gamestate2proto(state.GameState),
		LstmContextH: state.ActorState.LstmH,
		LstmContextC: state.ActorState.LstmC,
	}
	return s
}

func gamestate2proto(state *nolimitholdem.GameState) *infra.GameState {
	// Копирование мапы LegalActions
	legalActions := make(map[int32]bool)
	for k, _ := range state.LegalActions {
		legalActions[k] = true
	}

	// Копирование слайсов
	publicCards := make([]int32, len(state.PublicCards))
	for i, c := range state.PublicCards {
		publicCards[i] = int32(c)
	}

	privateCards := make([]int32, len(state.PrivateCards))
	for i, c := range state.PrivateCards {
		privateCards[i] = int32(c)
	}

	// Создание структуры GameState
	s := &infra.GameState{
		ActivePlayersMask: state.ActivePlayersMask,
		PlayersPots:       state.PlayersPots, // если это не ссылочный тип
		Stakes:            state.Stakes,
		LegalActions:      legalActions,
		Stage:             infra.GameStage(state.Stage),
		CurrentPlayer:     state.CurrentPlayer,
		PublicCards:       publicCards,
		PrivateCards:      privateCards,
	}
	return s
}
func proto2strat(probs *infra.ProbsResponse) *StrategyWithContext {
	r := &StrategyWithContext{
		Strategy: probs.ActionProbs,
		LstmH:    make([]float32, len(probs.LstmContextH)),
		LstmC:    make([]float32, len(probs.LstmContextC)),
	}

	copy(r.LstmH, probs.LstmContextH)
	copy(r.LstmC, probs.LstmContextC)

	return r
}
func sample2proto(gameSample *GameSample) *infra.GameSample {
	s := &infra.GameSample{
		Samples: make([]*infra.StateSample, len(gameSample.States)),
	}
	for i, sample := range gameSample.States {
		smpl := &infra.StateSample{
			GameState:    gamestate2proto(sample.GameState),
			LstmContextH: sample.ActorState.LstmH,
			LstmContextC: sample.ActorState.LstmC,
			Regrets:      map[int32]float32{},
			Iteration:    int32(sample.Iteration),
		}
		for k, v := range sample.Regrets {
			smpl.Regrets[k] = v
		}
		s.Samples[i] = smpl
	}
	return s
}

type execution_unit struct {
	rq     *infra.CFRState
	respCh chan *StrategyWithContext
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
	h.conn, err = grpc.NewClient(serverAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(512*1024*1024), // 512MB для отправки
			grpc.MaxCallRecvMsgSize(512*1024*1024), // 512MB для получения
		))
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

func (h *GRPCBatchExecutor) Train(learnerId int, samples []*GameSample) (float32, error) {
	req := &infra.TrainRequest{
		CurrentPlayer: int32(learnerId),
		GameSamples:   make([]*infra.GameSample, len(samples)),
	}
	for i, sample := range samples {
		req.GameSamples[i] = sample2proto(sample)
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

	req := &infra.ActionProbsRequest{
		States: make([]*infra.CFRState, 0, targetSize),
	}

	keys := make([]string, 0, targetSize)
	c := 0
	rp.Foreach(func(k string, v execution_unit) bool {
		req.States = append(req.States, v.rq)
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
		unit.respCh <- proto2strat(v)
		close(unit.respCh)
		rp.Delete(key)
	}
}

func (h *GRPCBatchExecutor) EnqueueGetStrategy(state *CFRState) chan *StrategyWithContext {
	h.executionLock.Lock()
	defer h.executionLock.Unlock()

	rp := h.requestsPool.Get(int(state.GameState.CurrentPlayer))

	req_id := uuid.NewString()
	for rp.Exists(req_id) {
		req_id = uuid.NewString()
	}

	ch := make(chan *StrategyWithContext, 1)

	rp.Set(req_id, execution_unit{
		rq:     cfrstate2proto(state),
		respCh: ch,
	})
	if rp.Count() >= h.batchSize {
		go func() {
			h.execute(int(state.GameState.CurrentPlayer))
		}()
	}
	return ch
}
