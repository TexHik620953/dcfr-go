package cfr

import (
	"context"
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
	}
	return s
}

func gamestate2proto(state *nolimitholdem.GameState) *infra.GameState {
	legalActions := make(map[int32]bool)
	for k := range state.LegalActions {
		legalActions[k] = true
	}

	publicCards := make([]int32, len(state.PublicCards))
	for i, c := range state.PublicCards {
		publicCards[i] = int32(c)
	}

	privateCards := make([]int32, len(state.PrivateCards))
	for i, c := range state.PrivateCards {
		privateCards[i] = int32(c)
	}

	s := &infra.GameState{
		ActivePlayersMask: state.ActivePlayersMask,
		PlayersPots:       state.PlayersPots,
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
		Strategy:       probs.ActionProbs,
		HistoryContext: make([]float32, len(probs.LstmContextH)),
	}
	copy(r.HistoryContext, probs.LstmContextH)
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

func strategySample2proto(gameSample *StrategyGameSample) *infra.StrategyGameSample {
	s := &infra.StrategyGameSample{
		Samples: make([]*infra.StrategySample, len(gameSample.States)),
	}
	for i, sample := range gameSample.States {
		smpl := &infra.StrategySample{
			GameState:    gamestate2proto(sample.GameState),
			LstmContextH: sample.ActorState.LstmH,
			Strategy:     map[int32]float32{},
			Iteration:    int32(sample.Iteration),
		}
		for k, v := range sample.Strategy {
			smpl.Strategy[k] = v
		}
		s.Samples[i] = smpl
	}
	return s
}

type execution_unit struct {
	rq     *infra.CFRState
	respCh chan *StrategyWithContext
}

// playerExecutor handles inference batching for a single player.
type playerExecutor struct {
	playerId     int
	actorClient  infra.ActorClient
	batchSize    int
	maxBatchSize int
	requests     Safemap[string, execution_unit]
	mu           sync.Mutex
}

func newPlayerExecutor(playerId int, client infra.ActorClient, batchSize, maxBatchSize int, autoExecInterval time.Duration) *playerExecutor {
	pe := &playerExecutor{
		playerId:     playerId,
		actorClient:  client,
		batchSize:    batchSize,
		maxBatchSize: maxBatchSize,
		requests:     safemap.New[string, execution_unit](),
	}
	go pe.watcher(autoExecInterval)
	return pe
}

func (pe *playerExecutor) watcher(interval time.Duration) {
	for {
		<-time.After(interval)
		if pe.requests.Count() > 0 {
			pe.doExecute(false)
		}
	}
}

func (pe *playerExecutor) enqueue(state *CFRState) chan *StrategyWithContext {
	req_id := uuid.NewString()
	for pe.requests.Exists(req_id) {
		req_id = uuid.NewString()
	}

	ch := make(chan *StrategyWithContext, 1)
	pe.requests.Set(req_id, execution_unit{
		rq:     cfrstate2proto(state),
		respCh: ch,
	})
	if pe.requests.Count() >= pe.batchSize {
		go pe.doExecute(true)
	}
	return ch
}

func (pe *playerExecutor) doExecute(requireBatchSize bool) {
	if !pe.mu.TryLock() {
		return
	}
	defer pe.mu.Unlock()

	targetSize := pe.requests.Count()
	if targetSize == 0 {
		return
	}
	if requireBatchSize && targetSize < pe.batchSize {
		return
	}
	if targetSize > pe.maxBatchSize {
		targetSize = pe.maxBatchSize
	}

	req := &infra.ActionProbsRequest{
		States: make([]*infra.CFRState, 0, targetSize),
	}

	keys := make([]string, 0, targetSize)
	c := 0
	pe.requests.Foreach(func(k string, v execution_unit) bool {
		req.States = append(req.States, v.rq)
		keys = append(keys, k)
		c++
		return c < targetSize
	})

	resp, err := pe.actorClient.GetProbs(context.Background(), req)
	if err != nil {
		log.Fatal(err)
	}

	for i, v := range resp.Responses {
		key := keys[i]
		unit, ex := pe.requests.Get(key)
		if !ex {
			log.Fatal("this should not happen")
		}
		unit.respCh <- proto2strat(v)
		close(unit.respCh)
		pe.requests.Delete(key)
	}
}

// GRPCBatchExecutor wraps 3 independent per-player executors.
type GRPCBatchExecutor struct {
	conn        *grpc.ClientConn
	actorClient infra.ActorClient
	players     [3]*playerExecutor
}

func NewGrpcBatchExecutor(serverAddr string, batchSize int, maxBatchSize int, autoExecInterval time.Duration) (*GRPCBatchExecutor, error) {
	conn, err := grpc.NewClient(serverAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(512*1024*1024),
			grpc.MaxCallRecvMsgSize(512*1024*1024),
		))
	if err != nil {
		return nil, err
	}

	client := infra.NewActorClient(conn)
	h := &GRPCBatchExecutor{
		conn:        conn,
		actorClient: client,
	}
	for i := range 3 {
		h.players[i] = newPlayerExecutor(i, client, batchSize, maxBatchSize, autoExecInterval)
	}
	return h, nil
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

func (h *GRPCBatchExecutor) TrainAvgStrategy(playerID int, samples []*StrategyGameSample) (float32, error) {
	req := &infra.TrainAvgStrategyRequest{
		CurrentPlayer: int32(playerID),
		GameSamples:   make([]*infra.StrategyGameSample, len(samples)),
	}
	for i, sample := range samples {
		req.GameSamples[i] = strategySample2proto(sample)
	}

	resp, err := h.actorClient.TrainAvgStrategy(context.Background(), req)
	if err != nil {
		return 0, err
	}
	return resp.Loss, nil
}

func (h *GRPCBatchExecutor) TrainDirect(playerID int, batchSize int, iterations int, dbPath string, maxSamples int) (float32, error) {
	resp, err := h.actorClient.TrainDirect(context.Background(), &infra.TrainDirectRequest{
		CurrentPlayer: int32(playerID),
		BatchSize:     int32(batchSize),
		Iterations:    int32(iterations),
		DbPath:        dbPath,
		MaxSamples:    int32(maxSamples),
	})
	if err != nil {
		return 0, err
	}
	return resp.AvgLoss, nil
}

func (h *GRPCBatchExecutor) EnqueueGetStrategy(state *CFRState) chan *StrategyWithContext {
	pid := int(state.GameState.CurrentPlayer)
	return h.players[pid].enqueue(state)
}
