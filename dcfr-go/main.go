package main

import (
	"context"
	"dcfr-go/cfr"
	"dcfr-go/common/bench"
	"dcfr-go/nolimitholdem"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"runtime"

	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

const (
	TRAVERSE_THREADS     = 20000
	CFR_ITERS            = 1000
	TRAVERSE_ITERS       = 100000
	ADV_TRAIN_ITERS      = 1000
	ADV_TRAIN_WARMUP     = 4500
	ADV_WARMUP_CFR_ITERS = 0
	AVG_TRAIN_ITERS      = 20
	BATCH_SIZE           = 6000
	MAX_SAMPLES          = 1_500_000
)

type StartupTask struct {
	PlayerId int
	CfrIter  int
}

type Checkpoint struct {
	CfrIteration int `json:"cfr_iteration"`
}

func loadCheckpoint(dir string) Checkpoint {
	data, err := os.ReadFile(filepath.Join(dir, "checkpoint.json"))
	if err != nil {
		return Checkpoint{CfrIteration: 0}
	}
	var cp Checkpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		return Checkpoint{CfrIteration: 0}
	}
	return cp
}

func saveCheckpoint(dir string, cp Checkpoint) error {
	data, err := json.Marshal(cp)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "checkpoint.json"), data, 0644)
}

type benchmarkState struct {
	name  string
	state *cfr.CFRState
}

func buildBenchmarkStates() []benchmarkState {
	allActions := map[nolimitholdem.Action]struct{}{
		nolimitholdem.ACTION_FOLD:            {},
		nolimitholdem.ACTION_CHECK_CALL:      {},
		nolimitholdem.ACTION_RAISE_QUARTER:   {},
		nolimitholdem.ACTION_RAISE_THIRD:     {},
		nolimitholdem.ACTION_RAISE_HALFPOT:   {},
		nolimitholdem.ACTION_RAISE_TWOTHIRDS: {},
		nolimitholdem.ACTION_RAISE_POT:       {},
		nolimitholdem.ACTION_RAISE_1_5X:      {},
		nolimitholdem.ACTION_RAISE_2X:        {},
		nolimitholdem.ACTION_ALL_IN:          {},
	}

	makeState := func(name string, private [2]nolimitholdem.Card, public []nolimitholdem.Card, stage nolimitholdem.GameStage, pots, stakes [3]int32) benchmarkState {
		return benchmarkState{
			name: name,
			state: &cfr.CFRState{
				GameState: &nolimitholdem.GameState{
					PrivateCards:      private[:],
					PublicCards:       public,
					Stage:             stage,
					CurrentPlayer:     0,
					ActivePlayersMask: []int32{1, 1, 1},
					PlayersPots:       pots[:],
					Stakes:            stakes[:],
					LegalActions:      allActions,
				},
				ActorState: &cfr.ActorState{},
			},
		}
	}

	// rank: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
	// card = suit*13 + rank
	return []benchmarkState{
		makeState("AA_preflop",
			[2]nolimitholdem.Card{nolimitholdem.NewCard(12, 0), nolimitholdem.NewCard(12, 1)},
			nil, nolimitholdem.STAGE_PREFLOP,
			[3]int32{0, 5, 10}, [3]int32{70, 65, 60}),

		makeState("72o_preflop",
			[2]nolimitholdem.Card{nolimitholdem.NewCard(0, 1), nolimitholdem.NewCard(5, 0)},
			nil, nolimitholdem.STAGE_PREFLOP,
			[3]int32{0, 5, 10}, [3]int32{70, 65, 60}),

		makeState("KK_set_flop",
			[2]nolimitholdem.Card{nolimitholdem.NewCard(11, 0), nolimitholdem.NewCard(11, 1)},
			[]nolimitholdem.Card{nolimitholdem.NewCard(11, 2), nolimitholdem.NewCard(6, 0), nolimitholdem.NewCard(2, 1)},
			nolimitholdem.STAGE_FLOP,
			[3]int32{20, 20, 20}, [3]int32{50, 50, 50}),

		makeState("missed_river",
			[2]nolimitholdem.Card{nolimitholdem.NewCard(5, 0), nolimitholdem.NewCard(4, 0)},
			[]nolimitholdem.Card{
				nolimitholdem.NewCard(11, 1), nolimitholdem.NewCard(10, 2),
				nolimitholdem.NewCard(8, 3), nolimitholdem.NewCard(1, 1),
				nolimitholdem.NewCard(0, 2),
			},
			nolimitholdem.STAGE_RIVER,
			[3]int32{30, 30, 30}, [3]int32{40, 40, 40}),
	}
}

func logBenchmarkStrategies(cfrIt int, states []benchmarkState, executor *cfr.GRPCBatchExecutor) {
	for _, bs := range states {
		result := <-executor.EnqueueGetStrategy(bs.state)
		probs := make([]string, 0, len(result.Strategy))
		for action, prob := range result.Strategy {
			if prob > 0.01 {
				probs = append(probs, fmt.Sprintf("%s=%.2f", nolimitholdem.Action2string[action], prob))
			}
		}
		log.Printf("[CFR %d] BENCH %s: %v", cfrIt, bs.name, probs)
	}
}

func run(
	cfrIt int,
	execCh chan<- StartupTask,
	wg *sync.WaitGroup,
	stats *cfr.CFRStats,
	memoryBuffer *cfr.BinaryMemoryBuffer,
	batchExecutor *cfr.GRPCBatchExecutor,
	benchStates []benchmarkState,
) {
	// Traverse
	stats.TreesTraversed.Store(0)
	elapsed := bench.MeasureExec(func() {
		for range TRAVERSE_ITERS {
			for pid := range 3 {
				wg.Add(1)
				execCh <- StartupTask{PlayerId: pid, CfrIter: cfrIt}
			}
		}
		wg.Wait()
	})

	// Drain writer to ensure all samples are on disk before training
	memoryBuffer.Drain()

	log.Printf("[CFR %d] Traversed in %s. Regret buffer: [%d, %d, %d]",
		cfrIt, elapsed,
		memoryBuffer.Count(0), memoryBuffer.Count(1), memoryBuffer.Count(2),
	)

	// Save networks
	if err := batchExecutor.Save(); err != nil {
		log.Fatalf("failed to save networks: %v", err)
	}

	// Train advantage networks — Python reads binary file directly
	advIters := ADV_TRAIN_ITERS
	if cfrIt < ADV_WARMUP_CFR_ITERS {
		advIters = ADV_TRAIN_WARMUP
	}
	elapsed = bench.MeasureExec(func() {
		var twg sync.WaitGroup
		for pid := range 3 {
			twg.Add(1)
			go func(pid int) {
				defer twg.Done()
				avgLoss, err := batchExecutor.TrainDirect(pid, BATCH_SIZE, advIters, memoryBuffer.FilePath(), MAX_SAMPLES)
				if err != nil {
					log.Fatalf("failed to train advantage P%d: %v", pid, err)
				}
				log.Printf("[CFR %d] Advantage P%d avg loss: %.6f", cfrIt, pid, avgLoss)
			}(pid)
		}
		twg.Wait()
	})
	log.Printf("[CFR %d] Advantage train finished in %s", cfrIt, elapsed)

	// Benchmark
	logBenchmarkStrategies(cfrIt, benchStates, batchExecutor)
}

func main() {
	runtime.SetBlockProfileRate(1)
	runtime.SetMutexProfileFraction(1)

	go func() {
		log.Println("pprof server on :6060")
		http.ListenAndServe(":6060", nil)
	}()

	// Buffers
	tempDir := os.Getenv("TEMP_DIR")
	if tempDir == "" {
		log.Fatal("TEMP_DIR environment variable is not set")
	}

	memoryBuffer, err := cfr.NewBinaryMemoryBuffer(filepath.Join(tempDir, "regret_buffer.bin"), MAX_SAMPLES, 256)
	if err != nil {
		log.Fatal(err)
	}
	defer memoryBuffer.Close()

	// Strategy buffer disabled — will be trained separately after advantage networks converge
	strategyBuffer := &cfr.NoopStrategyMemory{}

	log.Printf("Regret buffer: [%d, %d, %d]", memoryBuffer.Count(0), memoryBuffer.Count(1), memoryBuffer.Count(2))

	// Checkpoint
	cp := loadCheckpoint(tempDir)
	log.Printf("Loaded checkpoint: cfr_iteration=%d", cp.CfrIteration)

	// Neural
	neuralAddr := os.Getenv("NEURAL_ADDR")
	if neuralAddr == "" {
		neuralAddr = "127.0.0.1:1338"
	}

	actionsCache := &cfr.EmptyCache{}
	batchExecutor, err := cfr.NewGrpcBatchExecutor(neuralAddr, 4500, 50000, time.Millisecond*100)
	if err != nil {
		log.Fatal(err)
	}
	stats := &cfr.CFRStats{
		NodesVisited:   atomic.Int32{},
		TreesTraversed: atomic.Int32{},
	}
	actor := cfr.NewDeepCFRActor(actionsCache, batchExecutor)
	benchStates := buildBenchmarkStates()

	// Worker pool
	rng := rand.New(rand.NewSource(time.Now().UnixMilli()))
	var rngMut sync.Mutex
	execCh := make(chan StartupTask, 10)
	var wg sync.WaitGroup

	for tID := range TRAVERSE_THREADS {
		go func() {
			rngMut.Lock()
			game := nolimitholdem.NewGame(nolimitholdem.GameConfig{
				RandomSeed:      int64(rng.Int()) + int64(tID),
				ChipsForEach:    70,
				NumPlayers:      3,
				SmallBlindChips: 5,
			})
			traverser := cfr.New(
				int64(rng.Int())+int64(tID),
				game, actor, memoryBuffer, strategyBuffer, stats,
			)
			rngMut.Unlock()

			for task := range execCh {
				if _, err := traverser.TraverseTree(task.PlayerId, task.CfrIter); err != nil {
					log.Fatalf("failed to traverse tree: %v", err)
				}
				wg.Done()
			}
		}()
	}

	// CFR loop
	_, cancel := context.WithCancel(context.Background())
	go func() {
		for cfrIt := cp.CfrIteration; cfrIt < CFR_ITERS; cfrIt++ {
			iterElapsed := bench.MeasureExec(func() {
				run(cfrIt, execCh, &wg, stats, memoryBuffer, batchExecutor, benchStates)
			})
			log.Printf("CFR Iteration %d finished in %s", cfrIt, iterElapsed)
			if err := saveCheckpoint(tempDir, Checkpoint{CfrIteration: cfrIt + 1}); err != nil {
				log.Printf("WARNING: failed to save checkpoint: %v", err)
			}
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	cancel()
}
