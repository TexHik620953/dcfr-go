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

	"github.com/schollz/progressbar/v3"
)

type StartupTask struct {
	PlayerId int
	CfrIter  int
}

type benchmarkState struct {
	name  string
	state *cfr.CFRState
}

// buildBenchmarkStates creates fixed game states to monitor strategy evolution.
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
		// Preflop: AA (Ah As) — should raise/all-in heavily
		makeState("AA_preflop",
			[2]nolimitholdem.Card{nolimitholdem.NewCard(12, 0), nolimitholdem.NewCard(12, 1)},
			nil, nolimitholdem.STAGE_PREFLOP,
			[3]int32{0, 5, 10}, [3]int32{70, 65, 60}),

		// Preflop: 72o (7h 2s) — should fold most of the time
		makeState("72o_preflop",
			[2]nolimitholdem.Card{nolimitholdem.NewCard(0, 1), nolimitholdem.NewCard(5, 0)},
			nil, nolimitholdem.STAGE_PREFLOP,
			[3]int32{0, 5, 10}, [3]int32{70, 65, 60}),

		// Flop: KK with K on board (set) — should bet/raise
		makeState("KK_set_flop",
			[2]nolimitholdem.Card{nolimitholdem.NewCard(11, 0), nolimitholdem.NewCard(11, 1)},
			[]nolimitholdem.Card{nolimitholdem.NewCard(11, 2), nolimitholdem.NewCard(6, 0), nolimitholdem.NewCard(2, 1)},
			nolimitholdem.STAGE_FLOP,
			[3]int32{20, 20, 20}, [3]int32{50, 50, 50}),

		// River: low cards, missed draw — should check/fold
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

func logBenchmarkStrategies(cfr_it int, states []benchmarkState, executor *cfr.GRPCBatchExecutor) {
	for _, bs := range states {
		ch := executor.EnqueueGetStrategy(bs.state)
		result := <-ch

		probs := make([]string, 0, len(result.Strategy))
		for action, prob := range result.Strategy {
			if prob > 0.01 {
				probs = append(probs, fmt.Sprintf("%s=%.2f", nolimitholdem.Action2string[action], prob))
			}
		}
		log.Printf("[CFR_IT: %d] BENCH %s: %v", cfr_it, bs.name, probs)
	}
}

func main() {
	runtime.SetBlockProfileRate(1)
	runtime.SetMutexProfileFraction(1)

	go func() {
		log.Println("pprof server on :6060")
		http.ListenAndServe(":6060", nil)
	}()

	rng := rand.New(rand.NewSource(time.Now().UnixMilli()))
	var rngMut sync.Mutex

	os.MkdirAll("data", 0755)
	tempDir := os.Getenv("TEMP_DIR")
	if tempDir == "" {
		log.Fatal("TEMP_DIR environment variable is not set")
	}
	memoryBuffer, err := cfr.NewSQLiteMemoryBuffer(filepath.Join(tempDir, "regret_buffer.db"), 1_500_000)
	if err != nil {
		log.Fatal(err)
	}
	defer memoryBuffer.Close()

	strategyBuffer, err := cfr.NewSQLiteStrategyMemoryBuffer(filepath.Join(tempDir, "strategy_buffer.db"), 1_500_000)
	if err != nil {
		log.Fatal(err)
	}
	defer strategyBuffer.Close()

	log.Printf("Regret buffer: games=[%d, %d, %d]",
		memoryBuffer.Count(0), memoryBuffer.Count(1), memoryBuffer.Count(2),
	)
	log.Printf("Strategy buffer: games=[%d, %d, %d]",
		strategyBuffer.Count(0), strategyBuffer.Count(1), strategyBuffer.Count(2),
	)

	dataDir := tempDir
	cp := loadCheckpoint(dataDir)
	log.Printf("Loaded checkpoint: cfr_iteration=%d", cp.CfrIteration)

	neuralAddr := os.Getenv("NEURAL_ADDR")
	if neuralAddr == "" {
		neuralAddr = "127.0.0.1:1338"
	}

	actionsCache := &cfr.EmptyCache{}
	batchExecutor, err := cfr.NewGrpcBatchExecutor(neuralAddr, 5000, 50000, time.Millisecond*100)
	stats := &cfr.CFRStats{
		NodesVisited:   atomic.Int32{},
		TreesTraversed: atomic.Int32{},
	}
	if err != nil {
		log.Fatal(err)
	}
	actor := cfr.NewDeepCFRActor(actionsCache, batchExecutor)
	benchStates := buildBenchmarkStates()

	const TRAVERSE_THREADS = 60000
	const CFR_ITERS = 1000
	const TRAVERSE_ITERS = 20000
	const TRAIN_ITERS = 400
	const BATCH_SIZE = 5000

	ctx, cancel := context.WithCancel(context.Background())
	_ = ctx
	// Create workers threads
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
				game,
				actor,
				memoryBuffer,
				strategyBuffer,
				stats,
			)
			rngMut.Unlock()

			//Traversing tree
			for {
				iter_task := <-execCh
				_, err := traverser.TraverseTree(iter_task.PlayerId, iter_task.CfrIter)
				if err != nil {
					log.Fatalf("failed to traverse tree: %v", err)
				}
				wg.Done()
			}

		}()
	}
	go func() {
		totalTraversals := 3 * TRAVERSE_ITERS // 3 players × TRAVERSE_ITERS

		// CFR iterations
		for cfr_it := cp.CfrIteration; cfr_it < CFR_ITERS; cfr_it++ {
			cfr_it_elapsed := bench.MeasureExec(func() {
				// Traverse with progress bar
				stats.TreesTraversed.Store(0)
				bar := progressbar.NewOptions(totalTraversals,
					progressbar.OptionSetDescription(fmt.Sprintf("[CFR %d] Traversing", cfr_it)),
					progressbar.OptionShowCount(),
					progressbar.OptionShowIts(),
					progressbar.OptionSetWidth(30),
					progressbar.OptionThrottle(time.Second),
					progressbar.OptionClearOnFinish(),
				)

				elapsed := bench.MeasureExec(func() {
					// Update bar from stats counter
					traverseDone := make(chan struct{})
					go func() {
						ticker := time.NewTicker(500 * time.Millisecond)
						defer ticker.Stop()
						prev := 0
						for {
							select {
							case <-traverseDone:
								return
							case <-ticker.C:
								cur := int(stats.TreesTraversed.Load())
								if cur > prev {
									bar.Add(cur - prev)
									prev = cur
								}
							}
						}
					}()

					for range TRAVERSE_ITERS {
						for player_id := range 3 {
							wg.Add(1)
							execCh <- StartupTask{
								PlayerId: player_id,
								CfrIter:  cfr_it,
							}
						}
					}
					wg.Wait()
					close(traverseDone)
					bar.Finish()
				})
				log.Printf("[CFR_IT: %d] Finished traversing in %s. Games memory size: [%d, %d, %d]",
					cfr_it,
					elapsed,
					memoryBuffer.Count(0),
					memoryBuffer.Count(1),
					memoryBuffer.Count(2),
				)

				//actionsCache.Clear()
				err := batchExecutor.Save()
				if err != nil {
					log.Fatalf("failed to save networks: %v", err)
				}

				// Train advantage network
				elapsed = bench.MeasureExec(func() {
					for player_id := range 3 {
						trainBar := progressbar.NewOptions(TRAIN_ITERS,
							progressbar.OptionSetDescription(fmt.Sprintf("[CFR %d] Advantage P%d", cfr_it, player_id)),
							progressbar.OptionShowCount(),
							progressbar.OptionSetWidth(25),
							progressbar.OptionClearOnFinish(),
						)
						var lossSum float32
						var lossCount int
						for tIter := range TRAIN_ITERS {
							batch := memoryBuffer.GetSamples(player_id, BATCH_SIZE)
							if len(batch) == 0 {
								trainBar.Add(1)
								continue
							}
							loss, err := batchExecutor.Train(player_id, batch)
							if err != nil {
								log.Fatalf("failed to train: %v", err)
							}
							lossSum += loss
							lossCount++
							trainBar.Describe(fmt.Sprintf("[CFR %d] Advantage P%d loss=%.0f", cfr_it, player_id, loss))
							trainBar.Set(tIter + 1)
						}
						trainBar.Finish()
						if lossCount > 0 {
							log.Printf("[CFR_IT: %d] Advantage player %d avg loss: %.6f", cfr_it, player_id, lossSum/float32(lossCount))
						}
					}
				})
				log.Printf("[CFR_IT: %d] Advantage train finished in %s", cfr_it, elapsed)

				// Train average strategy network
				elapsed = bench.MeasureExec(func() {
					for player_id := range 3 {
						trainBar := progressbar.NewOptions(TRAIN_ITERS,
							progressbar.OptionSetDescription(fmt.Sprintf("[CFR %d] AvgStrategy P%d", cfr_it, player_id)),
							progressbar.OptionShowCount(),
							progressbar.OptionSetWidth(25),
							progressbar.OptionClearOnFinish(),
						)
						var lossSum float32
						var lossCount int
						for tIter := range TRAIN_ITERS {
							batch := strategyBuffer.GetSamples(player_id, 10000)
							if len(batch) == 0 {
								trainBar.Add(1)
								continue
							}
							loss, err := batchExecutor.TrainAvgStrategy(player_id, batch)
							if err != nil {
								log.Fatalf("failed to train avg strategy: %v", err)
							}
							lossSum += loss
							lossCount++
							trainBar.Describe(fmt.Sprintf("[CFR %d] AvgStrategy P%d loss=%.4f", cfr_it, player_id, loss))
							trainBar.Set(tIter + 1)
						}
						trainBar.Finish()
						if lossCount > 0 {
							log.Printf("[CFR_IT: %d] AvgStrategy player %d avg loss: %.6f", cfr_it, player_id, lossSum/float32(lossCount))
						}
					}
				})
				log.Printf("[CFR_IT: %d] Avg strategy train finished in %s", cfr_it, elapsed)

				// Log benchmark strategies
				logBenchmarkStrategies(cfr_it, benchStates, batchExecutor)
			})

			log.Printf("CFR Iteration %d finished in %s", cfr_it, cfr_it_elapsed)
			if err := saveCheckpoint(dataDir, Checkpoint{CfrIteration: cfr_it + 1}); err != nil {
				log.Printf("WARNING: failed to save checkpoint: %v", err)
			}
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	cancel()
}
