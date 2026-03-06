package main

import (
	"context"
	"dcfr-go/cfr"
	"dcfr-go/common/bench"
	"dcfr-go/nolimitholdem"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

type StartupTask struct {
	PlayerId int
	CfrIter  int
}

func main() {

	rng := rand.New(rand.NewSource(time.Now().UnixMilli()))
	var rngMut sync.Mutex

	memoryBuffer, err := cfr.NewMemoryBuffer(1_000_000) //10M
	if err != nil {
		log.Fatal(err)
	}
	strategyBuffer := cfr.NewStrategyMemoryBuffer(1_000_000) //10M

	err = memoryBuffer.Load()
	if err != nil {
		log.Printf("failed to load memory buffer: %v, creating new one", err)
	}

	actionsCache := cfr.NewActionsCache(200_000, 0.1)
	batchExecutor, err := cfr.NewGrpcBatchExecutor("127.0.0.1:1338", 500, 10000)
	stats := &cfr.CFRStats{
		NodesVisited:   atomic.Int32{},
		TreesTraversed: atomic.Int32{},
	}
	if err != nil {
		log.Fatal(err)
	}
	actor := cfr.NewDeepCFRActor(actionsCache, batchExecutor)

	const CFR_ITERS = 1000
	const TRAVERSE_ITERS = 20000
	const TRAIN_ITERS = 200 //2000

	ctx, cancel := context.WithCancel(context.Background())
	_ = ctx
	// Create workers threads
	execCh := make(chan StartupTask, 10)
	var wg sync.WaitGroup
	for tID := range TRAVERSE_ITERS {
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
		// CFR iterations
		for cfr_it := 0; cfr_it < CFR_ITERS; cfr_it++ {
			cfr_it_elapsed := bench.MeasureExec(func() {
				// Traverse
				elapsed := bench.MeasureExec(func() {
					for player_id := range 3 {
						for range TRAVERSE_ITERS {
							wg.Add(1)
							execCh <- StartupTask{
								PlayerId: player_id,
								CfrIter:  cfr_it,
							}
						}
					}
					wg.Wait()
				})
				log.Printf("[CFR_IT: %d] Finished traversing in %s. Games memory size: [%d, %d, %d]",
					cfr_it,
					elapsed,
					memoryBuffer.Count(0),
					memoryBuffer.Count(1),
					memoryBuffer.Count(2),
				)

				actionsCache.Clear()
				// Reset network
				err := batchExecutor.Save()
				if err != nil {
					log.Fatalf("failed to save networks: %v", err)
				}
				// Train
				elapsed = bench.MeasureExec(func() {
					for player_id := range 3 {
						var lossSum float32
						var lossCount int
						for tIter := range TRAIN_ITERS {
							batch := memoryBuffer.GetSamples(player_id, 10000)
							if len(batch) == 0 {
								continue
							}
							loss, err := batchExecutor.Train(player_id, batch)
							if err != nil {
								log.Fatalf("failed to train: %v", err)
							}
							lossSum += loss
							lossCount++
							if tIter%100 == 0 {
								fmt.Printf("Training player %d: %d/%d\n", player_id, tIter, TRAIN_ITERS)
							}
						}
						if lossCount > 0 {
							log.Printf("[CFR_IT: %d] Advantage player %d avg loss: %.6f", cfr_it, player_id, lossSum/float32(lossCount))
						}
					}
				})
				log.Printf("[CFR_IT: %d] Advantage train finished in %s", cfr_it, elapsed)

				// Train average strategy network
				elapsed = bench.MeasureExec(func() {
					for player_id := range 3 {
						var lossSum float32
						var lossCount int
						for tIter := range TRAIN_ITERS {
							batch := strategyBuffer.GetSamples(player_id, 10000)
							if len(batch) == 0 {
								continue
							}
							loss, err := batchExecutor.TrainAvgStrategy(player_id, batch)
							if err != nil {
								log.Fatalf("failed to train avg strategy: %v", err)
							}
							lossSum += loss
							lossCount++
							if tIter%100 == 0 {
								fmt.Printf("Training avg strategy player %d: %d/%d\n", player_id, tIter, TRAIN_ITERS)
							}
						}
						if lossCount > 0 {
							log.Printf("[CFR_IT: %d] AvgStrategy player %d avg loss: %.6f", cfr_it, player_id, lossSum/float32(lossCount))
						}
					}
				})
				log.Printf("[CFR_IT: %d] Avg strategy train finished in %s", cfr_it, elapsed)
			})

			log.Printf("CFR Iteration %d finished in %s", cfr_it, cfr_it_elapsed)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	cancel()
}
