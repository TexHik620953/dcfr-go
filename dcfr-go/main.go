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

	memoryBuffer, err := cfr.NewMemoryBuffer(5_000_000, 0.2, "host=127.0.0.1 user=postgres password=HermanFuLLer dbname=postgres port=5432")
	if err != nil {
		log.Fatal(err)
	}

	err = memoryBuffer.Load()
	if err != nil {
		log.Printf("failed to load memory buffer: %v, creating new one", err)
	}

	actionsCache := cfr.NewActionsCache(5_000_000, 0.1)
	batchExecutor, err := cfr.NewGrpcBatchExecutor("127.0.0.1:1338", 10000, 20000)
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
	const TRAIN_ITERS = 1000

	ctx, cancel := context.WithCancel(context.Background())
	_ = ctx
	// Create workers threads
	execCh := make(chan StartupTask, TRAVERSE_ITERS)
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
		for cfr_it := 3; cfr_it < CFR_ITERS; cfr_it++ {
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
				log.Printf("[CFR_IT: %d] Finished traversing in %s. Memory size: [%d, %d, %d]",
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
				// Не делаем ресет, чтобы файнтьюнить сеть а не обучать сначала
				/*
					err = batchExecutor.Reset()
					if err != nil {
						log.Fatalf("failed to reset networks: %v", err)
					}
				*/
				// Train
				elapsed = bench.MeasureExec(func() {
					for player_id := range 3 {
						for tIter := range TRAIN_ITERS {
							batch := memoryBuffer.GetSamples(player_id, 5000)
							if len(batch) == 0 {
								continue
							}
							_, err := batchExecutor.Train(player_id, batch)
							if err != nil {
								log.Fatalf("failed to train: %v", err)
							}
							if tIter%10 == 0 {
								fmt.Printf("Train iterations: %d/%d\n", tIter, TRAIN_ITERS)
							}
						}
					}
				})
				log.Printf("[CFR_IT: %d] Train iteration finished in %s", cfr_it, elapsed)

				// Saving memory buffers
				err = memoryBuffer.Save()
				if err != nil {
					log.Printf("failed to save memory buffer: %v", err)
				}
			})

			log.Printf("CFR Iteration %d finished in %s", cfr_it, cfr_it_elapsed)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	cancel()
}
