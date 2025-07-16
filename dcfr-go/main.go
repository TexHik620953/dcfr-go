package main

import (
	"dcfr-go/cfr"
	"dcfr-go/common/bench"
	"dcfr-go/nolimitholdem"
	"log"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

type StartupTask struct {
	PlayerId int
	CfrIter  int
}

func main() {

	rng := rand.New(rand.NewSource(time.Now().UnixMilli()))
	var rngMut sync.Mutex

	memoryBuffer, err := cfr.NewMemoryBuffer(7_000_000, 0.2, "host=pg user=postgres password=HermanFuLLer dbname=postgres port=5432")
	if err != nil {
		log.Fatal(err)
	}
	actionsCache := cfr.NewActionsCache(5_000_000, 0.1)
	batchExecutor, err := cfr.NewGrpcBatchExecutor("neural:1338", 30000, 50000)
	stats := &cfr.CFRStats{
		NodesVisited:   atomic.Int32{},
		TreesTraversed: atomic.Int32{},
	}
	if err != nil {
		log.Fatal(err)
	}
	actor := cfr.NewDeepCFRActor(actionsCache, batchExecutor, 3)

	const CFR_ITERS = 1000
	const TRAVERSE_ITERS = 100000
	const TRAIN_ITERS = 20

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
			err := batchExecutor.Reset()
			if err != nil {
				log.Fatalf("failed to reset networks: %v", err)
			}

			// Train
			elapsed = bench.MeasureExec(func() {
				for player_id := range 3 {
					for range TRAIN_ITERS {
						batch := memoryBuffer.GetSamples(player_id, 50000)
						if len(batch) == 0 {
							continue
						}
						_, err := batchExecutor.Train(player_id, batch)
						if err != nil {
							log.Fatalf("failed to train: %v", err)
						}
					}
				}
			})
			log.Printf("[CFR_IT: %d] Train iteration finished in %s", cfr_it, elapsed)
		})

		log.Printf("CFR Iteration %d finished in %s", cfr_it, cfr_it_elapsed)
	}

}
