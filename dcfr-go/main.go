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

	memoryBuffer := cfr.NewMemoryBuffer(10000000, 0.2)
	actionsCache := cfr.NewActionsCache(20000000, 0.1)
	batchExecutor, err := cfr.NewGrpcBatchExecutor("localhost:1338", 15000, 30000)
	stats := &cfr.CFRStats{
		NodesVisited:   atomic.Int32{},
		TreesTraversed: atomic.Int32{},
	}
	if err != nil {
		log.Fatal(err)
	}
	actor := cfr.NewDeepCFRActor(actionsCache, batchExecutor)

	threads := 20000
	execCh := make(chan StartupTask, threads)
	var wg sync.WaitGroup
	for tID := range threads {
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
	for cfr_it := range 100000 {

		elapsed := bench.MeasureExec(func() {
			// Run threads
			for range threads {
				for ply_id := range 3 {
					wg.Add(1)
					execCh <- StartupTask{
						PlayerId: ply_id,
						CfrIter:  cfr_it,
					}
				}
			}
			// Wait them to finish
			wg.Wait()
		})

		log.Printf("Finished traversing in %s, memory: [%d, %d, %d, (%d)] trees: %d, nodes: %d",
			elapsed,
			memoryBuffer.Count(0),
			memoryBuffer.Count(1),
			memoryBuffer.Count(2),
			memoryBuffer.Count(-1),
			stats.TreesTraversed.Load(),
			stats.NodesVisited.Load(),
		)

		elapsed = bench.MeasureExec(func() {
			for range 100 {
				// Train for every player
				for ply_id := range 3 {
					batch := memoryBuffer.GetSamples(ply_id, 10000)
					if len(batch) == 0 {
						continue
					}
					_, err := batchExecutor.Train(ply_id, batch)
					if err != nil {
						log.Fatalf("failed to train: %v", err)
					}
				}
				// Train average
				batch := memoryBuffer.GetSamples(-1, 20000)
				if len(batch) == 0 {
					continue
				}
				_, err := batchExecutor.TrainAvg(batch)
				if err != nil {
					log.Fatalf("failed to train: %v", err)
				}
			}
		})
		log.Printf("Train finished in %s", elapsed)
		err := batchExecutor.Save()
		if err != nil {
			log.Fatalf("failed to save: %v", err)
		}
		actionsCache.Clear()
	}

}
