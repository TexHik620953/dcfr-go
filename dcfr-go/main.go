package main

import (
	"dcfr-go/cfr"
	"dcfr-go/nolimitholdem"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	memoryBuffer := cfr.NewMemoryBuffer(700000, 0.2)
	actionsCache := cfr.NewActionsCache(10000000, 0.1)
	batchExecutor, err := cfr.NewGrpcBatchExecutor("localhost:1338", 15000, 20000)
	stats := &cfr.CFRStats{
		NodesVisited:   atomic.Int32{},
		TreesTraversed: atomic.Int32{},
	}
	if err != nil {
		log.Fatal(err)
	}
	actor := cfr.NewDeepCFRActor(actionsCache, batchExecutor)

	threads := 70000

	execCh := make(chan struct{}, threads)
	var wg sync.WaitGroup
	for tID := range threads {
		go func() {
			game := nolimitholdem.NewGame(nolimitholdem.GameConfig{
				RandomSeed:      int64(44 + tID),
				ChipsForEach:    70,
				NumPlayers:      3,
				SmallBlindChips: 5,
			})
			traverser := cfr.New(
				int64(44+tID),
				game,
				actor,
				memoryBuffer,
				stats,
			)

			//Traversing tree
			for {
				<-execCh
				for ply := range game.PlayersCount() {
					_, err := traverser.TraverseTree(ply)
					if err != nil {
						log.Fatalf("failed to traverse tree: %v", err)
					}
					game.Reset()
				}
				wg.Done()
			}

		}()
	}

	go func() {
		for {
			log.Printf("Trees traversed: %d, nodes visited: %d", stats.TreesTraversed.Load(), stats.NodesVisited.Load())
			<-time.After(time.Second * 30)
		}
	}()

	for {
		// Run threads
		for range threads {
			wg.Add(1)
			execCh <- struct{}{}
		}
		//Wait them to finish
		wg.Wait()
		log.Printf("Finished traversing, memory size: [%d %d %d]", memoryBuffer.Count(0), memoryBuffer.Count(1), memoryBuffer.Count(2))

		BATCH_SIZE := 5000
		log.Printf("Training")
		//Train network
		for range 50 {
			// Train
			for ply := range 3 {
				batch := memoryBuffer.GetSamples(ply, BATCH_SIZE)
				if len(batch) == 0 {
					continue
				}
				_, err := batchExecutor.Train(ply, batch)
				if err != nil {
					log.Fatalf("failed to train: %v", err)
				}
			}
		}
		batchExecutor.TrainAvg()

		actionsCache.Clear(-1)
		actionsCache.Clear(0)
		actionsCache.Clear(1)
		actionsCache.Clear(2)
	}

}
