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
	/*
		ctx := context.Background()
		cfg, err := appconfig.LoadAppConfig()
		if err != nil {
			log.Fatal(err.Error())
		}



		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
	*/

	/*
		f, err := os.Create("cpu.prof")
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	*/

	memoryBuffer := cfr.NewMemoryBuffer(1500000, 0.05)
	actionsCache := cfr.NewActionsCache(10000000, 0.1)
	batchExecutor, err := cfr.NewGrpcBatchExecutor("localhost:1338", 10000, 15000)
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
				ChipsForEach:    30,
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
				game.Reset()
				for ply := range game.PlayersCount() {
					_, err := traverser.TraverseTree(ply)
					if err != nil {
						log.Fatalf("failed to traverse tree: %v", err)
					}
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

		BATCH_SIZE := 15000
		log.Printf("Training")
		//Train network
		for range 90 {
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
