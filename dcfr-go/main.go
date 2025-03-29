package main

import (
	"dcfr-go/cfr"
	"dcfr-go/nolimitholdem"
	"log"
)

const BATCH_SIZE = 10000

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

	game := nolimitholdem.NewGame(nolimitholdem.GameConfig{
		RandomSeed:      44,
		ChipsForEach:    90,
		NumPlayers:      3,
		SmallBlindChips: 5,
	})

	actor, err := cfr.NewDeepCFRActor("localhost:1338")
	if err != nil {
		log.Fatal(err)
	}

	traverser := cfr.New(game,
		actor,
		cfr.NewMemoryBuffer(50000, 0.05),
	)

	for {
		game.Reset()
		for ply := range game.PlayersCount() {
			payoffs, err := traverser.TraverseTree(ply, false)
			if err != nil {
				log.Fatalf("failed to traverse tree: %v", err)
			}
			log.Printf("Traversed tree for player %d with payoffs: ", ply)
			log.Print(payoffs)
			log.Printf("\n")
			// Clearing cache
			actor.ClearCache()
		}

		log.Printf("Training\n")
		for range 10 {
			// Train
			for ply := range game.PlayersCount() {
				batch := traverser.Memory.GetSamples(ply, BATCH_SIZE)
				if len(batch) == 0 {
					continue
				}
				_, err := actor.Train(ply, batch)
				if err != nil {
					log.Fatalf("failed to train: %v", err)
				}
			}
		}
		actor.TrainAvg()

	}

}
