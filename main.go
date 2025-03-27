package main

import (
	"dcfr-go/cfr"
	"dcfr-go/nolimitholdem"
	"math/rand"
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

	game := nolimitholdem.NewGame(nolimitholdem.GameConfig{
		RandomSeed:      44,
		ChipsForEach:    250,
		NumPlayers:      3,
		SmallBlindChips: 5,
	})

	traverser := cfr.New(game, nolimitholdem.NewRandomActor(rand.New(rand.NewSource(77))))

	traverser.TraverseTree(0)
}
