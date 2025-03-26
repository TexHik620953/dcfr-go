package main

import (
	"dcfr-go/nolimitholdem"
	"fmt"
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
		RandomSeed:      42,
		ChipsForEach:    500,
		NumPlayers:      3,
		SmallBlindChips: 20,
	})
	for {
		for !game.IsOver() {
			legalActions := game.LegalActions()
			if _, ok := legalActions[nolimitholdem.ACTION_CHECK_CALL]; ok {
				game.Step(nolimitholdem.ACTION_CHECK_CALL)
			} else {
				game.Step(nolimitholdem.ACTION_FOLD)
			}
		}
		payoff := game.GetPayoffs()
		fmt.Println(payoff)
		game.Reset()
	}
}
