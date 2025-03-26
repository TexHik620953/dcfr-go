package nolimitholdem

import (
	"testing"
	"time"
)

func BenchmarkPokerGameDetailed(b *testing.B) {
	desiredN := 5000
	if b.N < desiredN {
		b.N = desiredN
	}

	config := GameConfig{
		RandomSeed:      42,
		ChipsForEach:    500,
		NumPlayers:      3,
		SmallBlindChips: 20,
	}

	var gamesCompleted int
	var totalSteps int

	game := NewGame(config)

	b.ResetTimer()
	startTime := time.Now()
	for i := 0; i < b.N; i++ {
		stepsInHand := 0

		for !game.IsOver() {
			legalActions := game.LegalActions()
			if _, ok := legalActions[ACTION_CHECK_CALL]; ok {
				game.Step(ACTION_CHECK_CALL)
			} else {
				game.Step(ACTION_FOLD)
			}
			stepsInHand++
		}

		gamesCompleted++
		totalSteps += stepsInHand
		game.Reset()
	}

	duration := time.Since(startTime)
	gamesPerSecond := float64(gamesCompleted) / duration.Seconds()

	b.ReportMetric(float64(totalSteps)/float64(b.N), "steps/game")

	b.ReportMetric(gamesPerSecond, "games/sec")
	b.ReportMetric(duration.Seconds()/float64(gamesCompleted), "sec/game")
}
