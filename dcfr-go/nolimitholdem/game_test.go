package nolimitholdem

import (
	"math/rand"
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
	actor := NewRandomActor(rand.New(rand.NewSource(config.RandomSeed)))

	b.ResetTimer()
	startTime := time.Now()
	for i := 0; i < b.N; i++ {
		game.Reset()
		stepsInHand := 0
		for !game.IsOver() {
			state := game.GetState(game.CurrentPlayer())
			action := actor.GetAction(state)
			game.Step(action)
			stepsInHand++
		}

		gamesCompleted++
		totalSteps += stepsInHand
	}

	duration := time.Since(startTime)
	gamesPerSecond := float64(gamesCompleted) / duration.Seconds()

	b.ReportMetric(float64(totalSteps)/float64(b.N), "steps/game")

	b.ReportMetric(gamesPerSecond, "games/sec")
	b.ReportMetric(duration.Seconds()/float64(gamesCompleted), "sec/game")
}

func TestPokerGameRollback(t *testing.T) {
	config := GameConfig{
		RandomSeed:      42,
		ChipsForEach:    500,
		NumPlayers:      3,
		SmallBlindChips: 20,
	}

	actor := NewRandomActor(rand.New(rand.NewSource(config.RandomSeed)))
	game := NewGame(config)
	game.Reset()

	for !game.IsOver() {
		state := game.GetState(game.CurrentPlayer())
		action := actor.GetAction(state)
		game.Step(action)
	}

}

func TestPokerGameReset(t *testing.T) {
	config := GameConfig{
		RandomSeed:      42,
		ChipsForEach:    500,
		NumPlayers:      3,
		SmallBlindChips: 20,
	}

	actor := NewRandomActor(rand.New(rand.NewSource(config.RandomSeed)))
	game := NewGame(config)
	game.Reset()

	for !game.IsOver() {
		state := game.GetState(game.CurrentPlayer())
		action := actor.GetAction(state)
		game.Step(action)
	}

}
