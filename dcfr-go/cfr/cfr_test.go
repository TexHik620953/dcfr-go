package cfr_test

import (
	"dcfr-go/cfr"
	"dcfr-go/nolimitholdem"
	"fmt"
	"sync/atomic"
	"testing"
)

type ActorMock struct {
}

func (h *ActorMock) GetProbs(learnerId int, state *nolimitholdem.GameState) (nolimitholdem.Strategy, error) {
	return map[nolimitholdem.Action]float32{
		nolimitholdem.ACTION_CHECK_CALL: 0.85,
		nolimitholdem.ACTION_FOLD:       0.15,
	}, nil
}

type MemoryBufferMock struct {
}

func (h *MemoryBufferMock) AddStrategySample(state *nolimitholdem.GameState, strategy nolimitholdem.Strategy, iteration int) {

}
func (h *MemoryBufferMock) AddSample(
	playerID int,
	state *nolimitholdem.GameState,
	regrets map[nolimitholdem.Action]float32,
	iteration int,
) {

}

func TestTraverse(t *testing.T) {

	game := nolimitholdem.NewGame(nolimitholdem.GameConfig{
		RandomSeed:      int64(42),
		ChipsForEach:    70,
		NumPlayers:      3,
		SmallBlindChips: 5,
	})
	actorMock := &ActorMock{}
	memBuff := &MemoryBufferMock{}

	stats := &cfr.CFRStats{
		NodesVisited:   atomic.Int32{},
		TreesTraversed: atomic.Int32{},
	}

	traverser := cfr.New(
		int64(42),
		game,
		actorMock,
		memBuff,
		stats,
	)

	traverser.TraverseTree(0, 0)
	fmt.Println(0)
}
