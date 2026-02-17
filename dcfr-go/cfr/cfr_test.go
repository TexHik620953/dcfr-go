package cfr_test

import (
	"dcfr-go/cfr"
	"dcfr-go/nolimitholdem"
	"fmt"
	"math/rand"
	"sync/atomic"
	"testing"
)

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
		ChipsForEach:    200,
		NumPlayers:      3,
		SmallBlindChips: 5,
	})

	actor := nolimitholdem.NewRandomActor(rand.New(rand.NewSource(42)))
	memBuff := &MemoryBufferMock{}

	stats := &cfr.CFRStats{
		NodesVisited:   atomic.Int32{},
		TreesTraversed: atomic.Int32{},
	}

	traverser := cfr.New(
		int64(42),
		game,
		actor,
		memBuff,
		stats,
	)

	traverser.TraverseTree(0, 0)
	fmt.Println(0)
}
