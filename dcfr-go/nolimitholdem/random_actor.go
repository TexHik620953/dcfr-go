package nolimitholdem

import (
	"math/rand"
)

type RandomActor struct {
	rand *rand.Rand
}

func NewRandomActor(rand *rand.Rand) *RandomActor {
	h := &RandomActor{
		rand: rand,
	}
	return h
}
func (h *RandomActor) GetAction(state *GameState) Action {
	act := h.GetProbs(state)
	maxV := float32(0)
	maxAct := ACTION_CHECK_CALL
	for a, v := range act {
		if v > maxV {
			maxV = v
			maxAct = a
		}
	}
	return maxAct
}

func (h *RandomActor) GetProbs(state *GameState) map[Action]float32 {
	r := make(map[Action]float32)
	sum := float32(0)
	for act, _ := range state.LegalActions {
		v := h.rand.Float32()
		r[act] = v
		sum += v
	}
	for act, v := range r {
		r[act] = v / sum
	}
	return r
}
