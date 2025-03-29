package random

import (
	"math/rand"
)

func Sample(rand *rand.Rand, probs map[int32]float32) int32 {
	type actionProb struct {
		val  int32
		prob float32
	}
	var actions []actionProb
	var sum float32 = 0.0

	for val, prob := range probs {
		actions = append(actions, actionProb{val, prob})
		sum += prob
	}

	// Проверка корректности распределения (опционально)
	if sum < 0.99 || sum > 1.01 {
		panic("invalid probs sum != 1")
	}
	r := rand.Float32()
	var cumulativeProb float32 = 0.0
	for _, ap := range actions {
		cumulativeProb += ap.prob
		if r < cumulativeProb {
			return ap.val
		}
	}

	return actions[len(actions)-1].val
}
