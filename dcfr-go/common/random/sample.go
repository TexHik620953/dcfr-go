package random

import (
	"fmt"
	"math/rand"
)

func Sample(rand *rand.Rand, probs map[int32]float32) (int32, error) {
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
	if sum < 0.95 || sum > 1.05 {
		return 0, fmt.Errorf("invalid probs sum != 1")
	}
	r := rand.Float32()
	var cumulativeProb float32 = 0.0
	for _, ap := range actions {
		cumulativeProb += ap.prob
		if r < cumulativeProb {
			return ap.val, nil
		}
	}

	return actions[len(actions)-1].val, nil
}
