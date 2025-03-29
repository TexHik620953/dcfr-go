package random

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func TestSample(t *testing.T) {
	values := map[int32]float32{
		0: 0.1,
		1: 0.1,
		2: 0.5,
		3: 0.3,
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	hist := map[int32]int{}
	for range 10000 {
		t := Sample(rng, values)
		v, ex := hist[t]
		if !ex {
			hist[t] = 1
		} else {
			hist[t] = v + 1
		}
	}
	fmt.Println(hist)
}
