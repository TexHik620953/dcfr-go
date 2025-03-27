package nolimitholdem

import (
	"math/rand"

	"github.com/idsulik/go-collections/v3/queue"
)

type Deck struct {
	rand *rand.Rand
	q    *queue.Queue[Card]
}

func (d *Deck) DeepCopy() *Deck {
	if d == nil {
		return nil
	}

	// Создаем новый генератор случайных чисел с тем же seed
	var newRand *rand.Rand
	if d.rand != nil {
		newRand = rand.New(rand.NewSource(d.rand.Int63()))
	}

	// Создаем новую очередь и копируем в нее все карты
	newQueue := queue.New[Card](4 * 13)
	if d.q != nil {
		d.q.ForEach(func(c Card) {
			newQueue.Enqueue(c)

		})
	}

	return &Deck{
		rand: newRand,
		q:    newQueue,
	}
}

func NewDeck(rand *rand.Rand) *Deck {
	h := &Deck{
		rand: rand,
	}
	h.Reset()
	return h
}

func (h *Deck) Reset() {
	h.q = queue.New[Card](4 * 13)
	perm := h.rand.Perm(4 * 13)
	for _, v := range perm {
		h.q.Enqueue(Card(v))
	}
}

func (h *Deck) Get() Card {
	val, ex := h.q.Dequeue()
	if !ex {
		panic("deck is empty")
	}
	return val
}
