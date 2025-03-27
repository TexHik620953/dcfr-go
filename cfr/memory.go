package cfr

import (
	"dcfr-go/common/linq"
	"dcfr-go/nolimitholdem"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// Sample хранит данные одного состояния для обучения
type Sample struct {
	State     *nolimitholdem.GameState         // Состояние игры
	Regrets   map[nolimitholdem.Action]float32 // Сожаления для каждого действия
	Weight    float32                          // Вес примера (reach probability)
	Iteration int                              // Итерация, на которой был собран пример
}

// MemoryBuffer кэширует данные для обучения нейросетей
type MemoryBuffer struct {
	mu         sync.RWMutex
	samples    map[int][]*Sample // samples[playerID] -> []*Sample
	maxSamples int               // Максимальное число примеров на игрока
	pruneRatio float32           // Доля старых примеров для удаления
	rng        *rand.Rand        // Генератор случайных чисел
}

// NewMemoryBuffer создает буфер с настройками
func NewMemoryBuffer(maxSamples int, pruneRatio float32) *MemoryBuffer {
	return &MemoryBuffer{
		samples:    make(map[int][]*Sample),
		maxSamples: maxSamples,
		pruneRatio: pruneRatio,
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// AddSample добавляет новый пример для игрока
func (m *MemoryBuffer) AddSample(
	playerID int,
	state *nolimitholdem.GameState,
	regrets map[nolimitholdem.Action]float32,
	reachProb float32,
	iteration int,
) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Создаем новый пример
	sample := &Sample{
		State:     state.Clone(),
		Regrets:   linq.CopyMap(regrets),
		Weight:    reachProb,
		Iteration: iteration,
	}

	// Добавляем в хранилище
	m.samples[playerID] = append(m.samples[playerID], sample)

	// Проверяем необходимость очистки
	if len(m.samples[playerID]) > m.maxSamples {
		m.pruneOldSamples(playerID)
	}
}

// GetSamples возвращает батч примеров для обучения
func (m *MemoryBuffer) GetSamples(playerID int, batchSize int) []*Sample {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.samples[playerID]) == 0 {
		return nil
	}

	// Выбираем случайные примеры (взвешенно по iteration)
	samples := make([]*Sample, 0, batchSize)
	for i := 0; i < batchSize; i++ {
		idx := m.weightedSampleIndex(playerID)
		samples = append(samples, m.samples[playerID][idx])
	}

	return samples
}

// weightedSampleIndex выбирает индекс с учетом весов примеров
func (m *MemoryBuffer) weightedSampleIndex(playerID int) int {
	totalWeight := float32(0)
	for _, s := range m.samples[playerID] {
		totalWeight += s.Weight * float32(s.Iteration)
	}

	target := m.rng.Float32() * totalWeight
	sum := float32(0)
	for i, s := range m.samples[playerID] {
		sum += s.Weight * float32(s.Iteration)
		if sum >= target {
			return i
		}
	}
	return len(m.samples[playerID]) - 1
}

// pruneOldSamples удаляет старые примеры
func (m *MemoryBuffer) pruneOldSamples(playerID int) {
	samples := m.samples[playerID]

	// Сортируем по iteration (старые в начале)
	sort.Slice(samples, func(i, j int) bool {
		return samples[i].Iteration < samples[j].Iteration
	})

	// Удаляем часть старых примеров
	removeCount := int(float32(len(samples)) * m.pruneRatio)
	m.samples[playerID] = samples[removeCount:]
}
