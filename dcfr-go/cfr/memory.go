package cfr

import (
	"dcfr-go/common/linq"
	"dcfr-go/nolimitholdem"
	"math/rand"
	"sync"
	"time"
)

// Sample хранит данные одного состояния для обучения
type Sample struct {
	State     *nolimitholdem.GameState         // Состояние игры
	Strategy  nolimitholdem.Strategy           // Стратегия
	Regrets   map[nolimitholdem.Action]float32 // Сожаления для каждого действия
	ReachProb float32                          // Вес примера (reach probability)
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

func (m *MemoryBuffer) Count(playerID int) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.samples[playerID])
}

// AddSample добавляет новый пример для игрока
func (m *MemoryBuffer) AddSample(
	playerID int,
	state *nolimitholdem.GameState,
	strategy nolimitholdem.Strategy,
	regrets map[nolimitholdem.Action]float32,
	reachProb float32,
	iteration int,
) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Создаем новый пример
	sample := &Sample{
		State:     state.Clone(),
		Strategy:  strategy,
		Regrets:   linq.CopyMap(regrets),
		ReachProb: reachProb,
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

	if batchSize > len(m.samples[playerID]) {
		batchSize = len(m.samples[playerID])
	}

	// Выбираем случайные примеры
	samples := make([]*Sample, 0, batchSize)
	for i := 0; i < batchSize; i++ {
		idx := m.rng.Int31n(int32(len(m.samples[playerID])))
		samples = append(samples, m.samples[playerID][idx])
	}
	return samples
}

// pruneOldSamples удаляет старые примеры
func (m *MemoryBuffer) pruneOldSamples(playerID int) {
	samples := m.samples[playerID]

	// Удаляем часть старых примеров
	removeCount := int(float32(len(samples)) * m.pruneRatio)
	m.samples[playerID] = samples[removeCount:]
}
