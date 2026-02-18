package cfr

import (
	"dcfr-go/common/linq"
	"dcfr-go/nolimitholdem"
	"encoding/json"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

type IMemoryBuffer interface {
	AddStrategySample(state *nolimitholdem.GameState, strategy nolimitholdem.Strategy, iteration int)
	AddSample(
		playerID int,
		gameID uuid.UUID,
		state *nolimitholdem.GameState,
		regrets map[nolimitholdem.Action]float32,
		iteration int,
		lstmH []float32,
		lstmC []float32,
	)
}

type ActorState struct {
	LstmH []float32
	LstmC []float32
}

func (h *ActorState) Clone() *ActorState {
	a := &ActorState{
		LstmH: make([]float32, len(h.LstmH)),
		LstmC: make([]float32, len(h.LstmC)),
	}
	copy(a.LstmC, h.LstmC)
	copy(a.LstmH, h.LstmH)
	return a
}

type GameSample struct {
	States []*StateSample
}

// GameSample хранит данные одного состояния для обучения
type StateSample struct {
	GameState  *nolimitholdem.GameState         // Состояние игры
	ActorState *ActorState                      // Внутреннее состояние игрока
	Regrets    map[nolimitholdem.Action]float32 // Сожаления для каждого действия
	Iteration  int                              // Итерация, на которой был собран пример
}

type Game struct {
	CreatedAt time.Time
	Samples   []*StateSample
}

// MemoryBuffer кэширует данные для обучения нейросетей
type MemoryBuffer struct {
	samp_mu    sync.RWMutex
	samples    map[int]map[uuid.UUID]*Game // samples[playerID] -> []*Sample
	maxSamples int                         // Максимальное число примеров на игрока
	pruneRatio float32                     // Доля старых примеров для удаления
	rng        *rand.Rand                  // Генератор случайных чисел

}

// NewMemoryBuffer создает буфер с настройками
func NewMemoryBuffer(maxSamples int, pruneRatio float32) (*MemoryBuffer, error) {
	m := &MemoryBuffer{
		samples:    make(map[int]map[uuid.UUID]*Game),
		maxSamples: maxSamples,
		pruneRatio: pruneRatio,
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	return m, nil
}

func (m *MemoryBuffer) Save() error {
	f, err := os.Create("bufferdata.json")
	if err != nil {
		return err
	}
	return json.NewEncoder(f).Encode(m.samples)
}
func (m *MemoryBuffer) Load() error {
	f, err := os.Open("bufferdata.json")
	if err != nil {
		return err
	}
	return json.NewDecoder(f).Decode(&m.samples)
}

func (m *MemoryBuffer) Count(playerID int) int {
	m.samp_mu.Lock()
	defer m.samp_mu.Unlock()
	return len(m.samples[playerID])
}

// AddSample добавляет новый пример для игрока и раздачи
func (m *MemoryBuffer) AddSample(
	playerID int,
	gameID uuid.UUID,
	state *CFRState,
	regrets map[nolimitholdem.Action]float32,
	iteration int,
) {
	m.samp_mu.Lock()
	defer m.samp_mu.Unlock()

	// Создаем новый пример
	sample := &StateSample{
		GameState:  state.GameState.Clone(),
		ActorState: state.ActorState.Clone(),
		Regrets:    linq.CopyMap(regrets),
		Iteration:  iteration,
	}
	// Получаем баккит игрока
	plyBucket, ex := m.samples[playerID]
	if !ex {
		plyBucket = make(map[uuid.UUID]*Game)
		m.samples[playerID] = plyBucket
	}
	// Получаем баккит игры
	game, ex := plyBucket[gameID]
	if !ex {
		game = &Game{
			Samples:   make([]*StateSample, 0),
			CreatedAt: time.Now(),
		}
		plyBucket[gameID] = game
	}

	// Добавляем в хранилище
	game.Samples = append(game.Samples, sample)

	// Проверяем необходимость очистки
	if len(m.samples[playerID]) > m.maxSamples {
		m.pruneOldSamples(playerID)
	}
}

// GetSamples возвращает батч примеров для обучения
func (m *MemoryBuffer) GetSamples(playerID int, batchSize int) []*GameSample {
	m.samp_mu.RLock()
	defer m.samp_mu.RUnlock()

	playerBucket := m.samples[playerID]
	samples := make([]*GameSample, 0, batchSize)

	keys := make([]uuid.UUID, 0, len(playerBucket))
	for u := range playerBucket {
		keys = append(keys, u)
	}

	collected := int(0)
	for {
		idx := m.rng.Int31n(int32(len(keys)))
		key := keys[idx]

		sample := playerBucket[key]
		samples = append(samples, &GameSample{
			States: sample.Samples,
		})
		collected += len(sample.Samples)
		if collected >= batchSize {
			break
		}
	}

	return samples
}

// pruneOldSamples удаляет старые примеры
func (m *MemoryBuffer) pruneOldSamples(playerID int) {
	playerBucket := m.samples[playerID]

	type gameEntry struct {
		id        uuid.UUID
		createdAt time.Time
	}

	entries := make([]gameEntry, 0, len(playerBucket))
	for id, game := range playerBucket {
		entries = append(entries, gameEntry{id, game.CreatedAt})
	}

	// Сортируем по времени создания (от старых к новым)
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].createdAt.Before(entries[j].createdAt)
	})

	// Удаляем часть старых примеров
	removeCount := int(float32(len(entries)) * m.pruneRatio)
	for i := 0; i < removeCount; i++ {
		delete(playerBucket, entries[i].id)
	}
}
