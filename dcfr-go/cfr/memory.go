package cfr

import (
	"database/sql/driver"
	"dcfr-go/common/linq"
	"dcfr-go/nolimitholdem"
	"encoding/json"
	"errors"
	"log"
	"math/rand"
	"os"
	"sync"
	"time"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

type IMemoryBuffer interface {
	AddStrategySample(state *nolimitholdem.GameState, strategy nolimitholdem.Strategy, iteration int)
	AddSample(
		playerID int,
		state *nolimitholdem.GameState,
		regrets map[nolimitholdem.Action]float32,
		iteration int,
	)
}

const STRATEGY_BATCH = 5000
const WRITER_THREADS = 100

type StrategySample struct {
	ID        uint          `gorm:"primaryKey"`
	State     GameStateJSON `gorm:"type:jsonb"`
	Strategy  StrategyJSON  `gorm:"type:jsonb"`
	Iteration int
}

// GameStateJSON - обёртка для сериализации GameState в JSON
type GameStateJSON struct {
	nolimitholdem.GameState
}

// StrategyJSON - обёртка для сериализации Strategy в JSON
type StrategyJSON struct {
	nolimitholdem.Strategy
}

// Value преобразует GameState в JSON для сохранения в БД
func (gs *GameStateJSON) Value() (driver.Value, error) {
	return json.Marshal(gs.GameState)
}

// Scan загружает GameState из JSON в поле БД
func (gs *GameStateJSON) Scan(value interface{}) error {
	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("неверный тип данных для GameState")
	}

	return json.Unmarshal(bytes, &gs.GameState)
}

// Value преобразует Strategy в JSON для сохранения в БД
func (s *StrategyJSON) Value() (driver.Value, error) {
	if s.Strategy == nil {
		return nil, nil
	}
	return json.Marshal(s.Strategy)
}

// Scan загружает Strategy из JSON в поле БД
func (s *StrategyJSON) Scan(value interface{}) error {
	if value == nil {
		s.Strategy = nil
		return nil
	}

	bytes, ok := value.([]byte)
	if !ok {
		return errors.New("неверный тип данных для Strategy")
	}

	return json.Unmarshal(bytes, &s.Strategy)
}

// Sample хранит данные одного состояния для обучения
type Sample struct {
	State *nolimitholdem.GameState // Состояние игры
	//Strategy  nolimitholdem.Strategy           // Стратегия
	Regrets   map[nolimitholdem.Action]float32 // Сожаления для каждого действия
	ReachProb float32                          // Вес примера (reach probability)
	Iteration int                              // Итерация, на которой был собран пример
}

// MemoryBuffer кэширует данные для обучения нейросетей
type MemoryBuffer struct {
	samp_mu    sync.RWMutex
	samples    map[int][]*Sample // samples[playerID] -> []*Sample
	maxSamples int               // Максимальное число примеров на игрока
	pruneRatio float32           // Доля старых примеров для удаления
	rng        *rand.Rand        // Генератор случайных чисел

	strat_samples chan *StrategySample

	db *gorm.DB
}

// NewMemoryBuffer создает буфер с настройками
func NewMemoryBuffer(maxSamples int, pruneRatio float32, dsn string) (*MemoryBuffer, error) {
	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
		Logger: logger.New(
			log.New(os.Stdout, "\r\n", log.LstdFlags), // io writer
			logger.Config{
				SlowThreshold:             time.Second,  // Slow SQL threshold
				LogLevel:                  logger.Error, // Log level
				IgnoreRecordNotFoundError: true,         // Ignore ErrRecordNotFound error for logger
				ParameterizedQueries:      true,         // Don't include params in the SQL log
				Colorful:                  true,         // Disable color
			},
		),
		SkipDefaultTransaction: true, // Улучшает производительность
		PrepareStmt:            true, // Подготавливает выражения для повторного использования
	})
	if err != nil {
		return nil, err
	}

	err = db.AutoMigrate(&StrategySample{})
	if err != nil {
		return nil, err
	}

	m := &MemoryBuffer{
		samples:       make(map[int][]*Sample),
		maxSamples:    maxSamples,
		pruneRatio:    pruneRatio,
		rng:           rand.New(rand.NewSource(time.Now().UnixNano())),
		strat_samples: make(chan *StrategySample, STRATEGY_BATCH),
		db:            db,
	}

	for range WRITER_THREADS {
		go m.dbWriter()
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

func (m *MemoryBuffer) dbWriter() {
	buf := make([]*StrategySample, 0, STRATEGY_BATCH)
	for data := range m.strat_samples {
		buf = append(buf, data)
		if len(buf) == STRATEGY_BATCH {
			err := m.db.Create(buf).Error
			if err != nil {
				log.Panicf("Failed to insert strat samples: %v", err)
			}
			buf = buf[:0]
		}
	}
}

func (m *MemoryBuffer) Count(playerID int) int {
	m.samp_mu.Lock()
	defer m.samp_mu.Unlock()
	return len(m.samples[playerID])
}

// AddSample добавляет новый пример для игрока
func (m *MemoryBuffer) AddSample(
	playerID int,
	state *nolimitholdem.GameState,
	regrets map[nolimitholdem.Action]float32,
	reachProb float32,
	iteration int,
) {
	m.samp_mu.Lock()
	defer m.samp_mu.Unlock()

	// Создаем новый пример
	sample := &Sample{
		State: state.Clone(),
		//Strategy:  strategy,
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

func (m *MemoryBuffer) AddStrategySample(
	state *nolimitholdem.GameState,
	strategy nolimitholdem.Strategy,
	iteration int,
) {
	m.strat_samples <- &StrategySample{
		State:     GameStateJSON{GameState: *state.Clone()},
		Strategy:  StrategyJSON{Strategy: linq.CopyMap(strategy)},
		Iteration: iteration,
	}
}

// GetSamples возвращает батч примеров для обучения
func (m *MemoryBuffer) GetSamples(playerID int, batchSize int) []*Sample {
	m.samp_mu.RLock()
	defer m.samp_mu.RUnlock()

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
