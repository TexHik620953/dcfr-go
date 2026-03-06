package cfr

import (
	"dcfr-go/common/linq"
	"dcfr-go/nolimitholdem"
	"encoding/json"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/google/uuid"
)

type ActorState struct {
	// LstmH is reused to store transformer history context (flattened sequence features)
	LstmH []float32
}

func (h *ActorState) Clone() *ActorState {
	a := &ActorState{}
	if h.LstmH != nil {
		a.LstmH = make([]float32, len(h.LstmH))
		copy(a.LstmH, h.LstmH)
	}
	return a
}

type GameSample struct {
	States []*StateSample
}

type StateSample struct {
	GameState  *nolimitholdem.GameState
	ActorState *ActorState
	Regrets    map[nolimitholdem.Action]float32
	Iteration  int
}

// StrategyGameSample holds strategy samples for one game (for average strategy network)
type StrategyGameSample struct {
	States []*StrategySample
}

type StrategySample struct {
	GameState  *nolimitholdem.GameState
	ActorState *ActorState
	Strategy   nolimitholdem.Strategy
	Iteration  int
}

// MemoryBuffer uses reservoir sampling to maintain a fixed-size buffer per player.
// Games are accumulated in a pending map during traversal, then flushed to the
// reservoir once complete.
type MemoryBuffer struct {
	mu         sync.Mutex
	reservoir  map[int][]*GameSample // reservoir[playerID] -> fixed-size slice
	pending    map[int]map[uuid.UUID]*GameSample
	maxSamples int
	totalSeen  map[int]int64
	rng        *rand.Rand
}

func NewMemoryBuffer(maxSamples int) (*MemoryBuffer, error) {
	m := &MemoryBuffer{
		reservoir:  make(map[int][]*GameSample),
		pending:    make(map[int]map[uuid.UUID]*GameSample),
		maxSamples: maxSamples,
		totalSeen:  make(map[int]int64),
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	return m, nil
}

func (m *MemoryBuffer) Save() error {
	f, err := os.Create("bufferdata.json")
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(m.reservoir)
}

func (m *MemoryBuffer) Load() error {
	f, err := os.Open("bufferdata.json")
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewDecoder(f).Decode(&m.reservoir)
}

func (m *MemoryBuffer) Count(playerID int) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.reservoir[playerID])
}

// AddSample adds a state sample to a pending game. Call FlushGame when the
// traversal for this game is complete.
func (m *MemoryBuffer) AddSample(
	playerID int,
	gameID uuid.UUID,
	state *CFRState,
	regrets map[nolimitholdem.Action]float32,
	iteration int,
) {
	m.mu.Lock()
	defer m.mu.Unlock()

	sample := &StateSample{
		GameState:  state.GameState.Clone(),
		ActorState: state.ActorState.Clone(),
		Regrets:    linq.CopyMap(regrets),
		Iteration:  iteration,
	}

	pending, ex := m.pending[playerID]
	if !ex {
		pending = make(map[uuid.UUID]*GameSample)
		m.pending[playerID] = pending
	}
	game, ex := pending[gameID]
	if !ex {
		game = &GameSample{States: make([]*StateSample, 0, 8)}
		pending[gameID] = game
	}
	game.States = append(game.States, sample)
}

// FlushGame moves a completed game from pending into the reservoir using
// reservoir sampling (Algorithm R).
func (m *MemoryBuffer) FlushGame(playerID int, gameID uuid.UUID) {
	m.mu.Lock()
	defer m.mu.Unlock()

	pending := m.pending[playerID]
	if pending == nil {
		return
	}
	game, ex := pending[gameID]
	if !ex {
		return
	}
	delete(pending, gameID)

	buf := m.reservoir[playerID]
	n := m.totalSeen[playerID] + 1
	m.totalSeen[playerID] = n

	if len(buf) < m.maxSamples {
		m.reservoir[playerID] = append(buf, game)
	} else {
		// Reservoir sampling: replace element j with probability maxSamples/n
		j := m.rng.Int63n(n)
		if j < int64(m.maxSamples) {
			buf[j] = game
		}
	}
}

// GetSamples returns a random batch of games for training
func (m *MemoryBuffer) GetSamples(playerID int, batchSize int) []*GameSample {
	m.mu.Lock()
	defer m.mu.Unlock()

	buf := m.reservoir[playerID]
	if len(buf) == 0 {
		return nil
	}

	samples := make([]*GameSample, 0, batchSize)
	collected := 0
	for collected < batchSize {
		idx := m.rng.Intn(len(buf))
		samples = append(samples, buf[idx])
		collected += len(buf[idx].States)
	}
	return samples
}

// StrategyMemoryBuffer stores strategy samples for average strategy network training.
// Uses the same reservoir sampling approach as MemoryBuffer.
type StrategyMemoryBuffer struct {
	mu         sync.Mutex
	reservoir  map[int][]*StrategyGameSample
	pending    map[int]map[uuid.UUID]*StrategyGameSample
	maxSamples int
	totalSeen  map[int]int64
	rng        *rand.Rand
}

func NewStrategyMemoryBuffer(maxSamples int) *StrategyMemoryBuffer {
	return &StrategyMemoryBuffer{
		reservoir:  make(map[int][]*StrategyGameSample),
		pending:    make(map[int]map[uuid.UUID]*StrategyGameSample),
		maxSamples: maxSamples,
		totalSeen:  make(map[int]int64),
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (m *StrategyMemoryBuffer) Count(playerID int) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.reservoir[playerID])
}

func (m *StrategyMemoryBuffer) AddSample(
	playerID int,
	gameID uuid.UUID,
	state *CFRState,
	strategy nolimitholdem.Strategy,
	iteration int,
) {
	m.mu.Lock()
	defer m.mu.Unlock()

	sample := &StrategySample{
		GameState:  state.GameState.Clone(),
		ActorState: state.ActorState.Clone(),
		Strategy:   linq.CopyMap(strategy),
		Iteration:  iteration,
	}

	pending, ex := m.pending[playerID]
	if !ex {
		pending = make(map[uuid.UUID]*StrategyGameSample)
		m.pending[playerID] = pending
	}
	game, ex := pending[gameID]
	if !ex {
		game = &StrategyGameSample{States: make([]*StrategySample, 0, 8)}
		pending[gameID] = game
	}
	game.States = append(game.States, sample)
}

func (m *StrategyMemoryBuffer) FlushGame(playerID int, gameID uuid.UUID) {
	m.mu.Lock()
	defer m.mu.Unlock()

	pending := m.pending[playerID]
	if pending == nil {
		return
	}
	game, ex := pending[gameID]
	if !ex {
		return
	}
	delete(pending, gameID)

	buf := m.reservoir[playerID]
	n := m.totalSeen[playerID] + 1
	m.totalSeen[playerID] = n

	if len(buf) < m.maxSamples {
		m.reservoir[playerID] = append(buf, game)
	} else {
		j := m.rng.Int63n(n)
		if j < int64(m.maxSamples) {
			buf[j] = game
		}
	}
}

func (m *StrategyMemoryBuffer) GetSamples(playerID int, batchSize int) []*StrategyGameSample {
	m.mu.Lock()
	defer m.mu.Unlock()

	buf := m.reservoir[playerID]
	if len(buf) == 0 {
		return nil
	}

	samples := make([]*StrategyGameSample, 0, batchSize)
	collected := 0
	for collected < batchSize {
		idx := m.rng.Intn(len(buf))
		samples = append(samples, buf[idx])
		collected += len(buf[idx].States)
	}
	return samples
}
