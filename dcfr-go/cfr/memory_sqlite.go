package cfr

import (
	"bytes"
	"database/sql"
	"dcfr-go/nolimitholdem"
	"encoding/gob"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	_ "modernc.org/sqlite"
)

func init() {
	gob.Register(map[nolimitholdem.Action]float32{})
	gob.Register(nolimitholdem.Strategy{})
}

func encodeGameSample(g *GameSample) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(g); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeGameSample(data []byte) (*GameSample, error) {
	var g GameSample
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&g); err != nil {
		return nil, err
	}
	return &g, nil
}

func encodeStrategyGameSample(g *StrategyGameSample) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(g); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeStrategyGameSample(data []byte) (*StrategyGameSample, error) {
	var g StrategyGameSample
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&g); err != nil {
		return nil, err
	}
	return &g, nil
}

// ===== Regret Memory Buffer (SQLite) =====

type regretFlushRequest struct {
	playerID  int
	data      []byte
	iteration int
}

type SQLiteMemoryBuffer struct {
	db         *sql.DB
	pending    sync.Map // map[pendingKey]*GameSample — lock-free
	maxSamples int
	countAtom  [3]atomic.Int32

	flushCh chan regretFlushRequest
	done    chan struct{}
}

type pendingKey struct {
	playerID int
	gameID   uuid.UUID
}

func NewSQLiteMemoryBuffer(dbPath string, maxSamples int) (*SQLiteMemoryBuffer, error) {
	db, err := sql.Open("sqlite", dbPath+"?_journal_mode=WAL&_synchronous=NORMAL")
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS regret_samples (
			slot INTEGER NOT NULL,
			player_id INTEGER NOT NULL,
			iteration INTEGER NOT NULL,
			data BLOB NOT NULL,
			PRIMARY KEY (player_id, slot)
		)
	`)
	if err != nil {
		return nil, fmt.Errorf("create table: %w", err)
	}

	var counts [3]int
	rows, err := db.Query("SELECT player_id, COUNT(*) FROM regret_samples GROUP BY player_id")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var pid, cnt int
		rows.Scan(&pid, &cnt)
		if pid >= 0 && pid < 3 {
			counts[pid] = cnt
		}
	}

	var totalSeen [3]int64
	rows2, err := db.Query("SELECT player_id, MAX(slot)+1 FROM regret_samples GROUP BY player_id")
	if err != nil {
		return nil, err
	}
	defer rows2.Close()
	for rows2.Next() {
		var pid int
		var maxSlot int64
		rows2.Scan(&pid, &maxSlot)
		if pid >= 0 && pid < 3 {
			if maxSlot > int64(maxSamples) {
				totalSeen[pid] = maxSlot
			} else {
				totalSeen[pid] = int64(counts[pid])
			}
		}
	}

	m := &SQLiteMemoryBuffer{
		db:         db,
		maxSamples: maxSamples,
		flushCh:    make(chan regretFlushRequest, 100_000),
		done:       make(chan struct{}),
	}
	for pid := 0; pid < 3; pid++ {
		m.countAtom[pid].Store(int32(counts[pid]))
	}

	go m.writer(counts, totalSeen)
	return m, nil
}

func (m *SQLiteMemoryBuffer) writer(counts [3]int, totalSeen [3]int64) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	batch := make([]regretFlushRequest, 0, 2000)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	writeBatch := func() {
		if len(batch) == 0 {
			return
		}
		tx, err := m.db.Begin()
		if err != nil {
			batch = batch[:0]
			return
		}
		for _, req := range batch {
			pid := req.playerID
			n := totalSeen[pid] + 1
			totalSeen[pid] = n

			if counts[pid] < m.maxSamples {
				slot := counts[pid]
				counts[pid]++
				m.countAtom[pid].Store(int32(counts[pid]))
				tx.Exec(
					"INSERT INTO regret_samples (slot, player_id, iteration, data) VALUES (?, ?, ?, ?)",
					slot, pid, req.iteration, req.data,
				)
			} else {
				j := rng.Int63n(n)
				if j < int64(m.maxSamples) {
					tx.Exec(
						"UPDATE regret_samples SET iteration=?, data=? WHERE player_id=? AND slot=?",
						req.iteration, req.data, pid, int(j),
					)
				}
			}
		}
		tx.Commit()
		batch = batch[:0]
	}

	for {
		select {
		case req, ok := <-m.flushCh:
			if !ok {
				writeBatch()
				close(m.done)
				return
			}
			batch = append(batch, req)
			if len(batch) >= 2000 {
				writeBatch()
			}
		case <-ticker.C:
			writeBatch()
		}
	}
}

func (m *SQLiteMemoryBuffer) Close() error {
	close(m.flushCh)
	<-m.done
	return m.db.Close()
}

func (m *SQLiteMemoryBuffer) Count(playerID int) int {
	if playerID >= 0 && playerID < 3 {
		return int(m.countAtom[playerID].Load())
	}
	return 0
}

// AddSample is lock-free — each goroutine writes to its own gameID key.
func (m *SQLiteMemoryBuffer) AddSample(
	playerID int,
	gameID uuid.UUID,
	state *CFRState,
	regrets map[nolimitholdem.Action]float32,
	iteration int,
) {
	sample := &StateSample{
		GameState:  state.GameState.Clone(),
		ActorState: state.ActorState.Clone(),
		Regrets:    copyActionMap(regrets),
		Iteration:  iteration,
	}

	key := pendingKey{playerID, gameID}
	val, _ := m.pending.LoadOrStore(key, &GameSample{States: make([]*StateSample, 0, 8)})
	game := val.(*GameSample)
	game.States = append(game.States, sample)
}

// FlushGame encodes the game and sends it to the writer goroutine (non-blocking).
func (m *SQLiteMemoryBuffer) FlushGame(playerID int, gameID uuid.UUID) {
	key := pendingKey{playerID, gameID}
	val, ok := m.pending.LoadAndDelete(key)
	if !ok {
		return
	}
	game := val.(*GameSample)

	data, err := encodeGameSample(game)
	if err != nil {
		return
	}

	iteration := 0
	if len(game.States) > 0 {
		iteration = game.States[0].Iteration
	}

	m.flushCh <- regretFlushRequest{
		playerID:  playerID,
		data:      data,
		iteration: iteration,
	}
}

func (m *SQLiteMemoryBuffer) GetSamples(playerID int, batchSize int) []*GameSample {
	count := m.Count(playerID)
	if count == 0 {
		return nil
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	slots := make([]int, 0, batchSize)
	collected := 0
	for collected < batchSize {
		slots = append(slots, rng.Intn(count))
		collected += 5
	}

	samples := make([]*GameSample, 0, len(slots))
	for _, slot := range slots {
		var data []byte
		err := m.db.QueryRow(
			"SELECT data FROM regret_samples WHERE player_id=? AND slot=?",
			playerID, slot,
		).Scan(&data)
		if err != nil {
			continue
		}
		game, err := decodeGameSample(data)
		if err != nil {
			continue
		}
		samples = append(samples, game)
	}
	return samples
}

func (m *SQLiteMemoryBuffer) Save() error { return nil }
func (m *SQLiteMemoryBuffer) Load() error { return nil }

// ===== Strategy Memory Buffer (SQLite) =====

type strategyFlushRequest struct {
	playerID  int
	data      []byte
	iteration int
}

type SQLiteStrategyMemoryBuffer struct {
	db         *sql.DB
	pending    sync.Map // map[pendingKey]*StrategyGameSample
	maxSamples int
	countAtom  [3]atomic.Int32

	flushCh chan strategyFlushRequest
	done    chan struct{}
}

func NewSQLiteStrategyMemoryBuffer(dbPath string, maxSamples int) (*SQLiteStrategyMemoryBuffer, error) {
	db, err := sql.Open("sqlite", dbPath+"?_journal_mode=WAL&_synchronous=NORMAL")
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS strategy_samples (
			slot INTEGER NOT NULL,
			player_id INTEGER NOT NULL,
			iteration INTEGER NOT NULL,
			data BLOB NOT NULL,
			PRIMARY KEY (player_id, slot)
		)
	`)
	if err != nil {
		return nil, fmt.Errorf("create table: %w", err)
	}

	var counts [3]int
	rows, err := db.Query("SELECT player_id, COUNT(*) FROM strategy_samples GROUP BY player_id")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var pid, cnt int
		rows.Scan(&pid, &cnt)
		if pid >= 0 && pid < 3 {
			counts[pid] = cnt
		}
	}

	var totalSeen [3]int64
	for pid := 0; pid < 3; pid++ {
		totalSeen[pid] = int64(counts[pid])
	}

	m := &SQLiteStrategyMemoryBuffer{
		db:         db,
		maxSamples: maxSamples,
		flushCh:    make(chan strategyFlushRequest, 100_000),
		done:       make(chan struct{}),
	}
	for pid := 0; pid < 3; pid++ {
		m.countAtom[pid].Store(int32(counts[pid]))
	}

	go m.writer(counts, totalSeen)
	return m, nil
}

func (m *SQLiteStrategyMemoryBuffer) writer(counts [3]int, totalSeen [3]int64) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	batch := make([]strategyFlushRequest, 0, 2000)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	writeBatch := func() {
		if len(batch) == 0 {
			return
		}
		tx, err := m.db.Begin()
		if err != nil {
			batch = batch[:0]
			return
		}
		for _, req := range batch {
			pid := req.playerID
			n := totalSeen[pid] + 1
			totalSeen[pid] = n

			if counts[pid] < m.maxSamples {
				slot := counts[pid]
				counts[pid]++
				m.countAtom[pid].Store(int32(counts[pid]))
				tx.Exec(
					"INSERT INTO strategy_samples (slot, player_id, iteration, data) VALUES (?, ?, ?, ?)",
					slot, pid, req.iteration, req.data,
				)
			} else {
				j := rng.Int63n(n)
				if j < int64(m.maxSamples) {
					tx.Exec(
						"UPDATE strategy_samples SET iteration=?, data=? WHERE player_id=? AND slot=?",
						req.iteration, req.data, pid, int(j),
					)
				}
			}
		}
		tx.Commit()
		batch = batch[:0]
	}

	for {
		select {
		case req, ok := <-m.flushCh:
			if !ok {
				writeBatch()
				close(m.done)
				return
			}
			batch = append(batch, req)
			if len(batch) >= 2000 {
				writeBatch()
			}
		case <-ticker.C:
			writeBatch()
		}
	}
}

func (m *SQLiteStrategyMemoryBuffer) Close() error {
	close(m.flushCh)
	<-m.done
	return m.db.Close()
}

func (m *SQLiteStrategyMemoryBuffer) Count(playerID int) int {
	if playerID >= 0 && playerID < 3 {
		return int(m.countAtom[playerID].Load())
	}
	return 0
}

// AddSample is lock-free.
func (m *SQLiteStrategyMemoryBuffer) AddSample(
	playerID int,
	gameID uuid.UUID,
	state *CFRState,
	strategy nolimitholdem.Strategy,
	iteration int,
) {
	sample := &StrategySample{
		GameState:  state.GameState.Clone(),
		ActorState: state.ActorState.Clone(),
		Strategy:   copyStrategyMap(strategy),
		Iteration:  iteration,
	}

	key := pendingKey{playerID, gameID}
	val, _ := m.pending.LoadOrStore(key, &StrategyGameSample{States: make([]*StrategySample, 0, 8)})
	game := val.(*StrategyGameSample)
	game.States = append(game.States, sample)
}

// FlushGame encodes and sends to writer goroutine (non-blocking).
func (m *SQLiteStrategyMemoryBuffer) FlushGame(playerID int, gameID uuid.UUID) {
	key := pendingKey{playerID, gameID}
	val, ok := m.pending.LoadAndDelete(key)
	if !ok {
		return
	}
	game := val.(*StrategyGameSample)

	data, err := encodeStrategyGameSample(game)
	if err != nil {
		return
	}

	iteration := 0
	if len(game.States) > 0 {
		iteration = game.States[0].Iteration
	}

	m.flushCh <- strategyFlushRequest{
		playerID:  playerID,
		data:      data,
		iteration: iteration,
	}
}

func (m *SQLiteStrategyMemoryBuffer) GetSamples(playerID int, batchSize int) []*StrategyGameSample {
	count := m.Count(playerID)
	if count == 0 {
		return nil
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	slots := make([]int, 0, batchSize)
	collected := 0
	for collected < batchSize {
		slots = append(slots, rng.Intn(count))
		collected += 5
	}

	samples := make([]*StrategyGameSample, 0, len(slots))
	for _, slot := range slots {
		var data []byte
		err := m.db.QueryRow(
			"SELECT data FROM strategy_samples WHERE player_id=? AND slot=?",
			playerID, slot,
		).Scan(&data)
		if err != nil {
			continue
		}
		game, err := decodeStrategyGameSample(data)
		if err != nil {
			continue
		}
		samples = append(samples, game)
	}
	return samples
}

func copyActionMap(m map[nolimitholdem.Action]float32) map[nolimitholdem.Action]float32 {
	cp := make(map[nolimitholdem.Action]float32, len(m))
	for k, v := range m {
		cp[k] = v
	}
	return cp
}

func copyStrategyMap(m nolimitholdem.Strategy) nolimitholdem.Strategy {
	cp := make(nolimitholdem.Strategy, len(m))
	for k, v := range m {
		cp[k] = v
	}
	return cp
}
