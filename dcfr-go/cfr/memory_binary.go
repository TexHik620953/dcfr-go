package cfr

import (
	"dcfr-go/nolimitholdem"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

const (
	headerSize    = 64
	fixedPartSize = 156 // bytes before context_h
	magicStr      = "DCFR"
)

// Binary record layout (little-endian, all fields contiguous):
//
//	[0:12]     active_players_mask  3×int32
//	[12:24]    players_pots         3×int32
//	[24:36]    stakes               3×int32
//	[36:76]    legal_actions        10×float32 (0.0 or 1.0)
//	[76:80]    stage                int32
//	[80:84]    current_player       int32
//	[84:104]   public_cards         5×int32 (-1 = absent)
//	[104:112]  private_cards        2×int32
//	[112:152]  regrets              10×float32
//	[152:156]  iteration            int32
//	[156:...]  context_h            hidden_dim×float32

type sampleFlushRequest struct {
	playerID int
	data     []byte
	syncCh   chan struct{} // non-nil = drain signal
}

type BinaryMemoryBuffer struct {
	file       *os.File
	filePath   string
	maxSamples int
	hiddenDim  int
	recordSize int
	countAtom  [3]atomic.Int32

	flushCh chan sampleFlushRequest
	done    chan struct{}
}

func NewBinaryMemoryBuffer(filePath string, maxSamples, hiddenDim int) (*BinaryMemoryBuffer, error) {
	recordSize := fixedPartSize + hiddenDim*4

	var counts [3]int
	var totalSeen [3]int64

	fileExists := false
	if info, err := os.Stat(filePath); err == nil && info.Size() >= headerSize {
		fileExists = true
	}

	f, err := os.OpenFile(filePath, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return nil, fmt.Errorf("open binary buffer: %w", err)
	}

	if fileExists {
		hdr := make([]byte, headerSize)
		if _, err := f.ReadAt(hdr, 0); err != nil {
			return nil, fmt.Errorf("read header: %w", err)
		}
		if string(hdr[0:4]) != magicStr {
			return nil, fmt.Errorf("invalid magic bytes")
		}
		storedMax := int(binary.LittleEndian.Uint32(hdr[8:12]))
		storedDim := int(binary.LittleEndian.Uint32(hdr[12:16]))
		if storedMax != maxSamples || storedDim != hiddenDim {
			return nil, fmt.Errorf("config mismatch: file max=%d dim=%d, want max=%d dim=%d",
				storedMax, storedDim, maxSamples, hiddenDim)
		}
		for i := 0; i < 3; i++ {
			counts[i] = int(binary.LittleEndian.Uint32(hdr[20+i*4 : 24+i*4]))
			totalSeen[i] = int64(binary.LittleEndian.Uint64(hdr[32+i*8 : 40+i*8]))
		}
	} else {
		totalSize := int64(headerSize) + int64(3)*int64(maxSamples)*int64(recordSize)
		if err := f.Truncate(totalSize); err != nil {
			return nil, fmt.Errorf("truncate file to %d bytes: %w", totalSize, err)
		}
		hdr := make([]byte, headerSize)
		copy(hdr[0:4], magicStr)
		binary.LittleEndian.PutUint32(hdr[4:8], 1) // version
		binary.LittleEndian.PutUint32(hdr[8:12], uint32(maxSamples))
		binary.LittleEndian.PutUint32(hdr[12:16], uint32(hiddenDim))
		binary.LittleEndian.PutUint32(hdr[16:20], 3) // num_players
		if _, err := f.WriteAt(hdr, 0); err != nil {
			return nil, fmt.Errorf("write header: %w", err)
		}
	}

	m := &BinaryMemoryBuffer{
		file:       f,
		filePath:   filePath,
		maxSamples: maxSamples,
		hiddenDim:  hiddenDim,
		recordSize: recordSize,
		flushCh:    make(chan sampleFlushRequest, 200_000),
		done:       make(chan struct{}),
	}
	for i := 0; i < 3; i++ {
		m.countAtom[i].Store(int32(counts[i]))
	}

	go m.writer(counts, totalSeen)
	return m, nil
}

func (m *BinaryMemoryBuffer) recordOffset(playerID, slot int) int64 {
	return int64(headerSize) +
		int64(playerID)*int64(m.maxSamples)*int64(m.recordSize) +
		int64(slot)*int64(m.recordSize)
}

func putInt32(buf []byte, off int, v int32) int {
	binary.LittleEndian.PutUint32(buf[off:], uint32(v))
	return off + 4
}

func putFloat32(buf []byte, off int, v float32) int {
	binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(v))
	return off + 4
}

func (m *BinaryMemoryBuffer) encodeRecord(
	gs *nolimitholdem.GameState,
	actorState *ActorState,
	regrets map[nolimitholdem.Action]float32,
	iteration int,
) []byte {
	buf := make([]byte, m.recordSize)
	off := 0

	// active_players_mask: 3×int32
	for i := 0; i < 3; i++ {
		v := int32(0)
		if i < len(gs.ActivePlayersMask) {
			v = gs.ActivePlayersMask[i]
		}
		off = putInt32(buf, off, v)
	}

	// players_pots: 3×int32
	for i := 0; i < 3; i++ {
		v := int32(0)
		if i < len(gs.PlayersPots) {
			v = gs.PlayersPots[i]
		}
		off = putInt32(buf, off, v)
	}

	// stakes: 3×int32
	for i := 0; i < 3; i++ {
		v := int32(0)
		if i < len(gs.Stakes) {
			v = gs.Stakes[i]
		}
		off = putInt32(buf, off, v)
	}

	// legal_actions: 10×float32
	for a := int32(0); a < 10; a++ {
		v := float32(0)
		if _, ok := gs.LegalActions[a]; ok {
			v = 1.0
		}
		off = putFloat32(buf, off, v)
	}

	// stage: int32
	off = putInt32(buf, off, int32(gs.Stage))

	// current_player: int32
	off = putInt32(buf, off, gs.CurrentPlayer)

	// public_cards: 5×int32
	for i := 0; i < 5; i++ {
		v := int32(-1)
		if i < len(gs.PublicCards) {
			v = int32(gs.PublicCards[i])
		}
		off = putInt32(buf, off, v)
	}

	// private_cards: 2×int32
	for i := 0; i < 2; i++ {
		v := int32(0)
		if i < len(gs.PrivateCards) {
			v = int32(gs.PrivateCards[i])
		}
		off = putInt32(buf, off, v)
	}

	// regrets: 10×float32
	for a := int32(0); a < 10; a++ {
		off = putFloat32(buf, off, regrets[a])
	}

	// iteration: int32
	off = putInt32(buf, off, int32(iteration))

	// context_h: hidden_dim×float32
	for i := 0; i < m.hiddenDim; i++ {
		v := float32(0)
		if actorState != nil && i < len(actorState.LstmH) {
			v = actorState.LstmH[i]
		}
		off = putFloat32(buf, off, v)
	}

	return buf
}

func (m *BinaryMemoryBuffer) writer(counts [3]int, totalSeen [3]int64) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	batch := make([]sampleFlushRequest, 0, 4000)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	writeHeader := func() {
		hdr := make([]byte, 36) // counts(12) + totalSeen(24)
		for i := 0; i < 3; i++ {
			binary.LittleEndian.PutUint32(hdr[i*4:], uint32(counts[i]))
		}
		for i := 0; i < 3; i++ {
			binary.LittleEndian.PutUint64(hdr[12+i*8:], uint64(totalSeen[i]))
		}
		m.file.WriteAt(hdr, 20) // offset 20 in header
	}

	writeBatch := func() {
		if len(batch) == 0 {
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
				m.file.WriteAt(req.data, m.recordOffset(pid, slot))
			} else {
				j := rng.Int63n(n)
				if j < int64(m.maxSamples) {
					m.file.WriteAt(req.data, m.recordOffset(pid, int(j)))
				}
			}
		}
		writeHeader()
		batch = batch[:0]
	}

	for {
		select {
		case req, ok := <-m.flushCh:
			if !ok {
				writeBatch()
				m.file.Sync()
				close(m.done)
				return
			}
			if req.syncCh != nil {
				writeBatch()
				m.file.Sync()
				close(req.syncCh)
				continue
			}
			batch = append(batch, req)
			if len(batch) >= 4000 {
				writeBatch()
			}
		case <-ticker.C:
			writeBatch()
		}
	}
}

// AddSample encodes the sample as a binary record and sends it to the writer.
// No cloning needed — encodeRecord reads fields and produces an independent byte slice.
func (m *BinaryMemoryBuffer) AddSample(
	playerID int,
	gameID uuid.UUID,
	state *CFRState,
	regrets map[nolimitholdem.Action]float32,
	iteration int,
) {
	data := m.encodeRecord(state.GameState, state.ActorState, regrets, iteration)
	m.flushCh <- sampleFlushRequest{
		playerID: playerID,
		data:     data,
	}
}

// FlushGame is a no-op — samples are written individually in AddSample.
func (m *BinaryMemoryBuffer) FlushGame(playerID int, gameID uuid.UUID) {}

// GetSamples is not used — Python reads the binary file directly via numpy.memmap.
func (m *BinaryMemoryBuffer) GetSamples(playerID int, batchSize int) []*GameSample {
	return nil
}

func (m *BinaryMemoryBuffer) Count(playerID int) int {
	if playerID >= 0 && playerID < 3 {
		return int(m.countAtom[playerID].Load())
	}
	return 0
}

// Drain waits until all pending samples are written to disk.
func (m *BinaryMemoryBuffer) Drain() {
	ch := make(chan struct{})
	m.flushCh <- sampleFlushRequest{syncCh: ch}
	<-ch
}

func (m *BinaryMemoryBuffer) FilePath() string {
	return m.filePath
}

func (m *BinaryMemoryBuffer) Close() error {
	close(m.flushCh)
	<-m.done
	return m.file.Close()
}
