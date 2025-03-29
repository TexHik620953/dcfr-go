package nolimitholdem

import (
	"hash/fnv"
	"sort"
)

type GameState struct {
	PlayersPots       []int32
	Stakes            []int32
	ActivePlayersMask []int32
	LegalActions      map[Action]struct{}
	Stage             GameStage
	CurrentPlayer     int32

	PublicCards  []Card
	PrivateCards []Card
}
type GameStateHash uint64

func (gs *GameState) Hash() GameStateHash {
	if gs == nil {
		return 0
	}

	h := fnv.New64a()

	// Хэшируем PlayersPots
	for _, pot := range gs.PlayersPots {
		h.Write([]byte{byte(pot >> 8), byte(pot)})
	}

	// Хэшируем ActivePlayersMask
	for _, ap := range gs.ActivePlayersMask {
		h.Write([]byte{byte(ap >> 8), byte(ap)})
	}

	// Хэшируем Stakes
	for _, stake := range gs.Stakes {
		h.Write([]byte{byte(stake >> 8), byte(stake)})
	}

	// Хэшируем LegalActions (сортируем для детерминизма)
	actions := make([]Action, 0, len(gs.LegalActions))
	for a := range gs.LegalActions {
		actions = append(actions, a)
	}
	sort.Slice(actions, func(i, j int) bool {
		return actions[i] < actions[j]
	})
	for _, a := range actions {
		h.Write([]byte{byte(a >> 8), byte(a)})
	}

	// Хэшируем Stage и CurrentPlayer
	h.Write([]byte{byte(gs.Stage), byte(gs.CurrentPlayer)})

	// Хэшируем PublicCards (уже сортированы по определению)
	for _, card := range gs.PublicCards {
		h.Write([]byte{byte(card >> 8), byte(card)})
	}

	// Хэшируем PrivateCards
	for _, card := range gs.PrivateCards {
		h.Write([]byte{byte(card >> 8), byte(card)})
	}

	return GameStateHash(h.Sum64())
}

func (h *GameState) Clone() *GameState {
	cp := &GameState{
		Stage:         h.Stage,
		CurrentPlayer: h.CurrentPlayer,
	}

	// Копируем слайсы
	cp.ActivePlayersMask = make([]int32, len(h.ActivePlayersMask))
	copy(cp.ActivePlayersMask, h.ActivePlayersMask)

	cp.PlayersPots = make([]int32, len(h.PlayersPots))
	copy(cp.PlayersPots, h.PlayersPots)

	cp.Stakes = make([]int32, len(h.Stakes))
	copy(cp.Stakes, h.Stakes)

	cp.PublicCards = make([]Card, len(h.PublicCards))
	copy(cp.PublicCards, h.PublicCards)

	cp.PrivateCards = make([]Card, len(h.PrivateCards))
	copy(cp.PrivateCards, h.PrivateCards)

	// Копируем map (LegalActions)
	cp.LegalActions = make(map[Action]struct{}, len(h.LegalActions))
	for action := range h.LegalActions {
		cp.LegalActions[action] = struct{}{}
	}
	return cp
}
