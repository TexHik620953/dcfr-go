package nolimitholdem

import (
	"math/rand"
	"slices"
)

type GameState struct {
	PlayersPots   []int
	Stakes        []int
	LegalActions  map[Action]struct{}
	Stage         GameStage
	CurrentPlayer int

	PublicCards  []Card
	PrivateCards []Card
}

func (h *GameState) Clone() *GameState {
	cp := &GameState{
		Stage:         h.Stage,
		CurrentPlayer: h.CurrentPlayer,
	}

	// Копируем слайсы
	cp.PlayersPots = make([]int, len(h.PlayersPots))
	copy(cp.PlayersPots, h.PlayersPots)

	cp.Stakes = make([]int, len(h.Stakes))
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

type GameConfig struct {
	RandomSeed      int64
	ChipsForEach    int
	NumPlayers      int
	SmallBlindChips int
	LogEnabled      bool
}
type Game struct {
	randGen *rand.Rand
	config  GameConfig

	deck    *Deck
	players []*Player

	publicCards []Card
	stage       GameStage

	deallerId   int
	gamePointer int

	round         *Round
	round_counter int
	game_number   int

	history []*Game
}

func (g *Game) DeepCopy() *Game {
	cp := &Game{
		config:        g.config,
		stage:         g.stage,
		deallerId:     g.deallerId,
		gamePointer:   g.gamePointer,
		round_counter: g.round_counter,
		game_number:   g.game_number,
		randGen:       g.randGen,
	}

	// Глубокое копирование колоды
	cp.deck = g.deck.DeepCopy()

	// Глубокое копирование игроков
	if g.players != nil {
		cp.players = make([]*Player, len(g.players))
		for i, player := range g.players {
			if player != nil {
				cp.players[i] = player.DeepCopy()
			}
		}
	}

	// Глубокое копирование общественных карт
	if g.publicCards != nil {
		cp.publicCards = make([]Card, len(g.publicCards))
		copy(cp.publicCards, g.publicCards)
	}

	// Глубокое копирование раунда
	if g.round != nil {
		cp.round = g.round.DeepCopy()
	}
	return cp
}
func (g *Game) load(cp *Game) {
	if cp == nil {
		return
	}
	// Восстанавливаем простые поля
	g.config = cp.config
	g.stage = cp.stage
	g.deallerId = cp.deallerId
	g.gamePointer = cp.gamePointer
	g.round_counter = cp.round_counter
	g.game_number = cp.game_number

	// Восстанавливаем генератор случайных чисел
	g.randGen = cp.randGen

	// Восстанавливаем игроков
	g.players = make([]*Player, len(cp.players))
	copy(g.players, cp.players)

	// Восстанавливаем общественные карты
	g.publicCards = make([]Card, len(cp.publicCards))
	copy(g.publicCards, cp.publicCards)

	// Восстанавливаем раунд
	g.round = cp.round

	g.deck = cp.deck
}

func NewGame(config GameConfig) *Game {
	randSource := rand.NewSource(config.RandomSeed)
	h := &Game{
		randGen:     rand.New(randSource),
		config:      config,
		players:     make([]*Player, config.NumPlayers),
		publicCards: make([]Card, 0),
		game_number: 0,
		history:     make([]*Game, 0),
	}
	h.deck = NewDeck(h.randGen)
	return h
}

func (h *Game) Reset() *GameState {
	//Reset deck
	h.deck.Reset()
	// Create players, and deal hole cards
	for i := range h.config.NumPlayers {
		h.players[i] = &Player{
			InitChips:     int(h.config.ChipsForEach),
			RemainedChips: int(h.config.ChipsForEach),
			HoleCards:     [2]Card{h.deck.Get(), h.deck.Get()},
			InChips:       0,
			Status:        PLAYERSTATUS_ACTIVE,
		}
	}
	h.history = make([]*Game, 0)
	h.stage = STAGE_PREFLOP
	h.round_counter = 0
	h.publicCards = h.publicCards[:0]
	h.game_number++

	//Set dealler
	h.deallerId = int(h.randGen.Int63()) % int(h.config.NumPlayers)

	// Small and big blind positions
	sb := (h.deallerId + 1) % h.config.NumPlayers
	bb := (h.deallerId + 2) % h.config.NumPlayers

	// Make them bet
	h.players[sb].Bet(h.config.SmallBlindChips)
	h.players[bb].Bet(h.config.SmallBlindChips * 2)

	// Player next to big blind plays first
	h.gamePointer = (bb + 1) % h.config.NumPlayers

	h.round = newRound(roundConfig{
		numPlayers: h.config.NumPlayers,
		bigBlind:   h.config.SmallBlindChips * 2,
		deallerId:  h.deallerId,
	})

	h.round.StartNewRound(h.gamePointer, h.players)

	return h.GetState(h.gamePointer)
}

func (h *Game) LegalActions() map[Action]struct{} {
	return h.round.LegalActions(h.players)
}

func (h *Game) Step(action Action) {
	if _, ex := h.LegalActions()[action]; !ex {
		panic("action not allowed")
	}

	// Take snapshot here, before any action
	snapshot := h.DeepCopy()
	h.history = append(h.history, snapshot)

	//log.Printf("Player %d makes %s\n", h.gamePointer, action2string[action])

	h.gamePointer = h.round.ProceedRound(h.players, action)

	//log.Printf("Turn goes to player %d", h.gamePointer)

	bypassed_players_count := 0
	players_in_bypass := make([]int, h.config.NumPlayers)
	remaining_player := -1
	for i, p := range h.players {
		if p.Status == PLAYERSTATUS_FOLDED || p.Status == PLAYERSTATUS_ALLIN {
			players_in_bypass[i] = 1
			bypassed_players_count++
		} else {
			players_in_bypass[i] = 0
			remaining_player = i
		}
	}

	if h.config.NumPlayers-bypassed_players_count == 1 {
		if h.round.round_raised[remaining_player] >= slices.Max(h.round.round_raised) {
			players_in_bypass[remaining_player] = 1
			bypassed_players_count++
		}
	}

	if h.round.IsOver() {
		h.gamePointer = (h.deallerId + 1) % h.config.NumPlayers
		if bypassed_players_count < h.config.NumPlayers {
			for players_in_bypass[h.gamePointer] == 1 {
				h.gamePointer = (h.gamePointer + 1) % h.config.NumPlayers
			}
		}
		//log.Printf("Turn goes to player %d", h.gamePointer)

		if h.round_counter == 0 {
			h.stage = STAGE_FLOP
			// Deal 3 cards to public
			h.publicCards = append(h.publicCards, h.deck.Get(), h.deck.Get(), h.deck.Get())
			if h.config.NumPlayers == bypassed_players_count {
				h.round_counter++
			}
			//log.Printf("Stage is now FLOP: %d", len(h.publicCards))
		}
		if h.round_counter == 1 {
			h.stage = STAGE_TURN
			h.publicCards = append(h.publicCards, h.deck.Get())
			if h.config.NumPlayers == bypassed_players_count {
				h.round_counter++
			}
			//log.Printf("Stage is now TURN: %d", len(h.publicCards))
		}
		if h.round_counter == 2 {
			h.stage = STAGE_RIVER
			h.publicCards = append(h.publicCards, h.deck.Get())
			if h.config.NumPlayers == bypassed_players_count {
				h.round_counter++
			}
			//log.Printf("Stage is now RIVER: %d", len(h.publicCards))
		}
		h.round_counter++
		h.round.StartNewRound(h.gamePointer, h.players)
	}
}

func (h *Game) StepBack() {
	if len(h.history) == 0 {
		panic("no checkpoint to step back")
	}
	popped := h.history[len(h.history)-1]
	h.history = h.history[:len(h.history)-1]
	h.load(popped)
}

func (h *Game) IsOver() bool {
	alive_count := 0
	alive_players := make([]int, h.config.NumPlayers)
	for i, p := range h.players {
		if p.Status == PLAYERSTATUS_ACTIVE || p.Status == PLAYERSTATUS_ALLIN {
			alive_players[i] = 1
			alive_count++
		} else {
			alive_players[i] = 0
		}
	}
	if alive_count == 1 {
		return true
	}
	if h.round_counter >= 4 {
		return true
	}
	return false
}

func (h *Game) GetPayoffs() []float32 {
	// Collect cards for all players and total chips
	players_cards := make([][]Card, h.config.NumPlayers)
	public_cards := make([]Card, len(h.publicCards))
	copy(public_cards, h.publicCards)

	active_players := 0
	remainingChips := float32(0)
	for i, p := range h.players {
		remainingChips += float32(p.InChips)
		if p.Status == PLAYERSTATUS_ACTIVE || p.Status == PLAYERSTATUS_ALLIN {
			players_cards[i] = make([]Card, 2)
			copy(players_cards[i], p.HoleCards[:])
			active_players++
		} else {
			players_cards[i] = nil
		}
	}

	// All other folded, we have winner
	payouts := make([]float32, len(h.players))
	if active_players == 1 {
		for i, p := range h.players {
			if p.Status != PLAYERSTATUS_FOLDED {
				payouts[i] = remainingChips
			} else {
				payouts[i] = -remainingChips / float32(len(h.players)-1)
			}
		}
		return payouts
	}

	winners := ComputeWinners(players_cards, public_cards)
	winnersCount := 0
	for _, v := range winners {
		if v == 1 {
			winnersCount++
		}
	}
	for i, v := range winners {
		if v == 1 {
			payouts[i] = remainingChips / float32(winnersCount)
		} else {
			payouts[i] = -remainingChips / float32(h.round.numPlayers-winnersCount)
		}
	}

	return payouts
}

func (h *Game) CurrentPlayer() int {
	return h.gamePointer
}
func (h *Game) PlayersCount() int {
	return h.config.NumPlayers
}

func (h *Game) GetState(playerId int) *GameState {
	// Public info
	state := &GameState{
		PlayersPots:   make([]int, len(h.players)),
		Stakes:        make([]int, len(h.players)),
		Stage:         h.stage,
		CurrentPlayer: h.gamePointer,
		PublicCards:   make([]Card, len(h.publicCards)),
		PrivateCards:  make([]Card, 2),
		LegalActions:  h.LegalActions(),
	}
	copy(state.PublicCards, h.publicCards)
	copy(state.PrivateCards, h.players[playerId].HoleCards[:])

	for i, ply := range h.players {
		state.PlayersPots[i] = ply.InChips
		state.Stakes[i] = ply.RemainedChips
	}

	// Private info
	return state
}
