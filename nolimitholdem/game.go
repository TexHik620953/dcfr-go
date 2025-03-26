package nolimitholdem

import (
	"math/rand"
	"slices"
)

type GameState struct {
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
}

func NewGame(config GameConfig) *Game {
	randSource := rand.NewSource(config.RandomSeed)
	h := &Game{
		randGen:     rand.New(randSource),
		config:      config,
		players:     make([]*Player, config.NumPlayers),
		publicCards: make([]Card, 0),
		game_number: 0,
	}
	h.deck = NewDeck(h.randGen)
	h.Reset()
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

	return &GameState{}
}

func (h *Game) LegalActions() map[Action]struct{} {
	return h.round.LegalActions(h.players)
}

func (h *Game) Step(action Action) int {
	if _, ex := h.LegalActions()[action]; !ex {
		panic("action not allowed")
	}

	// Take snapshot here, before any action
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
			//log.Printf("Stage is now FLOP")
		}
		if h.round_counter == 1 {
			h.stage = STAGE_TURN
			h.publicCards = append(h.publicCards, h.deck.Get())
			if h.config.NumPlayers == bypassed_players_count {
				h.round_counter++
			}
			//log.Printf("Stage is now TURN")
		}
		if h.round_counter == 2 {
			h.stage = STAGE_RIVER
			h.publicCards = append(h.publicCards, h.deck.Get())
			if h.config.NumPlayers == bypassed_players_count {
				h.round_counter++
			}
			//log.Printf("Stage is now RIVER")
		}
		h.round_counter++
		h.round.StartNewRound(h.gamePointer, h.players)
	}
	return h.gamePointer
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

func (h *Game) GetPayoffs() []int {
	// Collect cards for all players and total chips
	players_cards := make([][]Card, h.config.NumPlayers)
	public_cards := make([]Card, len(h.publicCards))
	copy(public_cards, h.publicCards)

	remainingChips := 0
	for i, p := range h.players {
		remainingChips += p.InChips
		if p.Status == PLAYERSTATUS_ACTIVE || p.Status == PLAYERSTATUS_ALLIN {
			players_cards[i] = make([]Card, 2)
			copy(players_cards[i], p.HoleCards[:])
		} else {
			players_cards[i] = nil
		}
	}

	winners := ComputeWinners(players_cards, public_cards)
	winnersCount := 0
	for _, v := range winners {
		if v == 1 {
			winnersCount++
		}
	}

	payouts := make([]int, h.config.NumPlayers)
	for i, v := range winners {
		if v == 1 {
			payouts[i] = remainingChips / winnersCount
		}
	}

	return payouts
}

func (h *Game) GetState(playerId int) *GameState {
	return &GameState{}
}
