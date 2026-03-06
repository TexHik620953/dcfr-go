package nolimitholdem

type Strategy = map[Action]float32

type GameStage int

const (
	STAGE_PREFLOP    = GameStage(0)
	STAGE_FLOP       = GameStage(1)
	STAGE_TURN       = GameStage(2)
	STAGE_RIVER      = GameStage(3)
	STAGE_END_HIDDEN = GameStage(4)
	STAGE_SHOWDOWN   = GameStage(5)
)

type Action = int32

const (
	ACTION_FOLD            = Action(0)
	ACTION_CHECK_CALL      = Action(1)
	ACTION_RAISE_QUARTER   = Action(2) // 0.25x pot
	ACTION_RAISE_THIRD     = Action(3) // 0.33x pot
	ACTION_RAISE_HALFPOT   = Action(4) // 0.5x pot
	ACTION_RAISE_TWOTHIRDS = Action(5) // 0.66x pot
	ACTION_RAISE_POT       = Action(6) // 1x pot
	ACTION_RAISE_1_5X      = Action(7) // 1.5x pot
	ACTION_RAISE_2X        = Action(8) // 2x pot
	ACTION_ALL_IN          = Action(9)

	NUM_ACTIONS = 10
)

var Action2string = map[Action]string{
	ACTION_FOLD:            "FOLD",
	ACTION_CHECK_CALL:      "CHECK_CALL",
	ACTION_RAISE_QUARTER:   "RAISE_0.25x",
	ACTION_RAISE_THIRD:     "RAISE_0.33x",
	ACTION_RAISE_HALFPOT:   "RAISE_0.5x",
	ACTION_RAISE_TWOTHIRDS: "RAISE_0.66x",
	ACTION_RAISE_POT:       "RAISE_POT",
	ACTION_RAISE_1_5X:      "RAISE_1.5x",
	ACTION_RAISE_2X:        "RAISE_2x",
	ACTION_ALL_IN:          "ALL_IN",
}

// RaiseMultipliers maps raise actions to their pot multiplier (numerator, denominator)
var RaiseMultipliers = map[Action][2]int{
	ACTION_RAISE_QUARTER:   {1, 4},
	ACTION_RAISE_THIRD:     {1, 3},
	ACTION_RAISE_HALFPOT:   {1, 2},
	ACTION_RAISE_TWOTHIRDS: {2, 3},
	ACTION_RAISE_POT:       {1, 1},
	ACTION_RAISE_1_5X:      {3, 2},
	ACTION_RAISE_2X:        {2, 1},
}

type Card int32

func NewCard(rank, suit int16) Card {
	return Card(suit*13 + rank)
}

// 0 1 2 3 4 5 6 7 8 9 10 11 12
func (card Card) GetCardSuite() int16 {
	return int16(card / 13)
}

// 0 1 2 3
func (card Card) GetCardRank() int16 {
	return int16(card % 13)
}
