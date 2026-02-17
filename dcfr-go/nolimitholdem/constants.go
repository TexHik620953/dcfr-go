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
	ACTION_FOLD          = Action(0)
	ACTION_CHECK_CALL    = Action(1)
	ACTION_RAISE_HALFPOT = Action(2)
	ACTION_RAISE_POT     = Action(3)
	ACTION_ALL_IN        = Action(4)
)

var Action2string = map[Action]string{
	ACTION_FOLD:          "FOLD",
	ACTION_CHECK_CALL:    "CHECK_CALL",
	ACTION_RAISE_HALFPOT: "RAISE_HALFPOT",
	ACTION_RAISE_POT:     "RAISE_POT",
	ACTION_ALL_IN:        "ALL_IN",
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
