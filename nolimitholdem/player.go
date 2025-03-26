package nolimitholdem

type PlayerStatus int

const (
	PLAYERSTATUS_ACTIVE = PlayerStatus(0)
	PLAYERSTATUS_FOLDED = PlayerStatus(1)
	PLAYERSTATUS_ALLIN  = PlayerStatus(2)
)

type Player struct {
	HoleCards [2]Card
	InitChips int

	RemainedChips int
	InChips       int

	Status PlayerStatus
}

func (h *Player) Bet(amount int) {
	if amount > h.RemainedChips {
		amount = h.RemainedChips
	}
	h.InChips += amount
	h.RemainedChips -= amount
}
