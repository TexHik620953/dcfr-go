package nolimitholdem

type Actor interface {
	GetAction(*GameState) Action
	GetProbs(*GameState) map[Action]float32
}
