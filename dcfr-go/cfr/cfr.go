package cfr

import (
	"context"
	"dcfr-go/nolimitholdem"
	"fmt"
	"log"
	"sync/atomic"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
)

var createBlackSquareNode = func(graph *graphviz.Graph, name, label string) *cgraph.Node {
	node, err := graph.CreateNodeByName(name)
	if err != nil {
		log.Fatal(err)
	}
	node.SetShape(cgraph.BoxShape) // Квадратная форма
	node.SetStyle("filled")        // Включить заливку
	node.SetFillColor("black")     // Чёрный цвет заливки
	node.SetFontColor("white")     // Белый текст
	node.SetLabel(label)           // Подпись узла
	node.SetWidth(1)               // Ширина (опционально)
	node.SetHeight(0.4)            // Высота (опционально)
	return node
}

var createEdge = func(graph *graphviz.Graph, from, to *cgraph.Node) {
	edge, _ := graph.CreateEdgeByName("", from, to)
	edge.SetPenWidth(5) // Толщина линии
	edge.SetArrowSize(0.5)
	edge.SetColor("#333333") // Цвет
}

type CFR struct {
	coreGame     *nolimitholdem.Game
	actor        *DeepCFRActor
	Memory       *MemoryBuffer
	iteration    int
	nodesVisited atomic.Int32
}

func New(game *nolimitholdem.Game, actor *DeepCFRActor, memory *MemoryBuffer) *CFR {
	h := &CFR{
		coreGame:  game,
		actor:     actor,
		Memory:    memory,
		iteration: 0,
	}

	h.coreGame.Reset()

	return h
}

func (h *CFR) TraverseTree(playerId int, buildGraph bool) ([]float32, error) {
	// Инициализация вероятностей достижения состояния
	reachProbs := make([]float32, h.coreGame.PlayersCount())
	for i := range reachProbs {
		reachProbs[i] = 1.0
	}

	ctx := context.Background()

	var gv *graphviz.Graphviz
	var graph *graphviz.Graph
	var err error
	var node *cgraph.Node
	if buildGraph {
		gv, err = graphviz.New(ctx)
		if err != nil {
			return nil, err
		}
		graph, err = gv.Graph()
		if err != nil {
			return nil, err
		}

		// Настройки графа
		graph.SetSize(40, 40)
		graph.SetRankSeparator(3.0) // Больше расстояния между уровнями
		graph.SetNodeSeparator(0.5) // Больше расстояния между узлами
		graph.SetLayout("dot")      // алгоритм раскладки
		graph.SetNodeSeparator(0.8) // расстояние между узлами
		graph.SetOverlap(false)     // избегать наложения
		graph.SetSplines("polyline")

		node = createBlackSquareNode(graph, "ROOT", fmt.Sprintf("p%d", h.coreGame.CurrentPlayer()))
		node.SetRoot(true)
	}

	payoffs, err := h.traverser(reachProbs, playerId, graph, node)
	if err != nil {
		return nil, err
	}
	h.iteration++

	if buildGraph {
		//graph.SetSplines("curved")
		err = gv.RenderFilename(ctx, graph, graphviz.PNG, "output.png")
		if err != nil {
			return nil, err
		}

	}

	return payoffs, nil
}

func (h *CFR) traverser(reachProbs []float32, learnerId int, graph *graphviz.Graph, node *cgraph.Node) ([]float32, error) {
	h.nodesVisited.Add(1)

	if h.nodesVisited.Load()%50000 == 0 {
		log.Printf("Visited %d nodes\n", h.nodesVisited.Load())
	}

	if h.coreGame.IsOver() {
		return h.coreGame.GetPayoffs(), nil
	}

	currentPlayer := h.coreGame.CurrentPlayer()
	state := h.coreGame.GetState(currentPlayer)
	actionProbs, err := h.actor.GetProbs(learnerId, state)
	if err != nil {
		return nil, err
	}

	totalPayoffs := make([]float32, h.coreGame.PlayersCount())
	actionPayoffs := make(map[nolimitholdem.Action][]float32)

	// Iterate over all possible actions
	for action, action_probability := range actionProbs {
		//Make a copy of original probabilities
		newReachProbs := make([]float32, len(reachProbs))
		copy(newReachProbs, reachProbs)
		newReachProbs[currentPlayer] *= action_probability

		var actionNode *cgraph.Node
		if graph != nil && node != nil {

			actionNode = createBlackSquareNode(graph, fmt.Sprintf("%d", h.nodesVisited.Load()), fmt.Sprintf("p%d", h.coreGame.CurrentPlayer()))
			createEdge(graph, node, actionNode)
		}

		h.coreGame.Step(action)
		childPayoffs, err := h.traverser(newReachProbs, learnerId, graph, actionNode)
		if err != nil {
			return nil, err
		}
		h.coreGame.StepBack()

		// Сохраняем результаты
		actionPayoffs[action] = childPayoffs
		for i, payoff := range childPayoffs {
			totalPayoffs[i] += float32(payoff) * action_probability
		}
	}

	if currentPlayer != learnerId {
		return totalPayoffs, nil
	}

	// CFR HERE
	regrets := make(nolimitholdem.Strategy)
	for action, payoffs := range actionPayoffs {
		regret := payoffs[learnerId] - totalPayoffs[learnerId]
		// CFR+: только положительные сожаления
		if regret > 0 {
			// Учитываем вероятность достижения оппонентами (product всех reachProbs кроме текущего игрока)
			oppReach := float32(1.0)
			for i, prob := range reachProbs {
				if i != learnerId {
					oppReach *= prob
				}
			}
			regrets[action] = regret * oppReach
		}
	}
	if len(regrets) > 0 {
		h.Memory.AddSample(learnerId, state, regrets, reachProbs[learnerId], h.iteration)
	}

	return totalPayoffs, nil
}
