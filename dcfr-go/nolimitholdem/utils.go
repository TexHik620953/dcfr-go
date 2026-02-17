package nolimitholdem

import (
	"slices"
)

func remove(l []Card, item Card) []Card {
	for i, other := range l {
		if other == item {
			return append(l[:i], l[i+1:]...)
		}
	}
	return l
}

func ConcatCards(holeCards, publicCards []Card) []Card {
	ply_total_cards := make([]Card, 0)
	ply_total_cards = append(ply_total_cards, holeCards...)
	ply_total_cards = append(ply_total_cards, publicCards...)
	return ply_total_cards
}

// Return flash cards with one suit
func GetFlush(cards ...Card) ([]Card, bool) {
	// For every suit
	for suit := range 4 {
		// Count cards
		cnt := 0
		for _, c := range cards {
			if c.GetCardSuite() == int16(suit) {
				cnt++
			}
		}
		if cnt >= 5 {
			// We got flash here
			flash_cards := make([]Card, 0)
			for _, c := range cards {
				if c.GetCardSuite() == int16(suit) {
					flash_cards = append(flash_cards, c)
				}
			}
			return flash_cards, true
		}
	}
	return nil, false
}

// Returns cards of straight
func GetStraight(cards ...Card) ([]Card, bool) {
	// Сортируем карты по рангу (от меньшего к большему)
	slices.SortFunc(cards, func(c1, c2 Card) int {
		return int(c1.GetCardRank()) - int(c2.GetCardRank())
	})

	// Убираем дубликаты рангов (например, два короля)
	uniqueCards := make([]Card, 0, len(cards))
	prevRank := int16(-1)
	for _, c := range cards {
		if c.GetCardRank() != prevRank {
			uniqueCards = append(uniqueCards, c)
			prevRank = c.GetCardRank()
		}
	}

	// Проверяем особый случай: стрит от туза до 5 (A-2-3-4-5)
	hasAce := len(uniqueCards) > 0 && uniqueCards[len(uniqueCards)-1].GetCardRank() == 12
	hasTwo := len(uniqueCards) > 0 && uniqueCards[0].GetCardRank() == 0
	if hasAce && hasTwo {
		// Проверяем наличие 3,4,5
		required := []int16{1, 2, 3}
		allPresent := true
		for _, r := range required {
			found := false
			for _, c := range uniqueCards {
				if c.GetCardRank() == r {
					found = true
					break
				}
			}
			if !found {
				allPresent = false
				break
			}
		}
		if allPresent {
			// Собираем стрит A-2-3-4-5
			straight := make([]Card, 0, 5)
			for _, r := range []int16{12, 0, 1, 2, 3} {
				for _, c := range cards {
					if c.GetCardRank() == r {
						straight = append(straight, c)
						break
					}
				}
			}
			return straight, true
		}
	}

	// Проверяем обычные стриты (5 подряд)
	if len(uniqueCards) >= 5 {
		for i := 0; i <= len(uniqueCards)-5; i++ {
			// Проверяем 5 карт подряд
			if uniqueCards[i+4].GetCardRank()-uniqueCards[i].GetCardRank() == 4 {
				// Нашли стрит, собираем карты
				straight := make([]Card, 0, 5)
				targetRank := uniqueCards[i].GetCardRank()
				for j := 0; j < 5; j++ {
					for _, c := range cards {
						if c.GetCardRank() == targetRank+int16(j) {
							straight = append(straight, c)
							break
						}
					}
				}
				return straight, true
			}
		}
	}

	return nil, false
}

// Return straight flash cards
func GetStraightFlash(cards ...Card) ([]Card, bool) {
	flashCards, isFlash := GetFlush(cards...)
	if !isFlash {
		return nil, false
	}
	return GetStraight(flashCards...)
}

// Returns four cards
func GetFour(cards ...Card) ([]Card, bool) {
	rankCount := make(map[int16][]Card)

	for _, c := range cards {
		rank := c.GetCardRank()
		rankCount[rank] = append(rankCount[rank], c)
	}

	for _, cards := range rankCount {
		if len(cards) >= 4 {
			// Возвращаем 4 карты одного ранга + старшую карту (кикер)
			result := cards[:4]
			return result, true
		}
	}

	return nil, false
}

// Returns three cards
func GetThree(cards ...Card) ([]Card, bool) {
	rankCount := make(map[int16][]Card)

	for _, c := range cards {
		rank := c.GetCardRank()
		rankCount[rank] = append(rankCount[rank], c)
	}

	for _, cards := range rankCount {
		if len(cards) >= 3 {
			// Возвращаем 3 карты одного ранга + 2 старшие карты (кикеры)
			result := cards[:3]
			return result, true
		}
	}

	return nil, false
}

// Returns pair cards
func GetPair(cards ...Card) ([]Card, bool) {
	rankCount := make(map[int16][]Card)

	for _, c := range cards {
		rank := c.GetCardRank()
		rankCount[rank] = append(rankCount[rank], c)
	}

	var pairs [][]Card
	for _, cards := range rankCount {
		if len(cards) >= 2 {
			pairs = append(pairs, cards[:2])
		}
	}

	if len(pairs) > 0 {
		// Сортируем пары по старшинству
		slices.SortFunc(pairs, func(a, b []Card) int {
			return int(b[0].GetCardRank()) - int(a[0].GetCardRank())
		})

		// Берем старшую пару
		result := pairs[0]
		return result, true
	}

	return nil, false
}

func GetTwoPairs(cards ...Card) ([]Card, bool) {
	// Создаем мапу для подсчета карт каждого ранга
	rankMap := make(map[int16][]Card)

	// Заполняем мапу
	for _, card := range cards {
		rank := card.GetCardRank()
		rankMap[rank] = append(rankMap[rank], card)
	}

	// Собираем все пары
	var pairs [][]Card
	for _, cards := range rankMap {
		if len(cards) >= 2 {
			pairs = append(pairs, cards[:2]) // Берем первые 2 карты этого ранга
		}
	}

	// Если меньше двух пар - комбинации нет
	if len(pairs) < 2 {
		return nil, false
	}

	// Сортируем пары по старшинству (от самой старшей к младшей)
	slices.SortFunc(pairs, func(a, b []Card) int {
		return int(b[0].GetCardRank()) - int(a[0].GetCardRank())
	})

	// Берем две старшие пары
	result := append(pairs[0], pairs[1]...)
	return result, true
}

// Returns full-house cards
func GetFullHouse(cards ...Card) ([]Card, bool) {
	rankCount := make(map[int16][]Card)

	for _, c := range cards {
		rank := c.GetCardRank()
		rankCount[rank] = append(rankCount[rank], c)
	}

	var threeOfAKind []Card
	var pair []Card

	for _, cards := range rankCount {
		if len(cards) >= 3 {
			if threeOfAKind == nil || cards[0].GetCardRank() > threeOfAKind[0].GetCardRank() {
				threeOfAKind = cards[:3]
			}
		}
	}

	if threeOfAKind == nil {
		return nil, false
	}

	for _, cards := range rankCount {
		if len(cards) >= 2 && cards[0].GetCardRank() != threeOfAKind[0].GetCardRank() {
			if pair == nil || cards[0].GetCardRank() > pair[0].GetCardRank() {
				pair = cards[:2]
			}
		}
	}

	if pair != nil {
		return append(threeOfAKind, pair...), true
	}

	return nil, false
}

func EvaluateHand(cards ...Card) ([]Card, int, string) {
	if len(cards) != 7 {
		panic("not enough cards for evaluations, 7 is required")
	}
	// Проверяем комбинации от старшей к младшей
	if combo, ok := GetStraightFlash(cards...); ok {
		return combo, 8, "Straight Flush"
	}
	if combo, ok := GetFour(cards...); ok {
		return combo, 7, "Four of a Kind"
	}
	if combo, ok := GetFullHouse(cards...); ok {
		return combo, 6, "Full House"
	}
	if combo, ok := GetFlush(cards...); ok {
		return combo, 5, "Flush"
	}
	if combo, ok := GetStraight(cards...); ok {
		return combo, 4, "Straight"
	}
	if combo, ok := GetThree(cards...); ok {
		return combo, 3, "Three of a Kind"
	}
	if combo, ok := GetTwoPairs(cards...); ok {
		return combo, 2, "Two Pairs"
	}
	if combo, ok := GetPair(cards...); ok {
		return combo, 1, "Pair"
	}

	cards = []Card{slices.MaxFunc(cards[:2], func(a, b Card) int {
		return int(a.GetCardRank() - b.GetCardRank())
	})}
	return cards, 0, "High Card"
}

type evalResult struct {
	playerId         int
	holeCards        []Card
	allCards         []Card
	combinationCards []Card

	category        int
	combName        string
	highestCardRank int16
}

func ComputeWinners(players_cards [][]Card, public_cards []Card) []int {
	data := make([]*evalResult, len(players_cards))
	folded := 0
	for i, c := range players_cards {
		data[i] = &evalResult{
			playerId: i,
		}
		if c == nil {
			// Player folded
			data[i].category = -1
			folded++
		} else {
			data[i].allCards = ConcatCards(c, public_cards)
			data[i].holeCards = make([]Card, 2)
			copy(data[i].holeCards, c)

			eval_c, eval_categ, comb_name := EvaluateHand(data[i].allCards...)
			data[i].combinationCards = eval_c
			data[i].category = eval_categ
			data[i].combName = comb_name
		}
	}

	result := make([]int, len(players_cards))
	if folded == len(players_cards)-1 {
		// All folded except one
		for i, ply := range data {
			if ply.category == -1 {
				result[i] = 0
			} else {
				result[i] = 1
			}
		}
		return result
	}

	slices.SortFunc(data, func(a, b *evalResult) int {
		return b.category - a.category
	})

	// Players with same category, check highest card to identify winners
	potentialWinners := make([]*evalResult, 0)
	for _, ply := range data {
		if ply.category == data[0].category {
			potentialWinners = append(potentialWinners, ply)
		}
	}
	// Only one winner, yay
	if len(potentialWinners) == 1 {
		winner := potentialWinners[0]
		result[winner.playerId] = 1
		return result
	}

	// If same combination, check for combination rank
	slices.SortFunc(potentialWinners, func(a, b *evalResult) int {
		return int(b.combinationCards[0].GetCardRank() - a.combinationCards[0].GetCardRank())
	})

	result[potentialWinners[0].playerId] = 1

	return result
}
