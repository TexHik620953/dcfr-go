package nolimitholdem

import (
	"slices"
)

// HandRank is a comparable rank vector for lexicographic comparison.
// Format: [category, ...ranks in priority order]
type HandRank []int16

func CompareHandRanks(a, b HandRank) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			if a[i] > b[i] {
				return 1
			}
			return -1
		}
	}
	return len(a) - len(b)
}

func ConcatCards(holeCards, publicCards []Card) []Card {
	result := make([]Card, 0, len(holeCards)+len(publicCards))
	result = append(result, holeCards...)
	result = append(result, publicCards...)
	return result
}

func getKickers(allCards []Card, comboCards []Card, numKickers int) []int16 {
	used := make(map[Card]bool, len(comboCards))
	for _, c := range comboCards {
		used[c] = true
	}
	remaining := make([]Card, 0, len(allCards)-len(comboCards))
	for _, c := range allCards {
		if !used[c] {
			remaining = append(remaining, c)
		}
	}
	slices.SortFunc(remaining, func(a, b Card) int {
		return int(b.GetCardRank() - a.GetCardRank())
	})
	result := make([]int16, 0, numKickers)
	for i := 0; i < numKickers && i < len(remaining); i++ {
		result = append(result, remaining[i].GetCardRank())
	}
	return result
}

func straightTopRank(combo []Card) int16 {
	hasAce := false
	hasTwo := false
	for _, c := range combo {
		if c.GetCardRank() == 12 {
			hasAce = true
		}
		if c.GetCardRank() == 0 {
			hasTwo = true
		}
	}
	if hasAce && hasTwo {
		return 3 // A-2-3-4-5: top is 5 (rank 3)
	}
	maxRank := int16(0)
	for _, c := range combo {
		if c.GetCardRank() > maxRank {
			maxRank = c.GetCardRank()
		}
	}
	return maxRank
}

// GetFlush returns flush cards (all cards of the flush suit)
func GetFlush(cards ...Card) ([]Card, bool) {
	for suit := range 4 {
		cnt := 0
		for _, c := range cards {
			if c.GetCardSuite() == int16(suit) {
				cnt++
			}
		}
		if cnt >= 5 {
			flushCards := make([]Card, 0, cnt)
			for _, c := range cards {
				if c.GetCardSuite() == int16(suit) {
					flushCards = append(flushCards, c)
				}
			}
			return flushCards, true
		}
	}
	return nil, false
}

// GetStraight returns the HIGHEST straight from given cards
func GetStraight(cards ...Card) ([]Card, bool) {
	slices.SortFunc(cards, func(c1, c2 Card) int {
		return int(c1.GetCardRank()) - int(c2.GetCardRank())
	})

	uniqueCards := make([]Card, 0, len(cards))
	prevRank := int16(-1)
	for _, c := range cards {
		if c.GetCardRank() != prevRank {
			uniqueCards = append(uniqueCards, c)
			prevRank = c.GetCardRank()
		}
	}

	// Check normal straights first (from highest window to find best)
	if len(uniqueCards) >= 5 {
		for i := len(uniqueCards) - 5; i >= 0; i-- {
			if uniqueCards[i+4].GetCardRank()-uniqueCards[i].GetCardRank() == 4 {
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

	// Wheel (A-2-3-4-5) as fallback
	hasAce := len(uniqueCards) > 0 && uniqueCards[len(uniqueCards)-1].GetCardRank() == 12
	hasTwo := len(uniqueCards) > 0 && uniqueCards[0].GetCardRank() == 0
	if hasAce && hasTwo {
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

	return nil, false
}

func GetStraightFlush(cards ...Card) ([]Card, bool) {
	flashCards, isFlash := GetFlush(cards...)
	if !isFlash {
		return nil, false
	}
	return GetStraight(flashCards...)
}

func GetFour(cards ...Card) ([]Card, bool) {
	rankCount := make(map[int16][]Card)
	for _, c := range cards {
		rankCount[c.GetCardRank()] = append(rankCount[c.GetCardRank()], c)
	}
	var best []Card
	for _, group := range rankCount {
		if len(group) >= 4 {
			if best == nil || group[0].GetCardRank() > best[0].GetCardRank() {
				best = group[:4]
			}
		}
	}
	if best != nil {
		return best, true
	}
	return nil, false
}

func GetThree(cards ...Card) ([]Card, bool) {
	rankCount := make(map[int16][]Card)
	for _, c := range cards {
		rankCount[c.GetCardRank()] = append(rankCount[c.GetCardRank()], c)
	}
	var best []Card
	for _, group := range rankCount {
		if len(group) >= 3 {
			if best == nil || group[0].GetCardRank() > best[0].GetCardRank() {
				best = group[:3]
			}
		}
	}
	if best != nil {
		return best, true
	}
	return nil, false
}

func GetPair(cards ...Card) ([]Card, bool) {
	rankCount := make(map[int16][]Card)
	for _, c := range cards {
		rankCount[c.GetCardRank()] = append(rankCount[c.GetCardRank()], c)
	}
	var best []Card
	for _, group := range rankCount {
		if len(group) >= 2 {
			if best == nil || group[0].GetCardRank() > best[0].GetCardRank() {
				best = group[:2]
			}
		}
	}
	if best != nil {
		return best, true
	}
	return nil, false
}

func GetTwoPairs(cards ...Card) ([]Card, bool) {
	rankCount := make(map[int16][]Card)
	for _, c := range cards {
		rankCount[c.GetCardRank()] = append(rankCount[c.GetCardRank()], c)
	}

	var pairs [][]Card
	for _, group := range rankCount {
		if len(group) >= 2 {
			pairs = append(pairs, group[:2])
		}
	}
	if len(pairs) < 2 {
		return nil, false
	}
	slices.SortFunc(pairs, func(a, b []Card) int {
		return int(b[0].GetCardRank()) - int(a[0].GetCardRank())
	})
	result := make([]Card, 0, 4)
	result = append(result, pairs[0]...)
	result = append(result, pairs[1]...)
	return result, true
}

func GetFullHouse(cards ...Card) ([]Card, bool) {
	rankCount := make(map[int16][]Card)
	for _, c := range cards {
		rankCount[c.GetCardRank()] = append(rankCount[c.GetCardRank()], c)
	}

	var trips []Card
	for _, group := range rankCount {
		if len(group) >= 3 {
			if trips == nil || group[0].GetCardRank() > trips[0].GetCardRank() {
				trips = group[:3]
			}
		}
	}
	if trips == nil {
		return nil, false
	}

	var pair []Card
	for _, group := range rankCount {
		if len(group) >= 2 && group[0].GetCardRank() != trips[0].GetCardRank() {
			if pair == nil || group[0].GetCardRank() > pair[0].GetCardRank() {
				pair = group[:2]
			}
		}
	}
	if pair == nil {
		return nil, false
	}

	result := make([]Card, 0, 5)
	result = append(result, trips...)
	result = append(result, pair...)
	return result, true
}

// EvaluateHandRank returns a comparable HandRank for 7 cards (2 hole + 5 community)
func EvaluateHandRank(allCards []Card) HandRank {
	if combo, ok := GetStraightFlush(allCards...); ok {
		return HandRank{8, straightTopRank(combo)}
	}
	if combo, ok := GetFour(allCards...); ok {
		rank := HandRank{7, combo[0].GetCardRank()}
		return append(rank, getKickers(allCards, combo, 1)...)
	}
	if combo, ok := GetFullHouse(allCards...); ok {
		return HandRank{6, combo[0].GetCardRank(), combo[3].GetCardRank()}
	}
	if flushCards, ok := GetFlush(allCards...); ok {
		slices.SortFunc(flushCards, func(a, b Card) int {
			return int(b.GetCardRank() - a.GetCardRank())
		})
		rank := HandRank{5}
		for i := 0; i < 5 && i < len(flushCards); i++ {
			rank = append(rank, flushCards[i].GetCardRank())
		}
		return rank
	}
	if combo, ok := GetStraight(allCards...); ok {
		return HandRank{4, straightTopRank(combo)}
	}
	if combo, ok := GetThree(allCards...); ok {
		rank := HandRank{3, combo[0].GetCardRank()}
		return append(rank, getKickers(allCards, combo, 2)...)
	}
	if combo, ok := GetTwoPairs(allCards...); ok {
		highPair := max(combo[0].GetCardRank(), combo[2].GetCardRank())
		lowPair := min(combo[0].GetCardRank(), combo[2].GetCardRank())
		rank := HandRank{2, highPair, lowPair}
		return append(rank, getKickers(allCards, combo, 1)...)
	}
	if combo, ok := GetPair(allCards...); ok {
		rank := HandRank{1, combo[0].GetCardRank()}
		return append(rank, getKickers(allCards, combo, 3)...)
	}

	sorted := make([]Card, len(allCards))
	copy(sorted, allCards)
	slices.SortFunc(sorted, func(a, b Card) int {
		return int(b.GetCardRank() - a.GetCardRank())
	})
	rank := HandRank{0}
	for i := 0; i < 5 && i < len(sorted); i++ {
		rank = append(rank, sorted[i].GetCardRank())
	}
	return rank
}

var categoryNames = [9]string{
	"High Card", "Pair", "Two Pairs", "Three of a Kind",
	"Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush",
}

// EvaluateHand returns (combo cards, category, name) for backward compatibility
func EvaluateHand(cards ...Card) ([]Card, int, string) {
	if combo, ok := GetStraightFlush(cards...); ok {
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
	sorted := make([]Card, len(cards))
	copy(sorted, cards)
	slices.SortFunc(sorted, func(a, b Card) int {
		return int(b.GetCardRank() - a.GetCardRank())
	})
	return sorted[:5], 0, "High Card"
}

// ComputeWinners returns a bitmask: 1 = winner, 0 = loser. Supports split pots (ties).
func ComputeWinners(playersCards [][]Card, publicCards []Card) []int {
	numPlayers := len(playersCards)
	result := make([]int, numPlayers)

	handRanks := make([]HandRank, numPlayers)
	for i, cards := range playersCards {
		if cards == nil {
			handRanks[i] = HandRank{-1}
		} else {
			handRanks[i] = EvaluateHandRank(ConcatCards(cards, publicCards))
		}
	}

	bestRank := HandRank{-1}
	for _, rank := range handRanks {
		if CompareHandRanks(rank, bestRank) > 0 {
			bestRank = rank
		}
	}

	for i, rank := range handRanks {
		if CompareHandRanks(rank, bestRank) == 0 {
			result[i] = 1
		}
	}
	return result
}
