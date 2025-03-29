package nolimitholdem

import "testing"

func TestAllCombinations(t *testing.T) {
	tests := []struct {
		name     string
		cards    []Card
		expected string
	}{
		{
			name: "Straight Flush",
			cards: []Card{
				NewCard(7, 0), // 9♥
				NewCard(6, 0), // 8♥
				NewCard(5, 0), // 7♥
				NewCard(4, 0), // 6♥
				NewCard(3, 0), // 5♥
				NewCard(0, 1), // 2♦
				NewCard(1, 2), // 3♣
			},
			expected: "Straight Flush",
		},
		{
			name: "Four of a Kind",
			cards: []Card{
				NewCard(8, 0), // 10♥
				NewCard(8, 1), // 10♦
				NewCard(8, 2), // 10♣
				NewCard(8, 3), // 10♠
				NewCard(3, 0), // 5♥
				NewCard(0, 1), // 2♦
				NewCard(1, 2), // 3♣
			},
			expected: "Four of a Kind",
		},
		{
			name: "Full House",
			cards: []Card{
				NewCard(8, 0), // 10♥
				NewCard(8, 1), // 10♦
				NewCard(8, 2), // 10♣
				NewCard(3, 0), // 5♥
				NewCard(3, 1), // 5♦
				NewCard(0, 1), // 2♦
				NewCard(1, 2), // 3♣
			},
			expected: "Full House",
		},
		{
			name: "Flush",
			cards: []Card{
				NewCard(8, 0), // 10♥
				NewCard(6, 0), // 8♥
				NewCard(5, 0), // 7♥
				NewCard(4, 0), // 6♥
				NewCard(2, 0), // 4♥
				NewCard(0, 1), // 2♦
				NewCard(1, 2), // 3♣
			},
			expected: "Flush",
		},
		{
			name: "Straight",
			cards: []Card{
				NewCard(8, 0), // 10♥
				NewCard(7, 1), // 9♦
				NewCard(6, 2), // 8♣
				NewCard(5, 0), // 7♥
				NewCard(4, 1), // 6♦
				NewCard(0, 1), // 2♦
				NewCard(1, 2), // 3♣
			},
			expected: "Straight",
		},
		{
			name: "Three of a Kind",
			cards: []Card{
				NewCard(8, 0), // 10♥
				NewCard(8, 1), // 10♦
				NewCard(8, 2), // 10♣
				NewCard(3, 0), // 5♥
				NewCard(0, 1), // 2♦
				NewCard(1, 2), // 3♣
				NewCard(2, 3), // 4♠
			},
			expected: "Three of a Kind",
		},
		{
			name: "Two Pairs",
			cards: []Card{
				NewCard(8, 0), // 10♥
				NewCard(8, 1), // 10♦
				NewCard(3, 0), // 5♥
				NewCard(3, 1), // 5♦
				NewCard(0, 1), // 2♦
				NewCard(1, 2), // 3♣
				NewCard(2, 3), // 4♠
			},
			expected: "Two Pairs",
		},
		{
			name: "Pair",
			cards: []Card{
				NewCard(8, 0), // 10♥
				NewCard(8, 1), // 10♦
				NewCard(3, 0), // 5♥
				NewCard(0, 1), // 2♦
				NewCard(1, 2), // 3♣
				NewCard(2, 3), // 4♠
				NewCard(5, 0), // 7♥
			},
			expected: "Pair",
		},
		{
			name: "High Card",
			cards: []Card{
				NewCard(8, 0), // 10♥
				NewCard(6, 1), // 8♦
				NewCard(5, 2), // 7♣
				NewCard(3, 0), // 5♥
				NewCard(0, 1), // 2♦
				NewCard(1, 2), // 3♣
				NewCard(2, 3), // 4♠
			},
			expected: "High Card",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, comboName := EvaluateHand(tt.cards...)
			if comboName != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, comboName)
			}
		})
	}
}

func TestSpecificCombinations(t *testing.T) {
	t.Run("Two Pairs with Three Pairs", func(t *testing.T) {
		cards := []Card{
			NewCard(8, 0), NewCard(8, 1), // Пара 10
			NewCard(6, 0), NewCard(6, 1), // Пара 8
			NewCard(3, 0), NewCard(3, 1), // Пара 5
			NewCard(0, 2), // Кикер 2
		}

		result, ok := GetTwoPairs(cards...)
		if !ok {
			t.Fatal("Should find two pairs")
		}

		expectedRanks := []int16{8, 8, 6, 6}
		for i := 0; i < 4; i++ {
			if result[i].GetCardRank() != expectedRanks[i] {
				t.Errorf("Expected rank %d at position %d, got %d", expectedRanks[i], i, result[i].GetCardRank())
			}
		}

		if len(result) != 4 {
			t.Error("Should return exactly 4 cards")
		}
	})

	t.Run("Full House with Two Three of a Kinds", func(t *testing.T) {
		cards := []Card{
			NewCard(8, 0), NewCard(8, 1), NewCard(8, 2), // Тройка 10
			NewCard(6, 0), NewCard(6, 1), NewCard(6, 2), // Тройка 8
			NewCard(0, 3), // Лишняя карта
		}

		result, ok := GetFullHouse(cards...)
		if !ok {
			t.Fatal("Should find full house")
		}

		expectedRanks := []int16{8, 8, 8, 6, 6}
		for i := 0; i < 5; i++ {
			if result[i].GetCardRank() != expectedRanks[i] {
				t.Errorf("Expected rank %d at position %d, got %d", expectedRanks[i], i, result[i].GetCardRank())
			}
		}
	})

	t.Run("Wheel Straight (A-2-3-4-5)", func(t *testing.T) {
		cards := []Card{
			NewCard(12, 0), // A
			NewCard(0, 0),  // 2
			NewCard(1, 0),  // 3
			NewCard(2, 0),  // 4
			NewCard(3, 0),  // 5
			NewCard(8, 1),  // 10
			NewCard(5, 2),  // 7
		}

		result, ok := GetStraight(cards...)
		if !ok {
			t.Fatal("Should find wheel straight")
		}

		expectedRanks := []int16{12, 0, 1, 2, 3}
		for i := 0; i < 5; i++ {
			if result[i].GetCardRank() != expectedRanks[i] {
				t.Errorf("Expected rank %d at position %d, got %d", expectedRanks[i], i, result[i].GetCardRank())
			}
		}
	})
}
