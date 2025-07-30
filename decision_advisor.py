"""
Provides player decision advice based on S17 basic strategy, common index plays,
and shoe composition analysis.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shoe import Shoe

STRATEGY_CONFIG = {
    "insurance_threshold": 3.0,
    "index_plays": {
        "16-vs-10": {"action": "Stand", "threshold": 0},
        "15-vs-10": {"action": "Stand", "threshold": 4},
        "13-vs-2":  {"action": "Stand", "threshold": -1},
        "12-vs-2":  {"action": "Stand", "threshold": 3},
        "12-vs-3":  {"action": "Stand", "threshold": 2},
        "12-vs-4":  {"action": "Stand", "threshold": -1, "condition": "below"},
        "10-vs-A":  {"action": "Double", "threshold": 4},
        "9-vs-2":   {"action": "Double", "threshold": 1},
        "9-vs-7":   {"action": "Double", "threshold": 3},
    }
}

class Hand:
    """Represents a player's or dealer's hand of cards."""
    def __init__(self, cards: list[str]):
        if not isinstance(cards, list):
            raise TypeError("Hand must be initialized with a list of card strings.")
        self.cards = cards
        self.value, self.is_soft = self._calculate_value()

    def _calculate_value(self) -> tuple[int, bool]:
        """Calculates the blackjack value of the hand, handling aces correctly."""
        total = 0
        ace_count = 0
        for card in self.cards:
            rank = card[:-1] if card.startswith("10") else card[0]
            if rank == 'A':
                ace_count += 1
                total += 11
            elif rank in ('K', 'Q', 'J', '10'):
                total += 10
            else:
                total += int(rank)
        
        soft = ace_count > 0
        while total > 21 and ace_count > 0:
            total -= 10
            ace_count -= 1
        
        if total > 21 and soft:
             soft = False
        elif ace_count == 0:
             soft = False

        return total, soft

    @property
    def is_pair(self) -> bool:
        """Checks if the hand is a pair (first two cards)."""
        if len(self.cards) != 2:
            return False
        rank1 = self.cards[0][:-1] if self.cards[0].startswith("10") else self.cards[0][0]
        rank2 = self.cards[1][:-1] if self.cards[1].startswith("10") else self.cards[1][0]
        return rank1 == rank2

def recommend_action(
    player_hand_cards: list[str],
    dealer_upcard: str,
    true_count: float,
    config: dict | None = None
) -> str:
    """
    Recommends a blackjack action based on S17 basic strategy and index plays.
    """
    if not player_hand_cards or not dealer_upcard:
        return "Awaiting Player and Dealer cards."

    cfg = config or STRATEGY_CONFIG
    player_hand = Hand(player_hand_cards)
    
    dealer_rank_str = dealer_upcard[0]
    if dealer_upcard.startswith("10"): dealer_rank_str = "10"
    if dealer_rank_str in ('J', 'Q', 'K'): dealer_rank_str = "10"
    
    dealer_up_value = Hand([dealer_upcard]).value

    if player_hand.value == 21 and len(player_hand.cards) == 2:
        return "Blackjack!"

    if dealer_rank_str == 'A' and len(player_hand.cards) == 2:
        return "Take Insurance" if true_count >= cfg["insurance_threshold"] else "Decline Insurance"

    # --- Splitting Logic ---
    if player_hand.is_pair:
        pair_rank_str = player_hand.cards[0][0]
        if pair_rank_str == 'A' or pair_rank_str == '8': return "Split"
        if pair_rank_str == '9' and dealer_up_value not in [7, 10, 11]: return "Split"
        if pair_rank_str == '7' and dealer_up_value <= 7: return "Split"
        if pair_rank_str == '6' and dealer_up_value <= 6: return "Split"
        # Only split 4s vs 5, 6 when Double After Split is allowed. Assume it is for advice.
        if pair_rank_str == '4' and dealer_up_value in [5, 6]: return "Split"
        if pair_rank_str in ['2', '3'] and dealer_up_value <= 7: return "Split"
        # Note: Do not split 10s or 5s in basic strategy.

    # --- Index Play Lookup (after splitting decisions) ---
    index_key = f"{player_hand.value}-vs-{dealer_rank_str}"
    if index_key in cfg["index_plays"]:
        play = cfg["index_plays"][index_key]
        threshold = play["threshold"]
        condition = play.get("condition", "above")
        
        if (condition == "above" and true_count >= threshold) or \
           (condition == "below" and true_count < threshold):
            return f"{play['action']} (Index Play)"

    # --- Soft Totals ---
    if player_hand.is_soft:
        if player_hand.value >= 19: return "Stand"
        if player_hand.value == 18:
            if dealer_up_value >= 9: return "Hit"
            if dealer_up_value in [2, 7, 8]: return "Stand"
            return "Double" if len(player_hand.cards) == 2 else "Stand"
        if len(player_hand.cards) == 2:
            if player_hand.value == 17 and dealer_up_value in [3,4,5,6]: return "Double"
            if player_hand.value in [15,16] and dealer_up_value in [4,5,6]: return "Double"
            if player_hand.value in [13,14] and dealer_up_value in [5,6]: return "Double"
        return "Hit"

    # --- Hard Totals ---
    if player_hand.value >= 17: return "Stand"
    if player_hand.value >= 13 and dealer_up_value <= 6: return "Stand"
    if player_hand.value == 12 and dealer_up_value in [4,5,6]: return "Stand"
    
    if len(player_hand.cards) == 2:
        if player_hand.value == 11: return "Double"
        if player_hand.value == 10 and dealer_up_value <= 9: return "Double"
        if player_hand.value == 9 and dealer_up_value in [3,4,5,6]: return "Double"
        
    return "Hit"