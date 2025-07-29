# decision_advisor.py
"""
Provides player decision advice based on the Omega II counting system,
index plays, and Bayesian analysis of the remaining shoe composition.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import bayesian_predictor

if TYPE_CHECKING:
    from shoe import Shoe

OMEGA_II_CONFIG = {
    "insurance_threshold": 2.0,
    "index_plays": {
        "16v10": 0, "15v10": 2, "14v10": 4, "13v2": -1, "12v3": 1, "12v4": -2,
        "11vA": -1, "10vA": 3, "10vT": 3, "9v2": 1, "9v7": 3,
    }
}

class Hand:
    """Represents a player's or dealer's hand of cards."""
    def __init__(self, cards: list[str]):
        self.cards = cards
        self.value, self.is_soft = self._calculate_value()

    def _calculate_value(self) -> tuple[int, bool]:
        total, ace_count = 0, 0
        for card in self.cards:
            rank = card[:-1] if card.startswith("10") else card[0]
            if rank == 'A':
                ace_count += 1
                total += 11
            elif rank in ('K', 'Q', 'J'):
                total += 10
            elif rank == '10':
                total += 10
            else:
                total += int(rank)
        
        is_soft = ace_count > 0
        while total > 21 and ace_count > 0:
            total -= 10
            ace_count -= 1
        return total, is_soft

    @property
    def is_pair(self) -> bool:
        if len(self.cards) != 2: return False
        r1 = self.cards[0][:-1] if self.cards[0].startswith("10") else self.cards[0][0]
        r2 = self.cards[1][:-1] if self.cards[1].startswith("10") else self.cards[1][0]
        return r1 == r2

def recommend_action(
    player_hand_cards: list[str],
    dealer_upcard_str: str,
    true_count: float,
    shoe: 'Shoe'
) -> str:
    if not player_hand_cards or not dealer_upcard_str:
        return "Awaiting full input"

    cfg = OMEGA_II_CONFIG
    player_hand = Hand(player_hand_cards)
    dealer_hand = Hand([dealer_upcard_str])
    
    # Bayesian Prediction
    next_cards_info = ""
    try:
        next_probs = bayesian_predictor.next_card_probabilities(shoe.get_remaining_cards(), top_n=3)
        if next_probs:
            card_preds = [f"{c} ({p:.1%})" for c, p in next_probs]
            next_cards_info = f"<br><i>Next Card Prediction: {', '.join(card_preds)}</i>"
    except Exception:
        pass

    # Basic Strategy and Index Plays
    action = _get_basic_strategy_action(player_hand, dealer_hand, cfg, true_count)

    return f'<b>{action}</b>{next_cards_info}'

def _get_basic_strategy_action(player_hand: Hand, dealer_hand: Hand, cfg: dict, tc: float) -> str:
    # Insurance
    if len(player_hand.cards) == 2 and dealer_hand.cards[0][0] == 'A':
        return "Take Insurance" if tc >= cfg["insurance_threshold"] else "Decline Insurance"

    # Splitting
    if player_hand.is_pair:
        pair_rank = player_hand.cards[0][0]
        if pair_rank in ('A', '8'): return "Split"
        # Additional split logic could be added here

    # Soft Totals
    if player_hand.is_soft:
        if player_hand.value >= 19: return "Stand"
        if player_hand.value == 18 and dealer_hand.value >= 9: return "Hit"
        if player_hand.value == 18 and dealer_hand.value in [2, 7, 8]: return "Stand"
        if len(player_hand.cards) == 2: # Double Down
             if player_hand.value == 18 and dealer_hand.value in [3,4,5,6]: return "Double"
             if player_hand.value == 17 and dealer_hand.value in [3,4,5,6]: return "Double"
        return "Hit"

    # Hard Totals (with Index Plays)
    key = f"{player_hand.value}v{dealer_hand.value}"
    if dealer_hand.value == 10: key = f"{player_hand.value}vT" # Consolidate 10-value cards
    if key in cfg['index_plays'] and tc >= cfg['index_plays'][key]:
        return "Stand (Index)"
    
    if len(player_hand.cards) == 2: # Double Down
        if player_hand.value == 11: return "Double"
        if player_hand.value == 10 and dealer_hand.value <= 9: return "Double"
        if player_hand.value == 9 and dealer_hand.value in [3,4,5,6]: return "Double"

    if player_hand.value >= 17: return "Stand"
    if player_hand.value >= 13 and dealer_hand.value <= 6: return "Stand"
    if player_hand.value == 12 and dealer_hand.value in [4,5,6]: return "Stand"
    
    return "Hit"