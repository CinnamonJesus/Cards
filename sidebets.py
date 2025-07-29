"""
Defines payout structures and utility functions for various Blackjack side bets.
"""
from __future__ import annotations

PAYOUT_21PLUS3 = {"flush": 5, "straight": 10, "three_kind": 30, "straight_flush": 40, "suited_trips": 100}
PAYOUT_PERFECT_PAIRS = {"mixed_pair": 6, "colored_pair": 12, "perfect_pair": 25}
PAYOUT_BUST = {3: 1, 4: 2, 5: 15, 6: 50, 7: 100, 8: 250}
PAYOUT_HOT3 = {"19": 1, "20": 2, "20_suited": 4, "21": 10, "21_suited": 20, "777": 100}

def get_card_details(card: str) -> tuple[str, str]:
    """Extracts the rank and suit from a card code."""
    rank = "10" if card.startswith("10") else card[0]
    suit = card[-1]
    return rank, suit