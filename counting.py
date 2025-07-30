"""
Implements various card counting systems for Blackjack.

This module provides a base class for card counting systems and specific
implementations such as Hi-Lo, Zen Count, Wong Halves, and Omega II. This
structure allows for the easy addition of new counting strategies.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import math

def get_rank(card_code: str) -> str:
    """Extracts the rank from a card code string."""
    return "10" if card_code.startswith("10") else card_code[0]

class CountingSystem(ABC):
    """Abstract base class for a card counting system."""
    values: dict[str, float]

    def __init__(self) -> None:
        self.running_count: float = 0.0

    def update(self, card_code: str) -> None:
        """Updates the running count based on a new card being dealt."""
        rank = get_rank(card_code)
        self.running_count += self.values.get(rank, 0.0)

    def undo(self, card_code: str) -> None:
        """Reverts the running count for a card that was removed."""
        rank = get_rank(card_code)
        self.running_count -= self.values.get(rank, 0.0)

    def reset(self) -> None:
        """Resets the running count to zero."""
        self.running_count = 0.0

    def true_count(self, decks_remaining: float) -> float:
        """
        Calculates the true count using floor rounding for a more conservative and
        standard calculation for betting purposes.
        """
        if decks_remaining <= 0:
            return 0.0
        # Floor division provides a more conservative (and standard) true count
        return math.floor(self.running_count / decks_remaining)

class ZenCount(CountingSystem):
    """Implements the Zen Count system, a balanced, level-2 counting system."""
    values: dict[str, float] = {
        "2": 1, "3": 1, "4": 2, "5": 2, "6": 2, "7": 1,
        "8": 0, "9": 0, "10": -2, "J": -2, "Q": -2, "K": -2, "A": -1
    }
    
class HiLoCount(CountingSystem):
    """Implements the Hi-Lo system, the most common balanced, level-1 count."""
    values: dict[str, float] = {
        "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 0,
        "8": 0, "9": 0, "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1
    }

class WongHalves(CountingSystem):
    """Implements the Wong Halves system, a balanced, level-3 count with fractional values."""
    values: dict[str, float] = {
        "2": 0.5, "3": 1.0, "4": 1.0, "5": 1.5, "6": 1.0, "7": 0.5,
        "8": 0.0, "9": -0.5, "10": -1.0, "J": -1.0, "Q": -1.0, "K": -1.0, "A": -1.0
    }

class Omega2Count(CountingSystem):
    """Implements the Omega II system. Aces are neutral (0) and often side-counted."""
    values: dict[str, float] = {
        "2": 1, "3": 2, "4": 2, "5": 2, "6": 2, "7": 1, # Corrected '3' from 1 to 2
        "8": 0, "9": -1, "10": -2, "J": -2, "Q": -2, "K": -2, "A": 0
    }