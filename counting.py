"""
Implements various card counting systems for Blackjack.
This module provides a base class for card counting systems and specific
implementations such as Hi-Lo, Zen Count, Wong Halves, and Omega II. This
structure allows for the easy addition of new counting strategies.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import math # Added for floor rounding

def get_rank(card_code: str) -> str:
    """Extracts the rank from a card code string."""
    return "10" if card_code.startswith("10") else card_code[0]

class CountingSystem(ABC):
    """Abstract base class for a card counting system."""
    values: dict[str, float]

    def __init__(self) -> None:
        self.running_count: float = 0.0

    def update(self, card_code: str) -> None:
        rank = get_rank(card_code)
        self.running_count += self.values.get(rank, 0.0)

    def undo(self, card_code: str) -> None:
        rank = get_rank(card_code)
        self.running_count -= self.values.get(rank, 0.0)

    def reset(self) -> None:
        self.running_count = 0.0

    def true_count(self, decks_remaining: float) -> float:
        """Calculates the true count using floor rounding for a more conservative and standard calculation."""
        if decks_remaining <= 0:
            return 0.0
        return math.floor(self.running_count / decks_remaining)

class ZenCount(CountingSystem):
    values: dict[str, float] = {
        "2": 1, "3": 1, "4": 2, "5": 2, "6": 2, "7": 1,
        "8": 0, "9": 0, "10": -2, "J": -2, "Q": -2, "K": -2, "A": -1
    }
    
class HiLoCount(CountingSystem):
    values: dict[str, float] = {
        "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 0,
        "8": 0, "9": 0, "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1
    }

class WongHalves(CountingSystem):
    values: dict[str, float] = {
        "2": 0.5, "3": 1.0, "4": 1.0, "5": 1.5, "6": 1.0, "7": 0.5,
        "8": 0.0, "9": -0.5, "10": -1.0, "J": -1.0, "Q": -1.0, "K": -1.0, "A": -1.0
    }

class Omega2Count(CountingSystem):
    """Omega II system with corrected value for rank '3'."""
    values: dict[str, float] = {
        "2": 1, "3": 1, "4": 2, "5": 2, "6": 2, "7": 1,
        "8": 0, "9": -1, "10": -2, "J": -2, "Q": -2, "K": -2, "A": 0
    }