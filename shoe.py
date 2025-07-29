# shoe.py
from __future__ import annotations
import random

class Shoe:
    def __init__(self, decks: int = 8):
        if not isinstance(decks, int) or decks <= 0:
            raise ValueError("Number of decks must be a positive integer.")
        self.decks: int = decks
        self.cards: dict[str, int] = {}
        self.total_cards: int = 0
        self.initial_card_count: int = 52 * self.decks
        self.reset_shoe()

    def __repr__(self) -> str:
        return (f"<Shoe(decks={self.decks}, "
                f"cards_remaining={self.total_cards}/{self.initial_card_count})>")

    def reset_shoe(self) -> None:
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        suits = ["S", "H", "D", "C"]
        self.cards = {f"{r}{s}": self.decks for r in ranks for s in suits}
        self.total_cards = self.initial_card_count

    def remove_card(self, card_code: str) -> None:
        if self.cards.get(card_code, 0) > 0:
            self.cards[card_code] -= 1
            self.total_cards -= 1
        else:
            raise ValueError(f"Card '{card_code}' is not available in the shoe.")

    def restore_card(self, card_code: str) -> None:
        if card_code in self.cards:
            if self.cards[card_code] < self.decks:
                self.cards[card_code] += 1
                self.total_cards += 1
            else:
                raise ValueError(f"Cannot restore '{card_code}'; max count reached.")
        else:
            raise KeyError(f"Invalid card code: '{card_code}'")

    def draw_random_card(self) -> str:
        if self.total_cards == 0:
            raise ValueError("Cannot draw from an empty shoe.")
        available_cards = [card for card, count in self.cards.items() if count > 0]
        if not available_cards:
             raise ValueError("Shoe state error: total_cards > 0 but no cards found.")
        card_to_draw = random.choice(available_cards)
        self.remove_card(card_to_draw)
        return card_to_draw

    def decks_remaining(self) -> float:
        if self.initial_card_count == 0:
            return 0.0
        return self.total_cards / 52.0

    def get_penetration(self) -> float:
        if self.initial_card_count == 0:
            return 0.0
        return (self.initial_card_count - self.total_cards) / self.initial_card_count

    def get_remaining_cards(self) -> dict[str, int]:
        return self.cards.copy()