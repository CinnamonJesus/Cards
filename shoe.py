from __future__ import annotations
import random

class Shoe:
    
    def __init__(self, decks: int = 8):
        """
        Initializes the shoe with a specified number of decks.

        Args:
            decks (int): The number of 52-card decks to include in the shoe.
        """
        if not isinstance(decks, int) or decks <= 0:
            raise ValueError("Number of decks must be a positive integer.")
            
        self.decks: int = decks
        self.cards: dict[str, int] = {}
        self.total_cards: int = 0
        self.initial_card_count: int = 52 * self.decks
        self.reset_shoe()

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the Shoe object."""
        return (
            f"<Shoe(decks={self.decks}, "
            f"cards_remaining={self.total_cards}/{self.initial_card_count})>"
        )

    def reset_shoe(self) -> None:
        """
        Resets the shoe to its original state with all cards present.
        """
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        suits = ["S", "H", "D", "C"]
        self.cards = {f"{r}{s}": self.decks for r in ranks for s in suits}
        self.total_cards = self.initial_card_count

    def remove_card(self, card_code: str) -> None:
        """
        Removes a single card from the shoe.

        Args:
            card_code (str): The string identifier for the card (e.g., 'AS', 'KH', '7D').

        Raises:
            ValueError: If the specified card is not available in the shoe.
        """
        if self.cards.get(card_code, 0) > 0:
            self.cards[card_code] -= 1
            self.total_cards -= 1
        else:
            raise ValueError(f"Card '{card_code}' is not available to remove from the shoe.")

    def restore_card(self, card_code: str) -> None:
        """
        Restores a single card to the shoe, for correcting input errors.

        Args:
            card_code (str): The string identifier for the card.
        
        Raises:
            ValueError: If trying to restore a card would exceed the initial count for that card.
        """
        if card_code in self.cards:
            if self.cards[card_code] < self.decks:
                self.cards[card_code] += 1
                self.total_cards += 1
            else:
                raise ValueError(f"Cannot restore '{card_code}'; shoe already contains the maximum number of this card.")
        else:
            raise KeyError(f"Attempted to restore an invalid card code: '{card_code}'")

    def draw_random_card(self) -> str:
        """
        Draws a single random card from the available cards in the shoe.
        
        Returns:
            str: The card code of the drawn card.
            
        Raises:
            ValueError: If the shoe is empty.
        """
        if self.total_cards == 0:
            raise ValueError("Cannot draw from an empty shoe.")
        
        # Create a list of cards weighted by their count
        available_cards = [card for card, count in self.cards.items() for _ in range(count)]
        if not available_cards:
             raise ValueError("Shoe state error: total_cards > 0 but no cards found.")
             
        card_to_draw = random.choice(available_cards)
        self.remove_card(card_to_draw) # This also decrements the counts
        return card_to_draw

    def decks_remaining(self) -> float:
        """
        Calculates the number of decks remaining in the shoe.

        Returns:
            float: The approximate number of decks left, used for true count calculation.
        """
        if self.initial_card_count == 0:
            return 0.0
        return self.total_cards / 52.0

    def get_penetration(self) -> float:
        """
        Calculates the shoe penetration as a fraction.

        Returns:
            float: A value between 0.0 and 1.0 representing the fraction of
                   cards that have been dealt from the shoe.
        """
        if self.initial_card_count == 0:
            return 0.0
        return (self.initial_card_count - self.total_cards) / self.initial_card_count
        
    def get_remaining_cards(self) -> dict[str, int]:
        """
        Returns a copy of the dictionary of cards remaining in the shoe.
        
        Returns:
            dict[str, int]: A dictionary where keys are card codes and values are their counts.
        """
        return self.cards.copy()
