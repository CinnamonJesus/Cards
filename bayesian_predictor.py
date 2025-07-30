"""
Performs Bayesian-style predictions about game outcomes through Monte Carlo simulation,
such as predicting the dealer's final hand total. This module also calculates
immediate, composition-dependent EV for certain side bets.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from random import choices
import numpy as np
from numba import njit, prange
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shoe import Shoe

# Import from the shared utility module
from numba_utils import (
    evaluate_21plus3_numba, 
    evaluate_perfect_pairs_numba, 
    evaluate_hot3_numba,
    draw_card as draw_card_numba,
)

def get_card_value(card: str) -> tuple[int, bool]:
    """Gets the numerical value of a card, returning value and if it's an Ace."""
    rank = card[:-1] if card.startswith("10") else card[0]
    if rank == 'A': return 11, True
    if rank in ('K', 'Q', 'J', '10'): return 10, False
    return int(rank), False

def next_card_probabilities(shoe_cards: dict[str, int], top_n: int = 5) -> list[tuple[str, float]]:
    """Calculates the probability of the next card dealt for the top N most likely cards."""
    total_cards = sum(shoe_cards.values())
    if total_cards == 0: return []

    probs = {card: count / total_cards for card, count in shoe_cards.items()}
    sorted_cards = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    return sorted_cards[:top_n]

def next_rank_probabilities(shoe_cards: dict[str, int]) -> dict[str, float]:
    """Calculates the probability distribution for the rank of the next card."""
    total_cards = sum(shoe_cards.values())
    if total_cards == 0: return {}
    
    rank_counts = Counter()
    for card, count in shoe_cards.items():
        rank = card[:-1] if card.startswith("10") else card[0]
        rank_counts[rank] += count
        
    return {rank: count / total_cards for rank, count in rank_counts.items()}

def next_suit_probabilities(shoe_cards: dict[str, int]) -> dict[str, float]:
    """Calculates the probability distribution for the suit of the next card."""
    total_cards = sum(shoe_cards.values())
    if total_cards == 0: return {}
        
    suit_counts = Counter()
    for card, count in shoe_cards.items():
        suit = card[-1]
        suit_counts[suit] += count
        
    return {suit: count / total_cards for suit, count in suit_counts.items()}

def rank_distribution_probabilities(shoe_cards: dict[str, int]) -> dict[str, float]:
    """Calculates the probability distribution for categorized ranks."""
    rank_probs = next_rank_probabilities(shoe_cards)
    if not rank_probs: return {}

    dist = {
        "Low (2-6)": 0.0,
        "Neutral (7-9)": 0.0,
        "High (10-K)": 0.0,
        "Ace": 0.0,
    }
    
    low_ranks = {'2', '3', '4', '5', '6'}
    neutral_ranks = {'7', '8', '9'}
    high_ranks = {'10', 'J', 'Q', 'K'}

    for rank, prob in rank_probs.items():
        if rank in low_ranks: dist["Low (2-6)"] += prob
        elif rank in neutral_ranks: dist["Neutral (7-9)"] += prob
        elif rank in high_ranks: dist["High (10-K)"] += prob
        elif rank == 'A': dist["Ace"] += prob
            
    return dist

def dealer_total_probabilities(
    dealer_upcard: str,
    shoe_cards: dict[str, int],
    simulations: int = 10000
) -> dict[int | str, float]:
    """
    Simulates the dealer's hand to find the probability distribution of final totals.
    """
    if dealer_upcard not in shoe_cards or shoe_cards[dealer_upcard] == 0:
        return {}

    sim_shoe = shoe_cards.copy()
    sim_shoe[dealer_upcard] -= 1
    final_totals = defaultdict(int)

    cards_population = list(sim_shoe.keys())
    counts_weights = list(sim_shoe.values())

    for _ in range(simulations):
        temp_shoe_counts = sim_shoe.copy()
        dealer_hand = [dealer_upcard]
        
        while True:
            total, aces = 0, 0
            for card in dealer_hand:
                val, is_ace = get_card_value(card)
                total += val
                if is_ace: aces += 1
            
            while total > 21 and aces > 0:
                total -= 10
                aces -= 1
            
            if total >= 17:
                final_totals["Bust" if total > 21 else total] += 1
                break

            total_remaining = sum(temp_shoe_counts.values())
            if total_remaining == 0:
                final_totals[total] += 1
                break
            
            drawn_card = choices(list(temp_shoe_counts.keys()), weights=list(temp_shoe_counts.values()), k=1)[0]
            dealer_hand.append(drawn_card)
            temp_shoe_counts[drawn_card] -= 1

    return {total: count / simulations for total, count in final_totals.items()}

@njit(parallel=True, cache=True)
def _run_side_bet_sim_chunk(shoe_counts: np.ndarray, rounds: int) -> np.ndarray:
    """Numba-jitted worker to simulate just the first 3 cards for side bet EV."""
    np.random.seed(np.random.randint(0, 1_000_000))
    results = np.zeros((rounds, 3), dtype=np.float64)

    for i in prange(rounds):
        temp_shoe = shoe_counts.copy()
        
        p1_idx = draw_card_numba(temp_shoe)
        p2_idx = draw_card_numba(temp_shoe)
        d1_idx = draw_card_numba(temp_shoe)

        if -1 in (p1_idx, p2_idx, d1_idx): continue

        p_ranks = np.array([p1_idx % 13, p2_idx % 13])
        p_suits = np.array([p1_idx // 13, p2_idx // 13])
        d_rank, d_suit = d1_idx % 13, d1_idx // 13

        results[i, 0] = evaluate_perfect_pairs_numba(p_ranks, p_suits)
        results[i, 1] = evaluate_21plus3_numba(p_ranks, d_rank, p_suits, d_suit)
        results[i, 2] = evaluate_hot3_numba(p_ranks, d_rank)
        
    return results

def run_side_bet_simulation(shoe: 'Shoe', num_rounds: int = 50000) -> dict[str, float]:
    """
    Calculates the immediate, composition-dependent EV for side bets using a
    fast, Numba-powered simulation.
    """
    shoe_dict = shoe.get_remaining_cards()
    
    shoe_counts = np.zeros(52, dtype=np.int32)
    ranks = {'A':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9, 'J':10, 'Q':11, 'K':12}
    suits = {'S':0, 'H':1, 'D':2, 'C':3}
    for card_str, count in shoe_dict.items():
        rank_str = card_str[:-1] if card_str.startswith("10") else card_str[0]
        suit_str = card_str[-1]
        idx = suits[suit_str] * 13 + ranks[rank_str]
        shoe_counts[idx] = count

    all_results = _run_side_bet_sim_chunk(shoe_counts, num_rounds)

    evs = {
        "perfect_pairs_ev": np.mean(all_results[:, 0]),
        "21+3_ev": np.mean(all_results[:, 1]),
        "hot3_ev": np.mean(all_results[:, 2]),
    }
    return evs