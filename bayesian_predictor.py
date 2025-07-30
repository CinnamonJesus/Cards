"""
Performs Bayesian-style predictions about game outcomes through Monte Carlo simulation,
such as predicting the dealer's final hand total and immediate side bet EV.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from random import choices
import numpy as np
from numba import njit, prange

# Import from the new, shared utility module
from numba_utils import (
    get_hand_total, evaluate_21plus3_numba, 
    evaluate_perfect_pairs_numba, evaluate_hot3_numba
)

def get_card_value(card: str) -> tuple[int, bool]:
    """Gets the numerical value of a card, returning value and if it's an Ace."""
    rank = card[:-1] if card.startswith("10") else card[0]
    if rank == 'A':
        return 11, True
    if rank in ('K', 'Q', 'J', '10'):
        return 10, False
    return int(rank), False

def next_card_probabilities(shoe_cards: dict[str, int], top_n: int = 5) -> list[tuple[str, float]]:
    """
    Calculates the probability of the next card dealt for the top N most likely cards.
    """
    total_cards = sum(shoe_cards.values())
    if total_cards == 0:
        return []

    probs = {card: count / total_cards for card, count in shoe_cards.items()}
    sorted_cards = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    return sorted_cards[:top_n]

def next_rank_probabilities(shoe_cards: dict[str, int]) -> list[tuple[str, float]]:
    """Calculates the probability distribution for the rank of the next card."""
    total_cards = sum(shoe_cards.values())
    if total_cards == 0:
        return []
    
    rank_counts = Counter()
    for card, count in shoe_cards.items():
        rank = card[:-1] if card.startswith("10") else card[0]
        rank_counts[rank] += count
        
    probs = {rank: count / total_cards for rank, count in rank_counts.items()}
    return sorted(probs.items(), key=lambda item: item[1], reverse=True)

def next_suit_probabilities(shoe_cards: dict[str, int]) -> list[tuple[str, float]]:
    """Calculates the probability distribution for the suit of the next card."""
    total_cards = sum(shoe_cards.values())
    if total_cards == 0:
        return []
        
    suit_counts = Counter()
    for card, count in shoe_cards.items():
        suit = card[-1]
        suit_counts[suit] += count
        
    probs = {suit: count / total_cards for suit, count in suit_counts.items()}
    return sorted(probs.items(), key=lambda item: item[1], reverse=True)

def dealer_total_probabilities(
    dealer_upcard: str,
    shoe_cards: dict[str, int],
    simulations: int = 10000
) -> dict[int | str, float]:
    """
    Simulates the dealer's hand to find the probability distribution of final totals.
    Dealer stands on all 17s.
    """
    if dealer_upcard not in shoe_cards or shoe_cards[dealer_upcard] == 0:
        return {}

    sim_shoe = shoe_cards.copy()
    sim_shoe[dealer_upcard] -= 1
    final_totals = defaultdict(int)

    for _ in range(simulations):
        temp_shoe_counts = sim_shoe.copy()
        cards, counts = list(temp_shoe_counts.keys()), list(temp_shoe_counts.values())
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
            
            drawn_card = choices(cards, weights=counts, k=1)[0]
            dealer_hand.append(drawn_card)
            temp_shoe_counts[drawn_card] -= 1
            cards, counts = list(temp_shoe_counts.keys()), list(temp_shoe_counts.values())

    return {total: count / simulations for total, count in final_totals.items()}

@njit(parallel=True, cache=True)
def _run_side_bet_sim_chunk(shoe_counts: np.ndarray, rounds: int) -> np.ndarray:
    """Numba-jitted worker to simulate just the first 3 cards for side bet EV."""
    np.random.seed(np.random.randint(0, 1_000_000))
    results = np.zeros((rounds, 3), dtype=np.float64)

    for i in prange(rounds):
        temp_shoe = shoe_counts.copy()
        
        drawn_cards_indices = np.zeros(3, dtype=np.int32)
        for k in range(3):
            total_cards = np.sum(temp_shoe)
            if total_cards == 0:
                drawn_cards_indices[k] = -1
                continue
            
            rand_val = np.random.random() * total_cards
            cumulative_sum = 0.0
            choice = 51
            for j in range(52):
                cumulative_sum += temp_shoe[j]
                if rand_val < cumulative_sum:
                    choice = j
                    break
            
            drawn_cards_indices[k] = choice
            temp_shoe[choice] -= 1

        if -1 in drawn_cards_indices: continue

        p1_idx, p2_idx, d1_idx = drawn_cards_indices
        p_ranks = np.array([p1_idx % 13, p2_idx % 13])
        p_suits = np.array([p1_idx // 13, p2_idx // 13])
        d_rank, d_suit = d1_idx % 13, d1_idx // 13

        results[i, 0] = evaluate_perfect_pairs_numba(p_ranks, p_suits)
        results[i, 1] = evaluate_21plus3_numba(p_ranks, d_rank, p_suits, d_suit)
        results[i, 2] = evaluate_hot3_numba(p_ranks, d_rank, p_suits, d_suit)
        
    return results

def run_side_bet_simulation(shoe: 'Shoe', num_rounds: int = 50000) -> dict[str, float]:
    """
    Calculates the immediate, composition-dependent EV for side bets.
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
