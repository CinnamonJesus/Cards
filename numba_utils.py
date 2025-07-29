# numba_utils.py
"""
A utility module for shared, Numba-jitted functions used across the
Blackjack simulator for high-performance calculations.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def get_card_value_numba(rank: int) -> int:
    """Gets the blackjack value from a numerical rank."""
    if rank == 0: return 11  # Ace
    if rank >= 9: return 10  # 10, J, Q, K
    return rank + 1

@njit(cache=True)
def get_hand_total(hand: np.ndarray) -> tuple[int, int]:
    """Calculates hand value from numerical ranks, returns (total, num_aces)."""
    total = 0
    aces = 0
    for rank in hand:
        if rank == 0:
            aces += 1
        total += get_card_value_numba(rank)

    temp_total = total
    temp_aces = aces
    while temp_total > 21 and temp_aces > 0:
        temp_total -= 10
        temp_aces -= 1
    return temp_total, aces

@njit(cache=True)
def draw_card(temp_shoe: np.ndarray) -> int:
    """Draws a single card index from the shoe, returning -1 if empty."""
    total_cards = np.sum(temp_shoe)
    if total_cards == 0:
        return -1

    rand_val = np.random.random() * total_cards
    cumulative_sum = 0.0
    for j in range(52):
        cumulative_sum += temp_shoe[j]
        if rand_val < cumulative_sum:
            temp_shoe[j] -= 1
            return j

    # Failsafe for floating point issues
    for j in range(51, -1, -1):
        if temp_shoe[j] > 0:
            temp_shoe[j] -= 1
            return j
    return -1

@njit(cache=True)
def evaluate_perfect_pairs_numba(p_ranks: np.ndarray, p_suits: np.ndarray) -> float:
    """Numba-compatible evaluation of Perfect Pairs side bet."""
    if p_ranks[0] != p_ranks[1]: return -1.0
    if p_suits[0] == p_suits[1]: return 25.0
    is_c1_red = (p_suits[0] == 1 or p_suits[0] == 2)
    is_c2_red = (p_suits[1] == 1 or p_suits[1] == 2)
    return 12.0 if is_c1_red == is_c2_red else 6.0

@njit(cache=True)
def evaluate_21plus3_numba(p_ranks: np.ndarray, d_rank: int, p_suits: np.ndarray, d_suit: int) -> float:
    """Numba-compatible evaluation of 21+3 side bet."""
    ranks = np.array([p_ranks[0], p_ranks[1], d_rank]); suits = np.array([p_suits[0], p_suits[1], d_suit])
    is_flush = (suits[0] == suits[1] == suits[2]); unique_ranks = np.unique(ranks)
    is_three_kind = len(unique_ranks) == 1
    if is_three_kind and is_flush: return 100.0
    sorted_ranks = np.sort(ranks)
    is_straight = (sorted_ranks[2] - sorted_ranks[0] == 2 and len(unique_ranks) == 3) or \
                  (sorted_ranks[0] == 0 and sorted_ranks[1] == 11 and sorted_ranks[2] == 12) # A-Q-K
    if is_straight and is_flush: return 40.0
    if is_three_kind: return 30.0
    if is_straight: return 10.0
    if is_flush: return 5.0
    return -1.0

@njit(cache=True)
def evaluate_hot3_numba(p_ranks: np.ndarray, d_rank: int, p_suits: np.ndarray, d_suit: int) -> float:
    """Numba-compatible evaluation of Hot 3 side bet."""
    ranks = np.array([p_ranks[0], p_ranks[1], d_rank]); suits = np.array([p_suits[0], p_suits[1], d_suit])
    if ranks[0] == 6 and ranks[1] == 6 and ranks[2] == 6: return 100.0 # 7-7-7
    total, _ = get_hand_total(ranks)
    is_suited = (suits[0] == suits[1] == suits[2])
    if total == 21: return 20.0 if is_suited else 10.0
    if total == 20: return 4.0 if is_suited else 2.0
    if total == 19: return 1.0
    return -1.0