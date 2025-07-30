"""
A utility module for shared, Numba-jitted functions used across the
Blackjack simulator for high-performance calculations.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def get_card_value_numba(rank: int) -> int:
    """Gets the blackjack value from a numerical rank (0=A, 1=2, ..., 12=K)."""
    if rank == 0: return 11  # Ace
    if rank >= 9: return 10  # 10, J, Q, K
    return rank + 1

@njit(cache=True)
def get_hand_total(hand_ranks: np.ndarray) -> tuple[int, bool]:
    """
    Calculates hand value from numerical ranks.
    Returns (total, is_soft). is_soft is True if an Ace is counted as 11.
    """
    total = 0
    aces = 0
    for rank in hand_ranks:
        if rank == 0:  # Ace
            aces += 1
        total += get_card_value_numba(rank)

    is_soft = False
    if aces > 0:
        if total > 21:
            effective_aces = aces
            while total > 21 and effective_aces > 0:
                total -= 10
                effective_aces -= 1
        if total <= 21 and aces > 0:
            is_soft = True

    return total, is_soft

@njit(cache=True)
def draw_card(temp_shoe: np.ndarray) -> int:
    """
    Draws a single card index from the shoe array, returning -1 if empty.
    This function modifies the temp_shoe array in place.
    """
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
    ranks = np.array([p_ranks[0], p_ranks[1], d_rank])
    suits = np.array([p_suits[0], p_suits[1], d_suit])
    
    is_flush = (suits[0] == suits[1] == suits[2])
    # FIX: Removed return_counts=True for Numba compatibility
    unique_ranks = np.unique(ranks)
    is_three_kind = (len(unique_ranks) == 1)

    if is_three_kind and is_flush: return 100.0
    
    sorted_ranks = np.sort(unique_ranks)
    is_straight = (len(sorted_ranks) == 3 and (sorted_ranks[2] - sorted_ranks[0] == 2)) or \
                  (np.all(np.isin(np.array([0, 11, 12]), sorted_ranks)))
    
    if is_straight and is_flush: return 40.0
    if is_three_kind: return 30.0
    if is_straight: return 10.0
    if is_flush: return 5.0
    
    return -1.0

@njit(cache=True)
def evaluate_hot3_numba(p_ranks: np.ndarray, d_rank: int) -> float:
    """Numba-compatible evaluation of Hot 3 side bet (simplified, no suit check)."""
    three_card_hand = np.array([p_ranks[0], p_ranks[1], d_rank])
    
    if three_card_hand[0] == 6 and three_card_hand[1] == 6 and three_card_hand[2] == 6: 
        return 100.0
        
    total, _ = get_hand_total(three_card_hand)
    
    if total == 21: return 10.0
    if total == 20: return 2.0
    if total == 19: return 1.0
    
    return -1.0