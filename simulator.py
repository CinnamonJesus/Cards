"""
Performs high-speed, parallel Monte Carlo simulations of Blackjack hands
to calculate the Expected Value (EV) of the main bet and various side bets.
This version has been stabilized and now includes logic for splitting pairs.
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor

from numba_utils import (
    get_hand_total,
    draw_card,
    evaluate_21plus3_numba,
    evaluate_perfect_pairs_numba,
    evaluate_hot3_numba,
)

@njit(cache=True)
def _resolve_outcome(player_total: int, dealer_total: int, bet_multiplier: float) -> float:
    """Compares final totals and returns the bet outcome."""
    if player_total > 21: return -bet_multiplier
    if dealer_total > 21 or player_total > dealer_total: return bet_multiplier
    if player_total < dealer_total: return -bet_multiplier
    return 0.0

@njit(cache=True)
def _play_single_hand(hand: np.ndarray, temp_shoe: np.ndarray, dealer_up_rank: int) -> tuple[int, float]:
    """
    Plays a single player hand (post-split or initial) according to basic strategy.
    Returns the final hand total and the bet multiplier.
    """
    if len(hand) == 2:
        hand_val, is_soft = get_hand_total(hand)
        do_double = False
        if is_soft:
            if hand_val in (17, 18) and dealer_up_rank in (2,3,4,5): do_double = True
        else:
            if hand_val == 11: do_double = True
            if hand_val == 10 and dealer_up_rank <= 8: do_double = True
            if hand_val == 9 and dealer_up_rank in (2,3,4,5): do_double = True
        
        if do_double:
            card_idx = draw_card(temp_shoe)
            if card_idx != -1: hand = np.append(hand, card_idx % 13)
            final_total, _ = get_hand_total(hand)
            return final_total, 2.0

    while True:
        player_total, is_soft = get_hand_total(hand)
        if player_total >= 21: return player_total, 1.0

        stand = False
        if is_soft:
            if player_total >= 19: stand = True
            if player_total == 18 and dealer_up_rank <= 7: stand = True
        else:
            if player_total >= 17: stand = True
            if player_total >= 13 and dealer_up_rank <= 5: stand = True
            if player_total == 12 and dealer_up_rank in (3,4,5): stand = True
        
        if stand: return player_total, 1.0

        card_idx = draw_card(temp_shoe)
        if card_idx == -1: return player_total, 1.0
        hand = np.append(hand, card_idx % 13)

@njit(parallel=True, cache=True)
def simulate_chunk(shoe_counts: np.ndarray, rounds: int) -> np.ndarray:
    """
    Runs a chunk of simulations in parallel, now with logic to handle one split.
    """
    np.random.seed(np.random.randint(0, 1_000_000))
    results = np.zeros((rounds, 5), dtype=np.float64)

    for i in prange(rounds):
        temp_shoe = shoe_counts.copy()
        
        p1_idx, p2_idx, d1_idx = draw_card(temp_shoe), draw_card(temp_shoe), draw_card(temp_shoe)
        if -1 in (p1_idx, p2_idx, d1_idx): continue

        p_ranks = np.array([p1_idx % 13, p2_idx % 13])
        p_suits = np.array([p1_idx // 13, p2_idx // 13])
        d_rank, d_suit = d1_idx % 13, d1_idx // 13

        results[i, 2] = evaluate_21plus3_numba(p_ranks, d_rank, p_suits, d_suit)
        results[i, 3] = evaluate_perfect_pairs_numba(p_ranks, p_suits)
        results[i, 4] = evaluate_hot3_numba(p_ranks, d_rank)

        player_total, _ = get_hand_total(p_ranks)
        d_hole_idx = draw_card(temp_shoe)
        if d_hole_idx == -1: continue
        
        dealer_hand = np.array([d_rank, d_hole_idx % 13])
        dealer_total, _ = get_hand_total(dealer_hand)

        if player_total == 21:
            results[i, 0] = 1.5 if dealer_total != 21 else 0.0
            continue
        if dealer_total == 21:
            results[i, 0] = -1.0
            continue
        
        # --- Splitting Logic ---
        is_pair = p_ranks[0] == p_ranks[1]
        pair_rank = p_ranks[0]
        should_split = False
        if is_pair:
            dealer_up_val = get_hand_total(np.array([d_rank]))[0]
            if pair_rank in (0, 7): should_split = True # Aces and 8s
            if pair_rank == 8 and dealer_up_val not in (7, 10, 11): should_split = True # 9s
            # Simplified split logic for simulation speed
            if pair_rank in (1, 2, 6) and dealer_up_val <= 7: should_split = True # 2s, 3s, 7s
            if pair_rank == 5 and dealer_up_val <= 6: should_split = True # 6s

        if should_split:
            # Play two separate hands
            hand1_card2_idx = draw_card(temp_shoe)
            hand2_card2_idx = draw_card(temp_shoe)
            if -1 in (hand1_card2_idx, hand2_card2_idx): continue

            hand1 = np.array([pair_rank, hand1_card2_idx % 13])
            hand2 = np.array([pair_rank, hand2_card2_idx % 13])
            
            p1_final, mult1 = _play_single_hand(hand1, temp_shoe, d_rank)
            p2_final, mult2 = _play_single_hand(hand2, temp_shoe, d_rank)
            
            # Dealer plays out their hand once
            while get_hand_total(dealer_hand)[0] < 17:
                card_idx = draw_card(temp_shoe)
                if card_idx == -1: break
                dealer_hand = np.append(dealer_hand, card_idx % 13)
            dealer_final_val = get_hand_total(dealer_hand)[0]

            outcome1 = _resolve_outcome(p1_final, dealer_final_val, mult1)
            outcome2 = _resolve_outcome(p2_final, dealer_final_val, mult2)
            results[i, 0] = outcome1 + outcome2
        else:
            # --- Standard Hand Logic ---
            player_final_total, bet_multiplier = _play_single_hand(p_ranks, temp_shoe, d_rank)
            while get_hand_total(dealer_hand)[0] < 17:
                card_idx = draw_card(temp_shoe)
                if card_idx == -1: break
                dealer_hand = np.append(dealer_hand, card_idx % 13)
            dealer_final_val = get_hand_total(dealer_hand)[0]
            results[i, 0] = _resolve_outcome(player_final_total, dealer_final_val, bet_multiplier)

        if get_hand_total(dealer_hand)[0] > 21:
            # Bust bet logic remains the same
            num_cards = len(dealer_hand)
            if num_cards == 3: results[i, 1] = 1.0
            elif num_cards == 4: results[i, 1] = 2.0
            elif num_cards >= 5: results[i, 1] = 15.0 # Simplified payout for >5 cards
            else: results[i, 1] = -1.0
        else:
            results[i, 1] = -1.0

    return results

class FastSimulator:
    def __init__(self, shoe_dict: dict[str, int]):
        self.shoe_counts = self._encode_shoe(shoe_dict)

    def _encode_shoe(self, shoe_dict: dict[str, int]) -> np.ndarray:
        """Encodes the shoe dictionary into a 52-element numpy array for Numba."""
        shoe_counts = np.zeros(52, dtype=np.int32)
        ranks = {'A':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9, 'J':10, 'Q':11, 'K':12}
        suits = {'S':0, 'H':1, 'D':2, 'C':3}
        for card_str, count in shoe_dict.items():
            rank_str = card_str[:-1] if card_str.startswith("10") else card_str[0]
            suit_str = card_str[-1]
            idx = suits[suit_str] * 13 + ranks[rank_str]
            shoe_counts[idx] = count
        return shoe_counts

    def run(self, total_rounds: int = 500_000, num_threads: int = 4) -> dict[str, float]:
        """Runs the simulation in parallel and returns the mean EV for each bet type."""
        if total_rounds < num_threads: num_threads = total_rounds
        if total_rounds == 0: return {}

        rounds_per_thread = total_rounds // num_threads

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(simulate_chunk, self.shoe_counts.copy(), rounds_per_thread) for _ in range(num_threads)]
            batch_results = [f.result() for f in futures]

        all_results = np.vstack(batch_results)
        evs = {
            "main_ev": np.mean(all_results[:, 0]), 
            "bust_ev": np.mean(all_results[:, 1]),
            "21+3_ev": np.mean(all_results[:, 2]), 
            "perfect_pairs_ev": np.mean(all_results[:, 3]),
            "hot3_ev": np.mean(all_results[:, 4]),
        }
        return evs