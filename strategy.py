"""
Provides high-level strategic advice by synthesizing information from card counters,
EV simulations, and Bayesian predictions. This module focuses on betting strategy
and overall game-state awareness, including Kelly Criterion bet sizing.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shoe import Shoe
    from counting import CountingSystem

import bayesian_predictor

class StrategyAdvisor:
    """
    Aggregates data from various sources to provide comprehensive betting advice.
    """
    def __init__(self, config: dict | None = None):
        """
        Initializes the advisor with a given configuration.
        """
        self.config = config or {
            'sidebet_threshold': 0.0,
            'dealer_bust_alert_threshold': 0.40,
            'kelly_fraction': 0.5,
            'kelly_fraction_name': 'Half'
        }

    def generate_recommendations(
        self,
        shoe: 'Shoe',
        counters: dict[str, 'CountingSystem'],
        sim_results: dict[str, float],
        bayes_dealer_totals: dict[str | int, float] | None = None
    ) -> list[str]:
        """
        Generates a list of string-based recommendations for the player.
        """
        recommendations = []
        
        # --- Main Bet Strategy ---
        main_ev = sim_results.get("main_ev", 0.0)
        advantage_pct = main_ev * 100
        kelly_fraction = self.config.get('kelly_fraction', 0.5)
        kelly_name = self.config.get('kelly_fraction_name', 'Half')

        recommendations.append("--- Main Bet Strategy ---")
        if main_ev > 0:
            bet_pct = advantage_pct * kelly_fraction
            recommendations.append(f"Player Advantage: {advantage_pct:+.2f}%. Favorable shoe.")
            recommendations.append(f"Optimal Bet ({kelly_name} Kelly): Bet {bet_pct:.2f}% of bankroll.")
        else:
            recommendations.append(f"Player Advantage: {advantage_pct:+.2f}%. No edge. Bet table minimum.")
        
        # --- Side Bet Analysis ---
        recommendations.append("\n--- Side Bet Analysis (Composition-Dependent EV) ---")
        immediate_evs = bayesian_predictor.run_side_bet_simulation(shoe, num_rounds=25000)
        side_bets = {
            "21+3": immediate_evs.get("21+3_ev", 0.0),
            "Perfect Pairs": immediate_evs.get("perfect_pairs_ev", 0.0),
            "Hot 3": immediate_evs.get("hot3_ev", 0.0),
            "Dealer Bust": sim_results.get("bust_ev", 0.0),
        }
        
        found_profitable_side_bet = False
        for name, ev in side_bets.items():
            if ev > self.config['sidebet_threshold']:
                recommendations.append(f"✅ {name}: Profitable (EV = {ev:+.3f})")
                found_profitable_side_bet = True
            else:
                recommendations.append(f"❌ {name}: Not Profitable (EV = {ev:+.3f})")

        if not found_profitable_side_bet:
            recommendations.append("No profitable side bets detected for this shoe composition.")

        # --- Bayesian Next Card Insights ---
        recommendations.append("\n--- Bayesian Next Card Insights ---")
        shoe_cards = shoe.get_remaining_cards()
        
        # 1. Top 5 Exact Cards
        top_cards = bayesian_predictor.next_card_probabilities(shoe_cards, top_n=5)
        if top_cards:
            card_preds = [f"{c} ({p:.1%})" for c, p in top_cards]
            recommendations.append(f"Top 5 Cards: {', '.join(card_preds)}")

        # 2. Rank Distribution
        rank_dist = bayesian_predictor.rank_distribution_probabilities(shoe_cards)
        if rank_dist:
            dist_preds = [f"{name}: {p:.1%}" for name, p in rank_dist.items()]
            recommendations.append(f"Rank Dist: {', '.join(dist_preds)}")
            if rank_dist.get("High (10-K)", 0) > 0.35: # Example insight
                 recommendations.append("Insight: Shoe is rich in 10-value cards.")

        # 3. Suit Distribution
        suit_dist = bayesian_predictor.next_suit_probabilities(shoe_cards)
        if suit_dist:
            suit_preds = [f"{s} ({p:.1%})" for s, p in suit_dist.items()]
            recommendations.append(f"Suit Dist: {', '.join(suit_preds)}")
            # Example insight for flush-based side bets
            if any(p > 0.3 for p in suit_dist.values()):
                recommendations.append("Insight: A specific suit is dominant, increasing flush odds.")


        # --- Bayesian Dealer Insights ---
        if bayes_dealer_totals:
            recommendations.append("\n--- Bayesian Dealer Insights ---")
            sorted_keys = sorted(bayes_dealer_totals.keys(), key=lambda k: (99, k) if isinstance(k, str) else (k, ""))
            
            for total in sorted_keys:
                prob = bayes_dealer_totals[total] * 100
                label = "Bust" if isinstance(total, str) else str(total)
                recommendations.append(f"  • P(Dealer Final = {label}): {prob:.1f}%")

            bust_chance = bayes_dealer_totals.get("Bust", 0.0)
            if bust_chance > self.config['dealer_bust_alert_threshold']:
                recommendations.append(f"\nALERT: High dealer bust probability ({bust_chance:.1%}).")

        return recommendations