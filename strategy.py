"""
Provides high-level strategic advice by synthesizing information from card counters,
EV simulations, and Bayesian predictions. This module focuses on betting strategy
and overall game-state awareness, including Kelly Criterion bet sizing.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

# Use typing.TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from shoe import Shoe
    from counting import CountingSystem

# Import the new on-demand side bet simulator
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
        
        # --- Main Bet Recommendation ---
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
        
        # --- Side Bet Analysis with Bayesian Overlay ---
        recommendations.append("\n--- Side Bet Analysis ---")
        
        # Get the immediate, composition-dependent EV
        immediate_evs = bayesian_predictor.run_side_bet_simulation(shoe, num_rounds=25000)

        side_bets_long_term = {
            "Perfect Pairs": sim_results.get("perfect_pairs_ev", 0.0),
            "21+3": sim_results.get("21+3_ev", 0.0),
            "Hot 3": sim_results.get("hot3_ev", 0.0),
            "Dealer Bust": sim_results.get("bust_ev", 0.0),
        }
        
        immediate_map = {
            "Perfect Pairs": immediate_evs.get("perfect_pairs_ev"),
            "21+3": immediate_evs.get("21+3_ev"),
            "Hot 3": immediate_evs.get("hot3_ev"),
        }

        found_profitable_side_bet = False
        for name, long_term_ev in side_bets_long_term.items():
            immediate_ev = immediate_map.get(name)
            
            if immediate_ev is not None:
                # Use immediate EV for the decision
                is_profitable = immediate_ev > self.config['sidebet_threshold']
                symbol = "✅" if is_profitable else "❌"
                recommendations.append(
                    f"{symbol} {name}: Profitable for NEXT HAND (EV = {immediate_ev:+.3f}). "
                    f"(Long-term EV: {long_term_ev:+.3f})"
                )
                if is_profitable:
                    found_profitable_side_bet = True
            else:
                # Fallback for bets without immediate calculation (like Bust)
                is_profitable = long_term_ev > self.config['sidebet_threshold']
                symbol = "✅" if is_profitable else "❌"
                recommendations.append(f"{symbol} {name}: Long-term EV = {long_term_ev:+.3f}")
                if is_profitable:
                    found_profitable_side_bet = True

        if not found_profitable_side_bet:
            recommendations.append("No profitable side bets detected for the next hand.")

        # --- Bayesian Insights ---
        if bayes_dealer_totals:
            recommendations.append("\n--- Bayesian Dealer Insights ---")
            sorted_keys = sorted(bayes_dealer_totals.keys(), key=lambda k: (99, k) if isinstance(k, str) else (k, ""))
            
            for total in sorted_keys:
                prob = bayes_dealer_totals[total] * 100
                label = "Bust" if isinstance(total, str) else str(total)
                recommendations.append(f"  • P(Dealer Total = {label}): {prob:.1f}%")

            bust_chance = bayes_dealer_totals.get("Bust", 0.0)
            if bust_chance > self.config['dealer_bust_alert_threshold']:
                recommendations.append("\nNote: High dealer bust probability. Player stands may be stronger.")

        return recommendations