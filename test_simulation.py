"""Independent tests for the Roulette Risk Lab calculations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


# Add the project root so tests work from either the root or tests directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation import (  # noqa: E402
    compare_strategies,
    generate_outcomes,
    simulate_strategy,
    theoretical_metrics,
)


class TheoreticalMetricsTests(unittest.TestCase):
    """Verify the exact probability formulas for both roulette wheels."""

    def test_american_expected_value_and_house_edge(self) -> None:
        metrics = theoretical_metrics("American")

        self.assertAlmostEqual(metrics["win_probability"], 18 / 38)
        self.assertAlmostEqual(metrics["expected_value_per_dollar"], -2 / 38)
        self.assertAlmostEqual(metrics["house_edge"], 2 / 38)

    def test_european_expected_value_and_house_edge(self) -> None:
        metrics = theoretical_metrics("European")

        self.assertAlmostEqual(metrics["win_probability"], 18 / 37)
        self.assertAlmostEqual(metrics["expected_value_per_dollar"], -1 / 37)
        self.assertAlmostEqual(metrics["house_edge"], 1 / 37)

    def test_variance_uses_one_minus_mean_squared(self) -> None:
        metrics = theoretical_metrics("American")
        expected_variance = 1 - ((-2 / 38) ** 2)

        self.assertAlmostEqual(metrics["variance_per_dollar"], expected_variance)


class SimulationBehaviorTests(unittest.TestCase):
    """Verify bankroll accounting and reproducibility."""

    def test_random_outcomes_are_reproducible(self) -> None:
        first = generate_outcomes("European", sessions=10, spins=25, seed=123)
        second = generate_outcomes("European", sessions=10, spins=25, seed=123)

        np.testing.assert_array_equal(first, second)

    def test_flat_betting_all_wins(self) -> None:
        outcomes = np.ones((2, 4), dtype=bool)
        result = simulate_strategy(
            outcomes=outcomes,
            wheel="American",
            strategy="Flat",
            starting_bankroll=100,
            base_bet=5,
            table_limit=100,
        )

        np.testing.assert_allclose(result.ending_bankrolls, [120, 120])
        np.testing.assert_allclose(result.total_wagered, [20, 20])
        np.testing.assert_array_equal(result.spins_played, [4, 4])

    def test_martingale_consecutive_losses_reach_ruin(self) -> None:
        outcomes = np.zeros((1, 5), dtype=bool)
        result = simulate_strategy(
            outcomes=outcomes,
            wheel="American",
            strategy="Martingale",
            starting_bankroll=100,
            base_bet=5,
            table_limit=100,
        )

        # Stakes are 5, 10, 20, 40, and the remaining 25 dollars.
        self.assertAlmostEqual(result.ending_bankrolls[0], 0)
        self.assertAlmostEqual(result.total_wagered[0], 100)
        self.assertEqual(result.ruin_spin[0], 5)

    def test_strategy_comparison_reuses_the_same_outcomes(self) -> None:
        first = compare_strategies(
            wheel="European",
            sessions=100,
            spins=20,
            seed=5415,
            starting_bankroll=100,
            base_bet=2,
            table_limit=50,
        )
        second = compare_strategies(
            wheel="European",
            sessions=100,
            spins=20,
            seed=5415,
            starting_bankroll=100,
            base_bet=2,
            table_limit=50,
        )

        for strategy in first:
            np.testing.assert_array_equal(
                first[strategy].ending_bankrolls,
                second[strategy].ending_bankrolls,
            )

    def test_theory_from_action_matches_house_edge_identity(self) -> None:
        outcomes = generate_outcomes("American", sessions=250, spins=40, seed=7)
        result = simulate_strategy(
            outcomes=outcomes,
            wheel="American",
            strategy="Fibonacci",
            starting_bankroll=150,
            base_bet=3,
            table_limit=75,
        )
        summary = result.summary()
        expected = -(2 / 38) * np.mean(result.total_wagered)

        self.assertAlmostEqual(summary["theory_profit_from_action"], expected)


if __name__ == "__main__":
    unittest.main()
