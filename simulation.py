"""Core probability and simulation logic for Roulette Risk Lab.

The functions in this module are intentionally separate from the Streamlit
interface. This makes the statistical calculations easier to read, test, and
reuse without launching the web app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


# Store the two standard wheel configurations used throughout the project.
WHEELS: Final[dict[str, dict[str, int]]] = {
    "European": {"pockets": 37, "red": 18, "black": 18, "green": 1},
    "American": {"pockets": 38, "red": 18, "black": 18, "green": 2},
}

STRATEGIES: Final[tuple[str, ...]] = ("Flat", "Martingale", "Fibonacci")


@dataclass(frozen=True)
class SimulationResult:
    """Container for the results from one simulated betting strategy."""

    strategy: str
    wheel: str
    starting_bankroll: float
    base_bet: float
    table_limit: float
    win_probability: float
    ending_bankrolls: np.ndarray
    total_wagered: np.ndarray
    spins_played: np.ndarray
    ruin_spin: np.ndarray
    sample_paths: np.ndarray

    @property
    def profits(self) -> np.ndarray:
        """Return each session's ending profit or loss."""

        return self.ending_bankrolls - self.starting_bankroll

    @property
    def house_edge(self) -> float:
        """Return the expected casino advantage per dollar wagered."""

        return 1.0 - (2.0 * self.win_probability)

    def summary(self) -> dict[str, float]:
        """Calculate the headline statistics displayed in the app."""

        profits = self.profits
        session_count = profits.size

        # Measure variation across complete simulated sessions.
        profit_sd = float(np.std(profits, ddof=1)) if session_count > 1 else 0.0
        monte_carlo_se = profit_sd / np.sqrt(session_count) if session_count else 0.0
        mean_profit = float(np.mean(profits))

        # Quantify Monte Carlo uncertainty around the simulated mean.
        ci_low = mean_profit - (1.96 * monte_carlo_se)
        ci_high = mean_profit + (1.96 * monte_carlo_se)

        # The expected loss remains the house edge times expected total action,
        # even when the next stake depends on earlier independent outcomes.
        mean_total_wagered = float(np.mean(self.total_wagered))
        theory_from_action = -self.house_edge * mean_total_wagered

        return {
            "sessions": float(session_count),
            "mean_ending_bankroll": float(np.mean(self.ending_bankrolls)),
            "median_ending_bankroll": float(np.median(self.ending_bankrolls)),
            "mean_profit": mean_profit,
            "median_profit": float(np.median(profits)),
            "profit_sd": profit_sd,
            "monte_carlo_se": float(monte_carlo_se),
            "mc_ci_low": float(ci_low),
            "mc_ci_high": float(ci_high),
            "probability_ahead": float(np.mean(profits > 0)),
            "probability_break_even": float(np.mean(np.isclose(profits, 0))),
            "probability_ruin": float(np.mean(self.ending_bankrolls <= 0)),
            "mean_total_wagered": mean_total_wagered,
            "mean_spins_played": float(np.mean(self.spins_played)),
            "theory_profit_from_action": float(theory_from_action),
            "fifth_percentile_profit": float(np.percentile(profits, 5)),
            "ninety_fifth_percentile_profit": float(np.percentile(profits, 95)),
            "minimum_profit": float(np.min(profits)),
            "maximum_profit": float(np.max(profits)),
        }


def validate_wheel(wheel: str) -> None:
    """Raise a clear error when an unsupported wheel is requested."""

    if wheel not in WHEELS:
        valid = ", ".join(WHEELS)
        raise ValueError(f"Unknown wheel '{wheel}'. Choose one of: {valid}.")


def theoretical_metrics(wheel: str) -> dict[str, float]:
    """Return exact results for a one-dollar even-money roulette bet."""

    validate_wheel(wheel)
    pockets = WHEELS[wheel]["pockets"]
    win_probability = WHEELS[wheel]["red"] / pockets
    loss_probability = 1.0 - win_probability

    # A one-dollar bet returns +$1 after a win and -$1 after a loss.
    expected_value = win_probability - loss_probability

    # Since X is always +1 or -1, E[X^2] equals 1.
    variance = 1.0 - (expected_value**2)

    return {
        "pockets": float(pockets),
        "red_pockets": float(WHEELS[wheel]["red"]),
        "black_pockets": float(WHEELS[wheel]["black"]),
        "green_pockets": float(WHEELS[wheel]["green"]),
        "win_probability": float(win_probability),
        "loss_probability": float(loss_probability),
        "expected_value_per_dollar": float(expected_value),
        "house_edge": float(-expected_value),
        "variance_per_dollar": float(variance),
        "standard_deviation_per_dollar": float(np.sqrt(variance)),
    }


def generate_outcomes(
    wheel: str,
    sessions: int,
    spins: int,
    seed: int,
) -> np.ndarray:
    """Generate a reproducible matrix where True represents a winning spin."""

    validate_wheel(wheel)

    if sessions <= 0 or spins <= 0:
        raise ValueError("Sessions and spins must both be positive integers.")

    probability = theoretical_metrics(wheel)["win_probability"]
    random_generator = np.random.default_rng(seed)

    # Each cell is independent because roulette spins are modeled independently.
    return random_generator.random((sessions, spins)) < probability


def fibonacci_multipliers(length: int) -> np.ndarray:
    """Create the 1, 1, 2, 3, 5... betting multipliers."""

    if length <= 0:
        raise ValueError("The Fibonacci sequence length must be positive.")

    sequence = np.ones(max(length, 2), dtype=float)
    for index in range(2, sequence.size):
        sequence[index] = sequence[index - 1] + sequence[index - 2]

    return sequence[:length]


def simulate_strategy(
    outcomes: np.ndarray,
    wheel: str,
    strategy: str,
    starting_bankroll: float,
    base_bet: float,
    table_limit: float,
    sample_path_count: int = 12,
) -> SimulationResult:
    """Apply one betting strategy to a shared matrix of roulette outcomes."""

    validate_wheel(wheel)

    if strategy not in STRATEGIES:
        valid = ", ".join(STRATEGIES)
        raise ValueError(f"Unknown strategy '{strategy}'. Choose one of: {valid}.")
    if outcomes.ndim != 2:
        raise ValueError("Outcomes must be a two-dimensional sessions-by-spins matrix.")
    if starting_bankroll <= 0 or base_bet <= 0 or table_limit <= 0:
        raise ValueError("Bankroll, base bet, and table limit must all be positive.")
    if base_bet > starting_bankroll:
        raise ValueError("The base bet cannot exceed the starting bankroll.")
    if table_limit < base_bet:
        raise ValueError("The table limit cannot be smaller than the base bet.")

    session_count, spin_count = outcomes.shape
    path_count = min(max(sample_path_count, 0), session_count)

    # Initialize every session with the same bankroll and progression state.
    bankrolls = np.full(session_count, float(starting_bankroll))
    total_wagered = np.zeros(session_count, dtype=float)
    spins_played = np.zeros(session_count, dtype=int)
    ruin_spin = np.full(session_count, -1, dtype=int)
    next_martingale_bet = np.full(session_count, float(base_bet))
    fibonacci_index = np.zeros(session_count, dtype=int)
    fibonacci_values = fibonacci_multipliers(spin_count + 2)

    # Retain only a small set of paths so charts stay readable and memory stays low.
    sample_paths = np.zeros((path_count, spin_count + 1), dtype=float)
    if path_count:
        sample_paths[:, 0] = starting_bankroll

    # Process one spin at a time while updating all sessions simultaneously.
    for spin_index in range(spin_count):
        active = bankrolls > 0

        if strategy == "Flat":
            planned_bets = np.full(session_count, float(base_bet))
        elif strategy == "Martingale":
            planned_bets = next_martingale_bet.copy()
        else:
            planned_bets = base_bet * fibonacci_values[fibonacci_index]

        # A player cannot wager more than the table limit or remaining bankroll.
        stakes = np.minimum(planned_bets, table_limit)
        stakes = np.minimum(stakes, bankrolls)
        stakes = np.where(active, stakes, 0.0)

        wins = outcomes[:, spin_index] & active
        losses = (~outcomes[:, spin_index]) & active

        # Even-money roulette pays one dollar of profit per dollar wagered.
        bankrolls[wins] += stakes[wins]
        bankrolls[losses] -= stakes[losses]
        bankrolls[np.isclose(bankrolls, 0.0)] = 0.0

        total_wagered += stakes
        spins_played += active.astype(int)

        # Record the first spin on which each session loses its entire bankroll.
        newly_ruined = active & (bankrolls <= 0) & (ruin_spin < 0)
        ruin_spin[newly_ruined] = spin_index + 1

        # Update the strategy state only after the current spin is resolved.
        if strategy == "Martingale":
            next_martingale_bet[wins] = base_bet
            next_martingale_bet[losses] = np.minimum(
                next_martingale_bet[losses] * 2.0,
                table_limit,
            )
        elif strategy == "Fibonacci":
            fibonacci_index[losses] = np.minimum(
                fibonacci_index[losses] + 1,
                fibonacci_values.size - 1,
            )
            fibonacci_index[wins] = np.maximum(fibonacci_index[wins] - 2, 0)

        if path_count:
            sample_paths[:, spin_index + 1] = bankrolls[:path_count]

    return SimulationResult(
        strategy=strategy,
        wheel=wheel,
        starting_bankroll=float(starting_bankroll),
        base_bet=float(base_bet),
        table_limit=float(table_limit),
        win_probability=float(theoretical_metrics(wheel)["win_probability"]),
        ending_bankrolls=bankrolls,
        total_wagered=total_wagered,
        spins_played=spins_played,
        ruin_spin=ruin_spin,
        sample_paths=sample_paths,
    )


def compare_strategies(
    wheel: str,
    sessions: int,
    spins: int,
    seed: int,
    starting_bankroll: float,
    base_bet: float,
    table_limit: float,
    sample_path_count: int = 12,
) -> dict[str, SimulationResult]:
    """Run all strategies against exactly the same random wheel outcomes."""

    shared_outcomes = generate_outcomes(wheel, sessions, spins, seed)

    return {
        strategy: simulate_strategy(
            outcomes=shared_outcomes,
            wheel=wheel,
            strategy=strategy,
            starting_bankroll=starting_bankroll,
            base_bet=base_bet,
            table_limit=table_limit,
            sample_path_count=sample_path_count,
        )
        for strategy in STRATEGIES
    }


def convergence_series(
    wheel: str,
    spins: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the running average profit from repeated one-dollar flat bets."""

    if spins <= 0:
        raise ValueError("Spins must be positive.")

    outcomes = generate_outcomes(wheel, sessions=1, spins=spins, seed=seed)[0]
    profits = np.where(outcomes, 1.0, -1.0)
    running_average = np.cumsum(profits) / np.arange(1, spins + 1)
    theory = theoretical_metrics(wheel)["expected_value_per_dollar"]

    return np.arange(1, spins + 1), running_average, float(theory)
