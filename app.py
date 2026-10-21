"""Streamlit interface for the Roulette Risk Lab class project."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulation import (
    STRATEGIES,
    WHEELS,
    SimulationResult,
    compare_strategies,
    convergence_series,
    theoretical_metrics,
)


# Configure the browser tab and use a wide canvas for analytical charts.
st.set_page_config(
    page_title="Roulette Risk Lab",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Apply a restrained visual system that remains readable on desktop and mobile.
st.markdown(
    """
    <style>
        :root {
            --ink: #10233f;
            --muted: #5f6f82;
            --line: #dfe6ee;
            --soft: #f5f7fa;
            --red: #c73a45;
            --gold: #d99a2b;
            --green: #16836f;
        }

        .stApp {
            background: #ffffff;
            color: var(--ink);
        }

        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        .hero {
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.4rem;
            padding-bottom: 1.35rem;
        }

        .eyebrow {
            color: var(--red);
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }

        .hero h1 {
            color: var(--ink);
            font-size: clamp(2.1rem, 5vw, 3.5rem);
            letter-spacing: -0.045em;
            line-height: 1.02;
            margin: 0;
        }

        .hero p {
            color: var(--muted);
            font-size: 1.08rem;
            margin: 0.7rem 0 0;
            max-width: 790px;
        }

        div[data-testid="stMetric"] {
            background: var(--soft);
            border: 1px solid var(--line);
            border-radius: 12px;
            min-height: 118px;
            padding: 1rem;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
        }

        div[data-testid="stSidebar"] {
            border-right: 1px solid var(--line);
        }

        .method-card {
            background: var(--soft);
            border: 1px solid var(--line);
            border-radius: 12px;
            height: 100%;
            padding: 1rem 1.15rem;
        }

        .method-card h4 {
            color: var(--ink);
            margin: 0 0 0.45rem;
        }

        .method-card p {
            color: var(--muted);
            margin: 0;
        }

        .small-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        [data-testid="stPlotlyChart"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-top: 1.2rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Keep visual styling consistent across every Plotly figure.
COLORS = {
    "navy": "#17365d",
    "red": "#c73a45",
    "gold": "#d99a2b",
    "green": "#16836f",
    "gray": "#7d8998",
    "light": "#e9eef4",
}

STRATEGY_COLORS = {
    "Flat": COLORS["navy"],
    "Martingale": COLORS["red"],
    "Fibonacci": COLORS["gold"],
}

PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


def format_currency(value: float, decimals: int = 2) -> str:
    """Format positive and negative dollar amounts consistently."""

    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def format_markdown_currency(value: float, decimals: int = 2) -> str:
    """Format currency without letting Markdown treat dollars as math."""

    return format_currency(value, decimals).replace("$", r"\$")


def format_probability(value: float, decimals: int = 1) -> str:
    """Convert a decimal probability to a percentage string."""

    return f"{100 * value:.{decimals}f}%"


def apply_plot_style(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    """Apply the shared chart layout."""

    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=height,
        # Leave enough room for both the chart title and a horizontal legend.
        margin={"l": 32, "r": 24, "t": 104, "b": 42},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#10233f", "size": 13},
        hoverlabel={"font": {"family": "Arial, sans-serif"}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#dfe6ee")
    fig.update_yaxes(gridcolor="#edf1f5", zerolinecolor="#c7d0db")
    return fig


@st.cache_data(show_spinner=False)
def run_all_strategies(
    wheel: str,
    sessions: int,
    spins: int,
    seed: int,
    starting_bankroll: float,
    base_bet: float,
    table_limit: float,
) -> dict[str, SimulationResult]:
    """Cache simulations so switching tabs does not repeat identical work."""

    return compare_strategies(
        wheel=wheel,
        sessions=sessions,
        spins=spins,
        seed=seed,
        starting_bankroll=starting_bankroll,
        base_bet=base_bet,
        table_limit=table_limit,
        sample_path_count=12,
    )


@st.cache_data(show_spinner=False)
def cached_convergence(
    wheel: str,
    spins: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cache the long-run convergence demonstration."""

    return convergence_series(wheel, spins, seed)


def make_path_chart(result: SimulationResult) -> go.Figure:
    """Plot a readable sample of bankroll paths from the selected strategy."""

    fig = go.Figure()
    spin_axis = np.arange(result.sample_paths.shape[1])

    for path_number, path in enumerate(result.sample_paths, start=1):
        fig.add_trace(
            go.Scatter(
                x=spin_axis,
                y=path,
                mode="lines",
                name=f"Session {path_number}",
                line={"color": STRATEGY_COLORS[result.strategy], "width": 1.35},
                opacity=0.35,
                hovertemplate="Spin %{x}<br>Bankroll $%{y:,.2f}<extra></extra>",
                showlegend=False,
            )
        )

    # Add the unconstrained flat-bet expectation as a mathematical reference.
    expected_path = (
        result.starting_bankroll
        - result.house_edge * result.base_bet * spin_axis
    )
    fig.add_trace(
        go.Scatter(
            x=spin_axis,
            y=expected_path,
            mode="lines",
            name="Flat-bet expectation",
            line={"color": COLORS["gray"], "width": 2.5, "dash": "dash"},
            hovertemplate="Spin %{x}<br>Expected bankroll $%{y:,.2f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=result.starting_bankroll,
        line_color="#9aa6b4",
        line_dash="dot",
        annotation_text="Starting bankroll",
        annotation_position="bottom right",
    )
    fig.update_xaxes(title="Spin")
    fig.update_yaxes(title="Bankroll", tickprefix="$", rangemode="tozero")
    return apply_plot_style(fig, f"Sample bankroll paths · {result.strategy}")


def make_distribution_chart(result: SimulationResult) -> go.Figure:
    """Plot the distribution of ending bankrolls across all sessions."""

    summary = result.summary()
    fig = go.Figure(
        go.Histogram(
            x=result.ending_bankrolls,
            nbinsx=45,
            marker={"color": STRATEGY_COLORS[result.strategy]},
            opacity=0.86,
            hovertemplate="Ending bankroll $%{x:,.2f}<br>Sessions %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=result.starting_bankroll,
        line_color="#10233f",
        line_width=2,
        line_dash="dot",
        annotation_text="Start",
        annotation_position="top left",
        annotation_yshift=12,
    )
    fig.add_vline(
        x=summary["mean_ending_bankroll"],
        line_color=COLORS["green"],
        line_width=2,
        annotation_text="Mean",
        annotation_position="top right",
        annotation_yshift=-10,
    )
    fig.update_xaxes(title="Ending bankroll", tickprefix="$", rangemode="tozero")
    fig.update_yaxes(title="Number of sessions")
    return apply_plot_style(fig, "Where sessions finished")


def make_profit_comparison_chart(summary_frame: pd.DataFrame) -> go.Figure:
    """Compare simulated and theoretical average profits by strategy."""

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary_frame["Strategy"],
            y=summary_frame["Simulated mean profit"],
            name="Simulated mean",
            marker_color=[STRATEGY_COLORS[name] for name in summary_frame["Strategy"]],
            hovertemplate="%{x}<br>Simulated %{y:$,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=summary_frame["Strategy"],
            y=summary_frame["Theory from action"],
            name="Expected from dollars wagered",
            marker_color="#aeb9c6",
            hovertemplate="%{x}<br>Expected %{y:$,.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="#10233f", line_width=1)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Average session profit", tickprefix="$", zeroline=True)
    return apply_plot_style(fig, "Average result versus mathematical expectation")


def make_risk_comparison_chart(summary_frame: pd.DataFrame) -> go.Figure:
    """Compare the chances of finishing ahead and going broke."""

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary_frame["Strategy"],
            y=100 * summary_frame["Probability ahead"],
            name="Finished ahead",
            marker_color=COLORS["green"],
            hovertemplate="%{x}<br>Ahead %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=summary_frame["Strategy"],
            y=100 * summary_frame["Probability of ruin"],
            name="Lost entire bankroll",
            marker_color=COLORS["red"],
            hovertemplate="%{x}<br>Ruin %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Share of sessions", ticksuffix="%", range=[0, 100])
    return apply_plot_style(fig, "Two very different views of risk")


def make_wheel_chart(wheel: str) -> go.Figure:
    """Show how green pockets create the house advantage."""

    counts = WHEELS[wheel]
    fig = go.Figure(
        go.Pie(
            labels=["Red", "Black", "Green"],
            values=[counts["red"], counts["black"], counts["green"]],
            hole=0.58,
            marker={"colors": [COLORS["red"], "#222a35", COLORS["green"]]},
            textinfo="label+value",
            hovertemplate="%{label}: %{value} pockets (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    return apply_plot_style(fig, f"{wheel} wheel composition", height=390)


def make_convergence_chart(wheel: str, seed: int) -> go.Figure:
    """Demonstrate the law of large numbers with repeated one-dollar bets."""

    spin_axis, running_average, theory = cached_convergence(wheel, 50_000, seed)

    # Plot more points early and fewer later to keep the interactive chart light.
    selected = np.unique(np.geomspace(1, spin_axis.size, 1_600).astype(int)) - 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spin_axis[selected],
            y=running_average[selected],
            mode="lines",
            name="Simulated running average",
            line={"color": COLORS["navy"], "width": 2},
            hovertemplate="%{x:,} spins<br>Average %{y:$,.4f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=theory,
        line_color=COLORS["red"],
        line_width=2.5,
        line_dash="dash",
        annotation_text=f"Theory {theory:.4f}",
        annotation_position="bottom right",
    )
    fig.update_xaxes(title="Cumulative spins", type="log")
    fig.update_yaxes(title="Average profit per $1 bet", tickformat="$.3f")
    return apply_plot_style(fig, "One random sequence moves toward the expected value", height=460)


def build_comparison_frame(
    results: dict[str, SimulationResult],
) -> pd.DataFrame:
    """Convert each strategy summary into one comparison table."""

    rows: list[dict[str, float | str]] = []
    for strategy in STRATEGIES:
        summary = results[strategy].summary()
        rows.append(
            {
                "Strategy": strategy,
                "Simulated mean profit": summary["mean_profit"],
                "Theory from action": summary["theory_profit_from_action"],
                "Median profit": summary["median_profit"],
                "Profit SD": summary["profit_sd"],
                "Probability ahead": summary["probability_ahead"],
                "Probability of ruin": summary["probability_ruin"],
                "Average wagered": summary["mean_total_wagered"],
                "Average spins played": summary["mean_spins_played"],
            }
        )

    return pd.DataFrame(rows)


# Present the purpose before asking the user to adjust assumptions.
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Interactive probability study</div>
        <h1>Roulette Risk Lab</h1>
        <p>
            Test how the wheel, bankroll, and betting system change short-run
            outcomes while the house edge shapes long-run performance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Collect inputs in a form so the model runs only when the user requests it.
with st.sidebar:
    st.header("Experiment settings")
    st.caption("Change the assumptions, then run the model.")

    with st.form("simulation_controls"):
        wheel = st.selectbox(
            "Roulette wheel",
            options=["American", "European"],
            help="American roulette has 0 and 00. European roulette has only 0.",
        )
        selected_strategy = st.selectbox(
            "Strategy to explore",
            options=list(STRATEGIES),
        )
        starting_bankroll = st.number_input(
            "Starting bankroll ($)",
            min_value=20,
            max_value=10_000,
            value=200,
            step=20,
        )
        base_bet = st.number_input(
            "Base bet ($)",
            min_value=1,
            max_value=1_000,
            value=5,
            step=1,
        )
        table_limit = st.number_input(
            "Table limit ($)",
            min_value=1,
            max_value=100_000,
            value=250,
            step=25,
            help="Progressive systems cannot place a bet above this amount.",
        )
        spins = st.select_slider(
            "Spins per session",
            options=[25, 50, 100, 200, 300, 500],
            value=100,
        )
        sessions = st.select_slider(
            "Simulated sessions",
            options=[500, 1_000, 2_500, 5_000, 10_000],
            value=5_000,
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=999_999,
            value=5_415,
            step=1,
            help="The same seed and settings reproduce the same results.",
        )
        submitted = st.form_submit_button("Run simulation", width="stretch")

    st.divider()
    sidebar_theory = theoretical_metrics(wheel)
    st.markdown("**Mathematical baseline**")
    st.metric("Chance of winning one bet", format_probability(sidebar_theory["win_probability"], 2))
    st.metric("House edge", format_probability(sidebar_theory["house_edge"], 2))
    st.caption("Every strategy faces the same expected loss per dollar wagered.")


# Stop early with an actionable message when the settings are impossible.
if base_bet > starting_bankroll:
    st.error("The base bet cannot exceed the starting bankroll. Adjust the sidebar settings.")
    st.stop()

if table_limit < base_bet:
    st.error("The table limit cannot be smaller than the base bet. Adjust the sidebar settings.")
    st.stop()


# Run all strategies against shared outcomes for a controlled comparison.
with st.spinner("Simulating independent roulette sessions..."):
    all_results = run_all_strategies(
        wheel=wheel,
        sessions=int(sessions),
        spins=int(spins),
        seed=int(seed),
        starting_bankroll=float(starting_bankroll),
        base_bet=float(base_bet),
        table_limit=float(table_limit),
    )

selected_result = all_results[selected_strategy]
selected_summary = selected_result.summary()
theory = theoretical_metrics(wheel)
comparison_frame = build_comparison_frame(all_results)


# Separate exploration, comparison, theory, and documentation into focused tabs.
simulation_tab, comparison_tab, math_tab, methods_tab = st.tabs(
    ["Simulation lab", "Strategy comparison", "The math", "Methods & rubric"]
)


with simulation_tab:
    wheel_article = "an" if wheel == "American" else "a"
    st.subheader(f"{selected_strategy} strategy on {wheel_article} {wheel} wheel")
    st.caption(
        f"{sessions:,} sessions × up to {spins} spins · seed {seed:,} · "
        f"{format_currency(base_bet, 0)} base bet"
    )

    # Two rows keep full metric labels readable at ordinary laptop widths.
    primary_metrics = st.columns(3)
    primary_metrics[0].metric(
        "Average ending bankroll",
        format_currency(selected_summary["mean_ending_bankroll"]),
        delta=format_currency(selected_summary["mean_profit"]),
    )
    primary_metrics[1].metric(
        "Finished ahead",
        format_probability(selected_summary["probability_ahead"]),
    )
    primary_metrics[2].metric(
        "Lost entire bankroll",
        format_probability(selected_summary["probability_ruin"]),
        delta=None,
    )

    secondary_metrics = st.columns(2)
    secondary_metrics[0].metric(
        "Average dollars wagered",
        format_currency(selected_summary["mean_total_wagered"], 0),
    )
    secondary_metrics[1].metric(
        "Profit volatility",
        format_currency(selected_summary["profit_sd"]),
        help="Standard deviation of complete-session profit.",
    )

    expected_action_profit = selected_summary["theory_profit_from_action"]
    st.info(
        f"The simulation produced an average profit of "
        f"**{format_markdown_currency(selected_summary['mean_profit'])}**. Based on the "
        f"{format_probability(theory['house_edge'], 2)} house edge and the average "
        f"amount wagered, the mathematical expectation is "
        f"**{format_markdown_currency(expected_action_profit)}** per session. The 95% Monte "
        f"Carlo interval for the estimated mean is "
        f"**{format_markdown_currency(selected_summary['mc_ci_low'])} to "
        f"{format_markdown_currency(selected_summary['mc_ci_high'])}**."
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            make_path_chart(selected_result),
            width="stretch",
            config=PLOT_CONFIG,
        )
    with chart_right:
        st.plotly_chart(
            make_distribution_chart(selected_result),
            width="stretch",
            config=PLOT_CONFIG,
        )

    st.markdown("#### Outcome range")
    outcome_frame = pd.DataFrame(
        {
            "Measure": [
                "5th percentile profit",
                "Median profit",
                "95th percentile profit",
                "Worst simulated result",
                "Best simulated result",
                "Average spins completed",
            ],
            "Value": [
                format_currency(selected_summary["fifth_percentile_profit"]),
                format_currency(selected_summary["median_profit"]),
                format_currency(selected_summary["ninety_fifth_percentile_profit"]),
                format_currency(selected_summary["minimum_profit"]),
                format_currency(selected_summary["maximum_profit"]),
                f"{selected_summary['mean_spins_played']:.1f} of {spins}",
            ],
        }
    )
    st.dataframe(outcome_frame, width="stretch", hide_index=True)

    # Let users export the simulated session-level data for independent analysis.
    session_data = pd.DataFrame(
        {
            "session": np.arange(1, sessions + 1),
            "ending_bankroll": selected_result.ending_bankrolls,
            "profit": selected_result.profits,
            "total_wagered": selected_result.total_wagered,
            "spins_played": selected_result.spins_played,
            "ruin_spin": np.where(selected_result.ruin_spin < 0, np.nan, selected_result.ruin_spin),
        }
    )
    st.download_button(
        "Download these simulation results",
        data=session_data.to_csv(index=False).encode("utf-8"),
        file_name=f"roulette_{wheel.lower()}_{selected_strategy.lower()}_results.csv",
        mime="text/csv",
    )


with comparison_tab:
    st.subheader("Same wheel outcomes, different betting rules")
    st.write(
        "Each strategy receives the exact same sequence of wins and losses. This "
        "controlled design isolates the effect of bet sizing from random luck."
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            make_profit_comparison_chart(comparison_frame),
            width="stretch",
            config=PLOT_CONFIG,
        )
    with chart_right:
        st.plotly_chart(
            make_risk_comparison_chart(comparison_frame),
            width="stretch",
            config=PLOT_CONFIG,
        )

    display_comparison = comparison_frame.copy()
    for column in [
        "Simulated mean profit",
        "Theory from action",
        "Median profit",
        "Profit SD",
        "Average wagered",
    ]:
        display_comparison[column] = display_comparison[column].map(format_currency)
    for column in ["Probability ahead", "Probability of ruin"]:
        display_comparison[column] = display_comparison[column].map(format_probability)
    display_comparison["Average spins played"] = display_comparison[
        "Average spins played"
    ].map(lambda value: f"{value:.1f}")

    st.dataframe(display_comparison, width="stretch", hide_index=True)

    st.warning(
        "Progressive systems can produce more winning sessions, but they do so by "
        "concentrating losses into fewer, much larger failures. Because each dollar "
        "still faces the same house edge, increasing total action increases expected loss."
    )

    explanation_columns = st.columns(3)
    explanations = {
        "Flat": (
            "Uses the same base bet every spin. It produces the clearest link between "
            "the number of spins and expected loss."
        ),
        "Martingale": (
            "Doubles after a loss and resets after a win. Long losing streaks collide "
            "with the bankroll or table limit."
        ),
        "Fibonacci": (
            "Moves one step forward after a loss and two steps back after a win. Its "
            "stake growth is slower than Martingale but still increases tail risk."
        ),
    }
    for column, strategy in zip(explanation_columns, STRATEGIES):
        with column:
            st.markdown(
                f"<div class='method-card'><h4>{strategy}</h4>"
                f"<p>{explanations[strategy]}</p></div>",
                unsafe_allow_html=True,
            )


with math_tab:
    st.subheader("Why the green pockets matter")

    wheel_column, formula_column = st.columns([0.9, 1.1])
    with wheel_column:
        st.plotly_chart(
            make_wheel_chart(wheel),
            width="stretch",
            config=PLOT_CONFIG,
        )

    with formula_column:
        st.markdown("#### Exact probability")
        st.latex(
            rf"P(\text{{win}})=\frac{{18}}{{{int(theory['pockets'])}}}"
            rf"={theory['win_probability']:.5f}"
        )
        st.latex(
            r"E[X]=P(\text{win})(+1)+P(\text{loss})(-1)"
        )
        st.latex(
            rf"E[X]={theory['expected_value_per_dollar']:.5f}\text{{ dollars per dollar wagered}}"
        )
        st.write(
            f"A one-dollar {wheel} roulette bet therefore loses "
            f"**{format_currency(theory['house_edge'])} on average**, or "
            f"**{format_probability(theory['house_edge'], 2)}** of every dollar wagered."
        )

        exact_columns = st.columns(2)
        exact_columns[0].metric(
            "Variance per $1 bet",
            f"{theory['variance_per_dollar']:.4f}",
        )
        exact_columns[1].metric(
            "Standard deviation",
            f"{theory['standard_deviation_per_dollar']:.4f}",
        )

    st.plotly_chart(
        make_convergence_chart(wheel, int(seed) + 17),
        width="stretch",
        config=PLOT_CONFIG,
    )
    st.caption(
        "The running average is noisy at first. As independent spins accumulate, it "
        "tends to move toward the exact expected value. A different seed creates a "
        "different path but does not change the theoretical target."
    )

    st.markdown("#### Expected value for a changing bet")
    st.latex(
        r"E[\text{session profit}]=-\text{house edge}\times E[\text{total dollars wagered}]"
    )
    st.write(
        "A strategy may change when and how much the player bets, but the next spin "
        "remains independent of earlier spins. Every new dollar placed on red or black "
        "therefore carries the same negative expected value."
    )


with methods_tab:
    st.subheader("Study design")
    st.markdown(
        "**Research question:** How do wheel type, bankroll, session length, table "
        "limits, and betting strategy affect expected loss and the distribution of "
        "roulette outcomes?"
    )

    method_columns = st.columns(3)
    method_cards = [
        (
            "Random process",
            "Each spin is an independent Bernoulli trial. A win returns the amount "
            "wagered as profit; a losing color or green pocket forfeits the stake.",
        ),
        (
            "Controlled comparison",
            "All strategies use the same simulated outcome matrix. Differences therefore "
            "come from betting rules rather than different random samples.",
        ),
        (
            "Reproducibility",
            "The seed fixes the pseudo-random sequence. Repeating the same settings and "
            "seed reproduces the same numerical results.",
        ),
    ]
    for column, (heading, body) in zip(method_columns, method_cards):
        with column:
            st.markdown(
                f"<div class='method-card'><h4>{heading}</h4><p>{body}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Assumptions")
    st.markdown(
        """
        - The player makes an even-money bet on red or black
        - Spins are independent and the wheel probabilities remain constant
        - A win earns the stake as profit; a loss forfeits the stake
        - A bet cannot exceed the remaining bankroll or selected table limit
        - Play stops when the bankroll reaches zero or the session reaches its spin limit
        - No taxes, comps, minimum table bet, dealer errors, or biased-wheel effects are modeled
        """
    )

    st.markdown("#### What the simulation reports")
    st.markdown(
        """
        - **Expected value:** the mathematical long-run average profit or loss
        - **Variance and standard deviation:** the spread of possible outcomes
        - **Probability of finishing ahead:** the share of sessions ending above the starting bankroll
        - **Probability of ruin:** the share of sessions that lose the full bankroll
        - **Monte Carlo interval:** uncertainty in the simulated estimate of average profit
        - **Outcome distribution:** the range and frequency of ending bankrolls
        """
    )

    st.markdown("#### Assignment alignment")
    rubric_frame = pd.DataFrame(
        {
            "Requirement": [
                "Probability and randomness",
                "Substantive analysis",
                "Interactivity",
                "Rigor",
                "Clarity",
                "Creativity",
            ],
            "Evidence in this app": [
                "Independent roulette outcomes generated from exact wheel probabilities",
                "Expected value, variance, risk of ruin, distributions, and strategy comparison",
                "User controls for the wheel, strategy, bankroll, bets, session size, and seed",
                "Exact mathematics, controlled simulations, reproducibility, and uncertainty intervals",
                "Definitions, direct interpretations, labeled charts, and downloadable results",
                "Connects common betting systems to both long-run expectation and tail risk",
            ],
        }
    )
    st.dataframe(rubric_frame, width="stretch", hide_index=True)

    with st.expander("Limitations and responsible interpretation"):
        st.write(
            "Simulation estimates vary from seed to seed, especially with fewer sessions. "
            "The app models standard roulette probabilities and does not claim that any "
            "strategy can predict future spins. Results are educational and should not be "
            "treated as a method for earning money through gambling."
        )

    st.markdown("#### Technical notes")
    st.write(
        "Built with Python, Streamlit, NumPy, pandas, and Plotly. The model uses "
        "vectorized session updates and stores only a small sample of full bankroll paths "
        "to keep the public app responsive."
    )

    st.caption(
        "Wheel definitions: a European wheel has 18 red, 18 black, and one green pocket; "
        "an American wheel has 18 red, 18 black, and two green pockets."
    )


# Add a restrained footer that identifies the educational purpose.
st.divider()
st.caption(
    "Roulette Risk Lab · Interactive probability and Monte Carlo analysis · "
    "Educational use only"
)
