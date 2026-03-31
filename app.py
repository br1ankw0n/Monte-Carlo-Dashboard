import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Distinct colors for multi-series plots (avoid similar blues on one axis)
_EDA_LINE_COLORS = (
    "#2563eb",
    "#ea580c",
    "#16a34a",
    "#9333ea",
    "#ca8a04",
    "#db2777",
    "#0891b2",
    "#4f46e5",
)


def plot_historical_log_returns(lr: pd.DataFrame):
    """Stacked subplots (one ticker per row) for readable daily log returns."""
    n_series = lr.shape[1]
    fig, axes = plt.subplots(
        n_series,
        1,
        figsize=(12, max(2.6, 2.3 * n_series)),
        sharex=True,
    )
    if n_series == 1:
        axes = np.array([axes])
    else:
        axes = np.asarray(axes).ravel()

    for i, (ax, col) in enumerate(zip(axes, lr.columns)):
        color = _EDA_LINE_COLORS[i % len(_EDA_LINE_COLORS)]
        ax.plot(lr.index, lr[col], color=color, linewidth=0.85, alpha=0.92)
        ax.axhline(0.0, color="#94a3b8", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_ylabel(str(col), fontsize=11, fontweight="600")
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis="both", labelsize=9)

    axes[-1].set_xlabel("Date", fontsize=10)
    fig.align_ylabels(axes)
    fig.suptitle("Daily log returns (one series per row)", fontsize=13, y=1.01)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    return fig

# Import your custom class
from montecarlo_class import CustomMonteCarlo

# Set up the page layout
st.set_page_config(page_title="Monte Carlo Portfolio Simulator", layout="wide")
st.title("📈 Monte Carlo Portfolio Simulation Dashboard")

# -------------------------------------------------------------------
# SIDEBAR: Configuration & Inputs
# -------------------------------------------------------------------
st.sidebar.header("1. Portfolio Setup")
tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "SPY, QQQ, TLT")
start_date = st.sidebar.date_input("Historical Start Date", date(2020, 1, 1))
end_date = st.sidebar.date_input("Historical End Date", date.today())

st.sidebar.header("2. Simulation Parameters")
sims = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
time_horizon = st.sidebar.number_input("Time Horizon (Days)", min_value=10, max_value=2520, value=252, step=10)
initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=1000, value=10000, step=1000)
rebalance_interval = st.sidebar.number_input("Rebalance Interval (Days)", min_value=1, max_value=252, value=21)

# Process tickers
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# -------------------------------------------------------------------
# STATE MANAGEMENT: Initialize Class & Fetch Data
# -------------------------------------------------------------------
if 'mc_engine' not in st.session_state or st.session_state.get('tickers') != tickers:
    st.session_state.mc_engine = None
    st.session_state.tickers = tickers

if st.sidebar.button("Fetch Historical Data & Init Engine"):
    with st.spinner("Fetching data and calculating parameters..."):
        engine = CustomMonteCarlo(tickers)
        engine.generate_parameters(start=start_date, end=end_date)
        st.session_state.mc_engine = engine
        st.success("Data fetched successfully!")

# -------------------------------------------------------------------
# MAIN DASHBOARD AREA
# -------------------------------------------------------------------
if st.session_state.mc_engine is not None:
    engine = st.session_state.mc_engine
    
    # Use radio + key (not st.tabs): tabs reset to the first panel on every rerun
    # (e.g. after clicking Run), which feels like a glitch. Radio selection persists.
    section = st.radio(
        "Section",
        ["Exploratory Data Analysis", "Historical Optimization", "Monte Carlo Simulations"],
        horizontal=True,
        label_visibility="collapsed",
        key="dashboard_section",
    )

    # --- EDA ---
    if section == "Exploratory Data Analysis":
        st.subheader("Historical Log Returns")
        _fig_lr = plot_historical_log_returns(engine.log_returns)
        st.pyplot(_fig_lr, use_container_width=True)
        plt.close(_fig_lr)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Annualized Mean Returns**")
            st.dataframe(engine.mean * 252)
        with col2:
            st.write("**Annualized Volatility**")
            st.dataframe(np.sqrt(engine.log_returns.var() * 252))
            
        st.subheader("Correlation Heatmap")
        engine.corr_heatmap()

    # --- Historical Optimization ---
    elif section == "Historical Optimization":
        st.subheader("Find Optimal Historical Weights")
        st.write("Runs a randomized search to find historical Sharpe and Sortino optimal allocations.")
        opt_candidates = st.number_input("Number of Candidates", min_value=1000, max_value=50000, value=5000)
        
        if st.button("Run Historical Optimization"):
            with st.spinner("Optimizing weights..."):
                engine.find_optimal_historical_weights(n_candidates=opt_candidates, interval=rebalance_interval)
                # Ensure your class prints/writes the output using st.write!

    # --- Monte Carlo Simulations ---
    else:
        st.subheader("Configure Weights & Run Simulation")
        
        # Dynamic weight inputs based on tickers
        st.write("Set Portfolio Weights (Must sum to 1.0)")
        cols = st.columns(len(tickers))
        weights = []
        for i, ticker in enumerate(tickers):
            with cols[i]:
                w = st.number_input(f"{ticker} Weight", min_value=0.0, max_value=1.0, value=1.0/len(tickers), format="%.2f")
                weights.append(w)
                
        if not np.isclose(sum(weights), 1.0):
            st.error(f"Weights currently sum to **{sum(weights):.2f}**. They must sum to **1.0**.")
        
        st.divider()
        
        sim_type = st.radio(
            "Select Distribution Model (applied to all assets)",
            ["Log-Normal", "Skew-Normal", "Student-t", "GH Skew-t (Azzalini Copula)"],
            horizontal=True
        )
        
        if st.button("Run Monte Carlo Simulation"):
            if np.isclose(sum(weights), 1.0):
                with st.spinner(f"Running {sim_type} simulation with {sims} paths..."):
                    
                    if sim_type == "Log-Normal":
                        engine.logNormMC(sims, time_horizon, initial_investment, rebalance_interval, weights)
                        results = engine.norm_results
                        portfolio_sims = engine.norm_portfolio_sims
                        
                    elif sim_type == "Skew-Normal":
                        engine.skewnormMC(sims, time_horizon, initial_investment, rebalance_interval, weights)
                        results = engine.skewnorm_results
                        portfolio_sims = engine.skewnorm_portfolio_sims
                        
                    elif sim_type == "Student-t":
                        engine.tDistMC(sims, time_horizon, initial_investment, rebalance_interval, weights)
                        results = engine.t_results
                        portfolio_sims = engine.t_portfolio_sims
                        
                    elif sim_type == "GH Skew-t (Azzalini Copula)":
                        engine.gh_azzalini_MC(sims, time_horizon, initial_investment, rebalance_interval, weights)
                        results = engine.gh_azzalini_results
                        portfolio_sims = engine.gh_azzalini_portfolio_sims

                    st.success("Simulation Complete!")
                    # Main path chart is already shown inside the MC method (st.pyplot + plt.close).

                    # --- Risk Metrics Section ---
                    st.subheader("Risk & Performance Metrics")

                    st.markdown("#### Value at Risk & Expected Shortfall")
                    engine.mcVAR(initial_investment, results, alpha=5)
                    engine.mcCTE(initial_investment, results, alpha=5)

                    g1, g2 = st.columns(2)
                    with g1:
                        st.markdown("#### Terminal wealth distribution")
                        engine.term_wealth(results)
                    with g2:
                        st.markdown("#### Max drawdown distribution")
                        engine.max_dd(portfolio_sims)

                    g3, g4 = st.columns(2)
                    with g3:
                        st.markdown("#### Sharpe ratio distribution")
                        engine.sharpe_distribution(portfolio_sims)
                    with g4:
                        st.markdown("#### Sortino ratio distribution")
                        engine.sortino_distribution(portfolio_sims)
            else:
                st.error("Please fix portfolio weights to sum to 1.0 before running.")
else:
    st.info("👈 Please enter tickers and fetch historical data from the sidebar to begin.")