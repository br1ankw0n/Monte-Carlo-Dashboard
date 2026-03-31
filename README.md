# Monte Carlo Portfolio Simulator (Streamlit)

Interactive dashboard for simulating multi-asset portfolio paths with several return-distribution models, plus historical diagnostics (correlation heatmap, Q–Q plots) and risk metrics (VaR/CTE, drawdowns, Sharpe/Sortino distributions).

---

## What’s in this project

- **`app.py`**: Streamlit UI (inputs, sections, charts).
- **`montecarlo_class.py`**: `CustomMonteCarlo` engine (data download, parameter estimation, simulations, plots, risk metrics, Q–Q plots).

---

## Features

- **Historical data fetch** from Yahoo Finance via `yfinance`
  - Computes **simple returns** and **log returns**
  - Annualized mean/volatility tables
  - Correlation heatmap

- **Historical optimization**
  - Randomized weight search (Dirichlet sampling) with periodic rebalancing
  - Reports best historical **Sharpe** and **Sortino** allocations

- **Monte Carlo simulation models**
  - **Log-Normal**: correlated Gaussian log-returns (Cholesky of covariance)
  - **Skew-Normal**: skew-normal marginals + Gaussian copula dependence
  - **Student-t**: multivariate t fit (via `sklarpy`) used to simulate heavy tails
  - **GH Skew-t (Azzalini Copula)**: GH skew-t marginals + Azzalini skew-t copula dependence
  Aas, K., & Haff, I. H. (2006). The generalized hyperbolic skew student’st-distribution. Journal of financial econometrics, 4(2), 275-309.
  Yoshiba, T. (2014). Maximum likelihood estimation of skew t-copula. Bank of Japan, The Institute of Statistical Mathematics.

- **Model diagnostics**
  - **Q–Q plots (per asset)**: compares historical daily log returns vs the fitted/theoretical marginal distribution used by the chosen simulation model.

- **Risk & performance metrics**
  - **VaR / CTE (Expected Shortfall)** on terminal wealth distribution
  - **Terminal wealth histogram**
  - **Max drawdown distribution**
  - **Sharpe / Sortino ratio distributions**

---

## Requirements

You’ll need Python 3.10+ (tested with 3.11) and the packages used by the engine/UI:

- `streamlit`
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `yfinance`
- `scipy`
- `sklarpy`
- `skewt_scipy`

If you use conda, you can install most of these via conda, and the rest via pip.

---

## How to use

### 1) Fetch historical data

- Enter tickers (comma-separated) in the sidebar (e.g. `SPY, QQQ, TLT`)
- Choose a date range
- Click **Fetch Historical Data & Init Engine**

This initializes the engine and computes historical return statistics and dependence structure.

### 2) Explore sections


- **Exploratory Data Analysis**
  - Stacked plots for daily log returns (one row per asset)
  - Annualized mean/vol tables
  - Correlation heatmap

- **Historical Optimization**
  - Click **Run Historical Optimization**
  - Increase candidates for a more exhaustive randomized search (slower)

- **Monte Carlo Simulations**
  - Choose weights (must sum to 1.0)
  - Choose distribution model
  - Click **Run Monte Carlo Simulation**
  - After the path chart, you’ll see:
    - **Q–Q plots** (per asset)
    - **Risk & performance metrics**

---

## Q–Q plots 

Each subplot compares:

- **X-axis**: theoretical quantiles from the chosen model’s marginal distribution
- **Y-axis**: empirical quantiles of historical daily log returns

If the points lie close to the red \(y=x\) line, the model’s marginal distribution matches the data well.

To avoid 1-in-a-million extremes dominating the axes, the plot uses a **0.1%–99.9%** zoom.

---

## Notes on performance

Some steps can be computationally heavy:

- **GH Skew-t (Azzalini Copula)** fitting and copula optimization can be slow.
- Q–Q plots simulate a large theoretical sample to get stable quantiles; you can reduce the sample size in `render_qq_plots()` if you need faster runs.



