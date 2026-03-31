import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.stats import kendalltau, norm, skewnorm, t, invgamma
from scipy.stats import multivariate_t
from sklarpy.multivariate import mvt_student_t
from skewt_scipy.skewt import skewt
from scipy.optimize import minimize, minimize_scalar
import scipy.optimize as optimize
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d, PchipInterpolator
import scipy.special as sp


def _sample_gh_skew_t(mu, beta, delta, nu, size, rng=None):
    """GH skew-t via normal variance–mean mixture (matches simulation in gh_azzalini_MC)."""
    shape = nu / 2.0
    scale = (delta**2) / 2.0
    w = invgamma.rvs(a=shape, scale=scale, size=size, random_state=rng)
    z = norm.rvs(size=size, random_state=rng)
    return mu + (beta * w) + (np.sqrt(w) * z)


class CustomMonteCarlo:
    def __init__(self, tickers):
        self.tickers = tickers

    def generate_parameters(self, start, end):
        # Generate parameters for simulation based on historical data, choose your own start and end date
        stockData = yf.download(self.tickers, start=start, end=end, multi_level_index=False, auto_adjust=True).Close
        stockData = stockData[self.tickers]
        self.n_assets = len(self.tickers)

        self.simple_returns = (stockData/stockData.shift(1)).dropna()
        self.log_returns = np.log(stockData/stockData.shift(1)).dropna()
        self.mean = self.log_returns.mean()
        self.var = self.log_returns.var()
        self.skew = self.log_returns.skew()
        self.covmatrix = self.log_returns.cov()
        self.corrmatrix = self.log_returns.corr()
        # self.tau, self.pval = kendalltau(self.log_returns) 

    def corr_heatmap(self):
        plt.figure(figsize=(8, 6)) # Optional: adjust figure size
        sns.heatmap(self.corrmatrix, annot=True, linewidths=.5)
        plt.title('Correlation Heatmap of Assets')
        
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    def find_optimal_historical_weights(self, n_candidates=5000, interval=21, rf=0.0, seed=42, dirichlet_alpha=1.0):
        R = self.simple_returns.to_numpy(dtype=float)
        T,n = R.shape
        rng = np.random.default_rng(seed)
        alpha = np.full(n, dirichlet_alpha, dtype=float)

        reb = np.zeros(T, dtype=bool)
        reb[::interval] = True

        initial = 1.0
        best_sharpe_weights = None
        best_sharpe = -np.inf
        best_sortino_weights = None
        best_sortino = -np.inf

        for _ in range(n_candidates):
            w = rng.dirichlet(alpha)
            V = np.empty((T+1, n), dtype=float)
            V[0] = initial * w

            t0 = 0
            while t0 < T:
                if reb[t0] and t0 !=0:
                    V[t0] = V[t0].sum() * w

                t1 = min(((t0//interval) + 1) * interval, T)

                seg = R[t0:t1]
                growth = np.cumprod(seg, axis=0)
                V[t0 + 1 : t1 + 1] = V[t0] * growth
                t0 = t1

            path = V.sum(axis=1)
            path_returns = np.log(path[1:]/path[:-1])
            mean = path_returns.mean() * 252
            vol = np.sqrt(path_returns.var() * 252)
            
            def calculate_downside_vol(returns):
                square_sum = 0
                for r in returns:
                    square_sum += min(r, 0)**2
                square_sum /= len(returns)
                downside_dev = np.sqrt(square_sum)
                return np.sqrt(252) * downside_dev

            downside_vol = calculate_downside_vol(path_returns)
            
            sharpe = mean/vol
            sortino = mean/downside_vol
            if sharpe > best_sharpe:
                best_sharpe_weights = w.copy()
                best_sharpe = sharpe.copy()
            if sortino > best_sortino:
                best_sortino_weights = w.copy()
                best_sortino = sortino.copy()

        st.write(f"**Sharpe-Optimal Weights:** {np.round(best_sharpe_weights, 4)}")
        st.write(f"**Sharpe Ratio:** {np.round(best_sharpe, 4)}")
        st.write(f"**Sortino-Optimal Weights:** {np.round(best_sortino_weights, 4)}")
        st.write(f"**Sortino Ratio:** {np.round(best_sortino, 4)}")
                                                
    def logNormMC(self, sims, time, initial, interval, weights):
        if not np.isclose(np.sum(weights), 1.0) or len(weights) != len(self.tickers):
            raise ValueError("Weights must match number of tickers and sum to 1.")
            
        covMatrix = self.covmatrix
        
        mc_sims = sims
        T = time 
        
        meanM = np.full(shape=(T, len(weights)), fill_value=self.mean)
        meanM = meanM.T
        
        self.norm_portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
        
        initialPortfolio = initial
        interval = interval
        weights = np.array(weights) 
        
        for j in range(0, mc_sims):
            Z = np.random.normal(size=(T, len(weights)))
            L = np.linalg.cholesky(covMatrix)
            dailyReturns = meanM + L @ Z.T
            current_port = initialPortfolio * weights

            for i in range(0, T):
                if i >0 and i% interval == 0:
                    current_port = np.sum(current_port) * weights
                current_port = current_port * np.exp(dailyReturns[:,i])
                self.norm_portfolio_sims[i,j] = np.sum(current_port)
        
        self.norm_portfolio_sims = np.vstack([np.full((1, sims), initial), self.norm_portfolio_sims])
        self.norm_results = pd.Series(self.norm_portfolio_sims[-1, :])
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.norm_portfolio_sims)
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Days')
        allocations = ', '.join([f"{w:.0%} {ticker}" for w, ticker in zip(weights, self.tickers)])
        plt.title(f"MC simulation of Log-Normal stock portfolio ({allocations})")
        
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    def skewnormMC(self, sims, time, initial, interval, weights): 
        X = self.log_returns.to_numpy()
        t_len, d = X.shape
        marg_params = []
        for i in range(d):
            a, loc, scale = skewnorm.fit(X[:, i])
            marg_params.append((a, loc, scale))
        self._skewnorm_marg_params = marg_params

        U = np.zeros_like(X)
        for i, (a, loc, scale) in enumerate(marg_params):
            U[:, i] = skewnorm.cdf(X[:, i], a, loc=loc, scale=scale)

        eps = 1e-10
        U = np.clip(U, eps, 1-eps)

        Z = norm.ppf(U)
        copula_corr_matrix = np.corrcoef(Z, rowvar=False)
        L = np.linalg.cholesky(copula_corr_matrix)
        
        mc_sims = sims
        T = time 
        
        self.skewnorm_portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
        interval = interval
        weights = np.array(weights) 

        for j in range(0, mc_sims):
            G = np.random.normal(size=(T, d))
            Y = G @ L.T
            U_star = norm.cdf(Y)
            U_star = np.clip(U_star, eps, 1 - eps)

            X_star = np.zeros_like(U_star)
            for i, (a, loc, scale) in enumerate(marg_params):
                X_star[:, i] = skewnorm.ppf(U_star[:, i], a, loc=loc, scale=scale)
            
            dailyReturns = X_star.T
            current_port = initial * weights
            
            for i in range(0, T):
                if i >0 and i%interval == 0:
                    current_port = np.sum(current_port) * weights
                current_port = current_port * np.exp(dailyReturns[:, i])
                self.skewnorm_portfolio_sims[i,j] = np.sum(current_port)

        self.skewnorm_portfolio_sims = np.vstack([np.full((1, sims), initial), self.skewnorm_portfolio_sims])
        self.skewnorm_results = pd.Series(self.skewnorm_portfolio_sims[-1, :])
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.skewnorm_portfolio_sims)
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Days')
        allocations = ', '.join([f"{w:.0%} {ticker}" for w, ticker in zip(weights, self.tickers)])
        plt.title(f"MC simulation of Skew-Normal stock portfolio ({allocations})")
        
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    def tDistMC(self, sims, time, initial, interval, weights):
        multivariate_t_fitted = mvt_student_t.fit(self.log_returns, method="mle")
        params = multivariate_t_fitted.params
        # sklarpy: `params` may be a dict, or `.to_dict` may be a dict (not a method)
        if isinstance(params, dict):
            p = params
        else:
            td = getattr(params, "to_dict", None)
            if callable(td):
                p = td()
            elif isinstance(td, dict):
                p = td
            else:
                p = dict(params)

        nu = p["dof"]
        mu = np.asarray(p["loc"]).ravel()
        shape = np.asarray(p["shape"])
        self._mv_t_params = p
        
        mc_sims = sims
        T = time 
        
        meanM = np.full(shape=(T, len(weights)), fill_value=mu)
        meanM = meanM.T
        
        self.t_portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
        
        initialPortfolio = initial
        interval = interval
        weights = np.array(weights) 
        
        for j in range(0, mc_sims):
            Z = np.random.normal(size=(T, len(weights)))
            L = np.linalg.cholesky(shape)
            S = np.random.chisquare(nu, size = T)
            dailyReturns = meanM + (L @ Z.T)*np.sqrt(nu/S)
            current_port = initialPortfolio * weights

            for i in range(0, T):
                if i >0 and i% interval == 0:
                    current_port = np.sum(current_port) * weights
                current_port = current_port * np.exp(dailyReturns[:,i])
                self.t_portfolio_sims[i,j] = np.sum(current_port)
        
        self.t_portfolio_sims = np.vstack([np.full((1, sims), initial), self.t_portfolio_sims])
        self.t_results = pd.Series(self.t_portfolio_sims[-1, :])
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.t_portfolio_sims)
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Days')
        allocations = ', '.join([f"{w:.0%} {ticker}" for w, ticker in zip(weights, self.tickers)])
        plt.title(f"MC simulation of Student-t stock portfolio ({allocations})")
        
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    def mcVAR(self, initial, results, alpha):
        if isinstance(results, pd.Series):
            var_pct = np.round((initial - np.percentile(results, alpha))*100/initial,2)
            st.write(f"**Value at Risk ({100-alpha}% VaR):** {var_pct}%")
        else:
            raise TypeError("Expected a pandas data series")

    def mcCTE(self, initial, results, alpha):
        if isinstance(results, pd.Series):
            belowVAR = results <= np.percentile(results, alpha)
            cte_pct = np.round((initial - results[belowVAR].mean())*100/initial,2)
            st.write(f"**Conditional Tail Expectation ({100-alpha}% CTE):** {cte_pct}%")
        else:
            raise TypeError("Expected a pandas data series")
        
    def term_wealth(self, results):
        terminal_wealth = pd.Series(results)
        plt.figure()
        plt.hist(terminal_wealth, bins=50)
        
        fig = plt.gcf()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.write(f"**Median Wealth:** ${np.round(terminal_wealth.median(),2)}")

    def max_dd(self, portfolio_sims):
        max_dd_dist = []
        paths = portfolio_sims.T
        for path in paths:
            path = pd.Series(path)
            cummax = path.cummax()
            max_dd = 0
            for i in range(len(path)):
                max_dd = max(((cummax.iloc[i] - path.iloc[i])/cummax.iloc[i]), max_dd)
            max_dd_dist.append(max_dd)

        max_dd_dist = pd.Series(max_dd_dist)
        plt.figure()
        plt.hist(max_dd_dist)
        
        fig = plt.gcf()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.write(f"**Median Max Drawdown:** {np.round(max_dd_dist.median()*100,2)}%")
        st.write(f"**Worst Max Drawdown:** {np.round(max(max_dd_dist)*100,2)}%")

    def sharpe_distribution(self, portfolio_sims):
        sims = portfolio_sims.T
        dailys = sims[:, 1:]/sims[:, :-1] 
        dailys = np.log(dailys)

        means = dailys.mean(axis=1, keepdims=True)
        means *= 252

        variances = dailys.var(axis=1, keepdims=True)
        vols = np.sqrt(variances*252)

        sharpes = means/vols
        sharpes = pd.Series(sharpes.ravel())
        
        plt.figure()
        plt.hist(sharpes)
        
        fig = plt.gcf()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.write(f"**Median Sharpe Ratio:** {np.round(sharpes.median(), 2)}") 
    
    def sortino_distribution(self, portfolio_sims):
        sims = portfolio_sims.T
        dailys = sims[:, 1:]/sims[:, :-1] 
        dailys = np.log(dailys)

        means = dailys.mean(axis=1, keepdims=True)
        means *= 252

        downside_dailys = np.minimum(dailys, 0)
        downside_devs = np.sqrt((downside_dailys**2).mean(axis=1, keepdims=True))
        downside_vols = downside_devs * np.sqrt(252)

        sortinos = means/downside_vols
        sortinos = pd.Series(sortinos.ravel())
        
        plt.figure()
        plt.hist(sortinos)
        
        fig = plt.gcf()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        st.write(f"**Median Sortino Ratio:** {np.round(sortinos.median(), 2)}")

    def render_qq_plots(self, sim_type_label, theoretical_sample_size=400_000):
        """
        Q–Q plots: historical log returns vs. the same marginal law used for the chosen MC model.
        """
        lr = self.log_returns
        d = lr.shape[1]
        ncol = 2 if d > 1 else 1
        nrow = int(np.ceil(d / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 6.2 * nrow))
        if d == 1:
            ax_list = [axes]
        else:
            ax_list = np.asarray(axes).ravel()

        rng = np.random.default_rng(42)

        for j in range(d):
            ax = ax_list[j]
            ticker = str(self.tickers[j])
            real = np.asarray(lr.iloc[:, j], dtype=float)
            real = real[~np.isnan(real)]
            real_sorted = np.sort(real)
            n_real = len(real_sorted)
            pct = np.linspace(0.0, 100.0, n_real)

            if sim_type_label == "Log-Normal":
                mu = float(self.mean.iloc[j])
                sig = float(np.sqrt(self.var.iloc[j]))
                theoretical = rng.normal(mu, sig, size=theoretical_sample_size)
                xlab = "Theoretical Gaussian quantiles"
                ttl = f"{ticker} vs Gaussian (log-returns)"

            elif sim_type_label == "Skew-Normal":
                a, loc, scale = self._skewnorm_marg_params[j]
                theoretical = skewnorm.rvs(a, loc=loc, scale=scale, size=theoretical_sample_size, random_state=rng)
                xlab = "Theoretical skew-normal quantiles"
                ttl = f"{ticker} vs fitted skew-normal"

            elif sim_type_label == "Student-t":
                p = self._mv_t_params
                nu = float(p["dof"])
                mu_v = np.asarray(p["loc"]).ravel()
                sh = np.asarray(p["shape"])
                mu_j = float(mu_v[j])
                sj = float(np.sqrt(sh[j, j]))
                theoretical = stats.t.rvs(
                    df=nu,
                    loc=mu_j,
                    scale=sj,
                    size=theoretical_sample_size,
                    random_state=rng,
                )
                xlab = "Theoretical Student-t quantiles"
                ttl = f"{ticker} vs marginal Student-t (MV fit)"

            elif sim_type_label == "GH Skew-t (Azzalini Copula)":
                par = self._gh_marginal_params[j]
                theoretical = _sample_gh_skew_t(
                    par["mu"],
                    par["beta"],
                    par["delta"],
                    par["nu"],
                    size=theoretical_sample_size,
                    rng=rng,
                )
                xlab = "Theoretical GH skew-t quantiles"
                ttl = f"{ticker} vs fitted GH skew-t"

            else:
                raise ValueError(f"Unknown sim type for Q-Q: {sim_type_label}")

            theoretical_quantiles = np.percentile(theoretical, pct)
            ax.scatter(
                theoretical_quantiles,
                real_sorted,
                alpha=0.45,
                edgecolors="k",
                linewidths=0.3,
                s=18,
            )
            lo_t, hi_t = np.percentile(theoretical, [0.1, 99.9])
            lo_r, hi_r = np.percentile(real_sorted, [0.1, 99.9])
            vmin = min(lo_t, lo_r)
            vmax = max(hi_t, hi_r)
            ax.plot([vmin, vmax], [vmin, vmax], color="red", linestyle="--", linewidth=2, label="y = x")
            ax.set_title(ttl, fontsize=11, fontweight="600")
            ax.set_xlabel(xlab, fontsize=10)
            ax.set_ylabel("Historical log-return quantiles", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", fontsize=8)

            ax.set_xlim(lo_t, hi_t)
            ax.set_ylim(lo_r, hi_r)

        for k in range(d, len(ax_list)):
            ax_list[k].set_visible(False)

        fig.suptitle(
            "Q–Q plots: model vs historical daily log returns",
            fontsize=13,
            fontweight="600",
            y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    def gh_azzalini_MC(self, sims, time, initial, interval, weights):
      
        def initialize_params(x):
            mu_init = np.mean(x)
            beta_init = 0.0       
            delta_init = np.std(x)
            nu_init = 4.0         
            return mu_init, beta_init, delta_init, nu_init
    
        def e_step(x, mu, beta, delta, nu):
            q_x = np.sqrt(delta**2 + (x - mu)**2)
            abs_beta = np.abs(beta)
            
            if abs_beta < 1e-5:
                xi = (q_x**2) / (nu - 1)
                rho = (nu + 1) / (q_x**2)
                chi = np.log(q_x**2 / 2) - sp.digamma((nu + 1) / 2)
            else:
                arg = abs_beta * q_x
                k_num_xi  = sp.kve((1 - nu) / 2, arg)
                k_num_rho = sp.kve((nu + 3) / 2, arg)
                k_den     = sp.kve((nu + 1) / 2, arg)
                
                eps = 1e-5
                order = (nu + 1) / 2
                k_den_plus = sp.kve(order + eps, arg)
                k_den_minus = sp.kve(order - eps, arg)
                dk_dorder_e = (k_den_plus - k_den_minus) / (2 * eps)
                
                xi = (q_x / abs_beta) * (k_num_xi / k_den)
                rho = (abs_beta / q_x) * (k_num_rho / k_den)
                chi = np.log(q_x / abs_beta) - (1 / k_den) * dk_dorder_e
                
            return xi, rho, chi
        
        def m_step(x, xi, rho, chi):
            n = len(x)
            sum_rho = np.sum(rho)
            sum_xi = np.sum(xi)
            sum_x = np.sum(x)
            sum_x_rho = np.sum(x * rho)
            
            denominator = max(sum_rho * sum_xi - n**2, 1e-8)
                
            mu = (sum_x_rho * sum_xi - sum_x * n) / denominator
            beta = (sum_rho * sum_x - sum_x_rho * n) / denominator
            
            mean_rho = np.mean(rho)
            mean_chi = np.mean(chi)
            
            def nu_objective(nu_val):
                obj = -1.0 * (
                    (nu_val / 2) * np.log(nu_val / (2 * mean_rho)) 
                    - sp.gammaln(nu_val / 2) 
                    - (nu_val / 2 + 1) * mean_chi 
                    - (nu_val / 2)
                )
                return obj
        
            res = minimize_scalar(nu_objective, bounds=(2.01, 35.0), method='bounded')
            nu = res.x
            delta = np.sqrt(nu / mean_rho)
            
            return mu, beta, delta, nu
        
        def fit_gh_skew_t(x, max_iter=100, tol=1e-5):
            x = np.asarray(x, dtype=np.float64)
            mu, beta, delta, nu = initialize_params(x)
            # st.write(f"Initial: mu={mu:.4f}, beta={beta:.4f}, delta={delta:.4f}, nu={nu:.4f}")
            
            for i in range(max_iter):
                mu_old, beta_old, delta_old, nu_old = mu, beta, delta, nu
                xi, rho, chi = e_step(x, mu, beta, delta, nu)
                mu, beta, delta, nu = m_step(x, xi, rho, chi)
                
                param_diff = (np.abs(mu - mu_old) + 
                              np.abs(beta - beta_old) + 
                              np.abs(delta - delta_old) + 
                              np.abs(nu - nu_old))
                              
                if param_diff < tol:
                    # st.write(f"Converged successfully in {i+1} iterations.")
                    break
                    
            return {'mu': mu, 'beta': beta, 'delta': delta, 'nu': nu}
    
        def generate_gh_skew_t_sample(mu, beta, delta, nu, size=1000000):
            shape = nu / 2.0
            scale = (delta ** 2) / 2.0
            w = invgamma.rvs(a=shape, scale=scale, size=size)
            z = norm.rvs(size=size)
            x_sim = mu + (beta * w) + (np.sqrt(w) * z)
            return x_sim

        def transform_to_uniforms(real_data, mu, beta, delta, nu, num_sim=1000000):
            theoretical_data = generate_gh_skew_t_sample(mu, beta, delta, nu, size=num_sim)
            x_theoretical = np.sort(theoretical_data)
            y_cdf = np.linspace(0, 1, num_sim)
            gh_skew_t_cdf = interp1d(x_theoretical, y_cdf, kind='linear', 
                                     bounds_error=False, fill_value=(0.0, 1.0))
            uniform_data = gh_skew_t_cdf(real_data)
            return uniform_data

        def fit_gh_marginals_and_transform(returns_data):
            n_assets = returns_data.shape[1]
            uniform_data = np.zeros_like(returns_data)
            marginal_params = []
            
            for j in range(n_assets):
                asset_returns = returns_data.iloc[:, j]
                params = fit_gh_skew_t(asset_returns)
                marginal_params.append(params)
                uniform_data[:, j] = transform_to_uniforms(asset_returns, **marginal_params[j], num_sim=1000000)
                
            return uniform_data, marginal_params
    
        st.info("Fitting GH Skew-t marginals and mapping to uniform space...")
        uniform_data, marginal_params = fit_gh_marginals_and_transform(self.log_returns)
        self._gh_marginal_params = marginal_params
        
        def angles_to_correlation_matrix(angles, d):
            L = np.zeros((d, d))
            L[0, 0] = 1.0
            angle_idx = 0
            
            for i in range(1, d):
                for j in range(i):
                    prod = 1.0
                    for k in range(j):
                        prod *= np.sin(angles[angle_idx - j + k])
                    L[i, j] = np.cos(angles[angle_idx]) * prod
                    angle_idx += 1
                
                prod = 1.0
                for k in range(i):
                    prod *= np.sin(angles[angle_idx - i + k])
                L[i, i] = prod
                
            return L @ L.T
    
        def azzalini_univariate_pdf(x, alpha, nu):
            t_pdf = stats.t.pdf(x, df=nu)
            arg = alpha * x * np.sqrt((nu + 1) / (nu + x**2))
            t_cdf = stats.t.cdf(arg, df=nu + 1)
            return 2 * t_pdf * t_cdf
        
        def fit_azzalini_copula(uniform_data):
            N, d = uniform_data.shape
            num_angles = int(d * (d - 1) / 2)
        
            def build_fast_quantile_interpolators(alpha_vec, nu):
                interpolators = []
                x_grid = np.linspace(-15, 15, 2000) 
                
                for alpha in alpha_vec:
                    pdf_grid = azzalini_univariate_pdf(x_grid, alpha, nu)
                    cdf_grid = cumulative_trapezoid(pdf_grid, x_grid, initial=0)
                    cdf_grid = cdf_grid / cdf_grid[-1] 
                    unique_idx = np.unique(cdf_grid, return_index=True)[1]
                    cdf_clean = cdf_grid[unique_idx]
                    x_clean = x_grid[unique_idx]
                    interpolators.append(PchipInterpolator(cdf_clean, x_clean))
                    
                return interpolators
        
            def copula_log_likelihood(params):
                nu = params[0]
                alpha = params[1:d+1]            
                angles = params[d+1:]            
                
                if nu <= 2.01:
                    return 1e10 
                    
                Psi = angles_to_correlation_matrix(angles, d)
                
                try:
                    interpolators = build_fast_quantile_interpolators(alpha, nu)
                    x_matrix = np.zeros((N, d))
                    for j in range(d):
                        x_matrix[:, j] = interpolators[j](uniform_data[:, j])
                        
                    multi_t_pdf = stats.multivariate_t.pdf(x_matrix, loc=np.zeros(d), shape=Psi, df=nu)
                    inv_Psi = np.linalg.inv(Psi)
                    mahalanobis_sq = np.sum(x_matrix @ inv_Psi * x_matrix, axis=1)
                    skew_arg = (x_matrix @ alpha) * np.sqrt((nu + d) / (nu + mahalanobis_sq))
                    multi_t_cdf = stats.t.cdf(skew_arg, df=nu + d)
                    
                    joint_density = 2 * multi_t_pdf * multi_t_cdf
                    
                    marginal_densities = np.ones(N)
                    for j in range(d):
                        marginal_densities *= azzalini_univariate_pdf(x_matrix[:, j], alpha[j], nu)
                    
                    copula_density = joint_density / marginal_densities
                    copula_density = np.clip(copula_density, 1e-10, np.inf)
                    log_lik = np.sum(np.log(copula_density))
                    
                    return -log_lik 
                    
                except Exception as e:
                    return 1e10
        
            st.info(f"Starting MLE Optimization for {d}D Copula... This may take a minute.")
            
            initial_guess = np.concatenate(([5.0], np.zeros(d), np.full(num_angles, np.pi/4)))
            bounds = [(2.01, 30)] + [(None, None)] * d + [(-np.pi, np.pi)] * num_angles
            
            result = optimize.minimize(
                copula_log_likelihood, 
                initial_guess, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'disp': False, 'ftol': 1e-4} 
            )
            
            final_nu = result.x[0]
            final_alpha = result.x[1:d+1]
            final_Psi = angles_to_correlation_matrix(result.x[d+1:], d)
            
            st.success("Copula Optimization Complete!")
            st.write(f"**Degrees of Freedom (nu):** {final_nu:.2f}")
            st.write(f"**Skewness Vector (alpha):** {np.round(final_alpha, 4)}")
            
            return {"nu": final_nu, "alpha": final_alpha, "Psi": final_Psi}
    
        copula_params = fit_azzalini_copula(uniform_data)
        
        st.info(f"Initializing engine: {sims:,} sims over {time} days...")
        
        nu = copula_params["nu"]
        alpha = copula_params["alpha"]
        Psi = copula_params["Psi"]
        d = len(alpha)
        weights = np.array(weights)
        
        N_total = sims * time
        delta_copula = alpha / np.sqrt(1 + alpha**2)
        
        st.write("Simulating joint copula structure...")
        
        V = np.random.gamma(shape=nu/2, scale=2/nu, size=(N_total, 1))
        Z0 = np.random.normal(0, 1, size=(N_total, 1))
        Z = np.random.multivariate_normal(np.zeros(d), Psi, size=N_total)
        
        Y = delta_copula * np.abs(Z0) + np.sqrt(1 - delta_copula**2) * Z
        X = Y / np.sqrt(V)
        
        st.write("Mapping marginals through PIT...")
        simulated_returns_flat = np.zeros((N_total, d))
        
        x_grid = np.linspace(-25, 25, 3000)
        y_cdf_theory = np.linspace(0, 1, 1000000) 
        
        for j in range(d):
            pdf_grid = azzalini_univariate_pdf(x_grid, alpha[j], nu)
            cdf_grid = cumulative_trapezoid(pdf_grid, x_grid, initial=0)
            cdf_grid = cdf_grid / cdf_grid[-1]
            unique_idx = np.unique(cdf_grid, return_index=True)[1]
            
            x_to_u = PchipInterpolator(x_grid[unique_idx], cdf_grid[unique_idx])
            U_j = x_to_u(X[:, j])
            U_j = np.clip(U_j, 1e-5, 1 - 1e-5)
            
            m_params = marginal_params[j]
            theoretical_data = generate_gh_skew_t_sample(
                m_params['mu'], m_params['beta'], m_params['delta'], m_params['nu'], size=1000000
            )
            theoretical_sorted = np.sort(theoretical_data)
            
            u_to_returns = interp1d(y_cdf_theory, theoretical_sorted, kind='linear', 
                                    bounds_error=False, fill_value=(theoretical_sorted[0], theoretical_sorted[-1]))
            
            simulated_returns_flat[:, j] = u_to_returns(U_j)
            
        st.write("Calculating portfolio paths...")
        daily_log_returns = simulated_returns_flat.reshape(sims, time, d)
        daily_returns_multiplier = np.exp(daily_log_returns)
        self.gh_azzalini_daily_returns_multiplier = daily_returns_multiplier.copy()
        
        self.gh_azzalini_portfolio_sims = np.zeros((time, sims))
        current_port = np.full((sims, d), initial) * weights
        
        for i in range(time):
            if i > 0 and i % interval == 0:
                total_value = np.sum(current_port, axis=1, keepdims=True)
                current_port = total_value * weights
                
            current_port = current_port * daily_returns_multiplier[:, i, :]
            self.gh_azzalini_portfolio_sims[i, :] = np.sum(current_port, axis=1)
            
        self.gh_azzalini_portfolio_sims = np.vstack([np.full((1, sims), initial), self.gh_azzalini_portfolio_sims])
        self.gh_azzalini_results = pd.Series(self.gh_azzalini_portfolio_sims[-1, :])
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.gh_azzalini_portfolio_sims)
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Days')
        allocations = ', '.join([f"{w:.0%} {ticker}" for w, ticker in zip(weights, self.tickers)])
        plt.title(f"Azzalini-GH MC Simulation ({allocations})\nShowing {sims} sample paths")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    def calculate_universal_kelly_weights(self, daily_returns_multiplier_tensor):
        sims, time, d = daily_returns_multiplier_tensor.shape
        
        def objective(w):
            daily_port_returns = np.sum(daily_returns_multiplier_tensor * w, axis=2)
            terminal_wealth = np.prod(daily_port_returns, axis=1)
            safe_wealth = np.clip(terminal_wealth, 1e-10, np.inf)
            expected_log_wealth = np.mean(np.log(safe_wealth))
            return -expected_log_wealth * 100
    
        init_guess = np.ones(d) / d
        bounds = [(0.0, 1) for _ in range(d)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        result = minimize(
            objective, 
            init_guess, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'ftol': 1e-8, 'disp': False} 
        )
        
        optimal_weights = result.x
        optimal_weights = np.round(optimal_weights, 4)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        st.write(f"**Optimal Kelly Weights Found:** {optimal_weights}")
        return optimal_weights