"""
Multi-Strategy Quantitative Portfolio Optimization Engine.

Implements core quantitative asset allocation paradigms:
1. Markowitz Modern Portfolio Theory (Max Sharpe Ratio via SciPy SLSQP)
2. Rockafellar-Uryasev (2000) Convex Conditional Value-at-Risk (Min CVaR / Expected Shortfall via SciPy HiGHS LP)
3. Black-Litterman Bayesian Asset Allocation (Prior Equilibrium + Views Matrix Blending with Quadratic Utility)
4. Equal Risk Contribution (Risk Parity via Marginal Risk Decomposition)
5. Discrete Capital Allocation (Integer Share Execution)
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog


class MultiStrategyPortfolioOptimizer:
    """
    Quantitative Portfolio Optimizer supporting Markowitz, Rockafellar-Uryasev CVaR LP,
    Black-Litterman Bayesian asset allocation, and Risk Parity.
    """

    def __init__(self, data_dict, risk_free_rate=0.04):
        self.data = data_dict
        self.prices = data_dict['prices']
        self.daily_returns = data_dict['daily_returns']
        self.expected_returns = data_dict['expected_returns']
        self.cov_matrix = data_dict['shrunk_covariance']
        self.tickers = data_dict['tickers']
        self.num_assets = len(self.tickers)
        self.rf = risk_free_rate

    def optimize_max_sharpe(self):
        """
        Solves Markowitz Maximum Sharpe Ratio optimization via SLSQP subject to full investment and long-only bounds.
        """
        mu = self.expected_returns.values
        cov = self.cov_matrix.values

        def neg_sharpe(weights):
            p_ret = np.dot(weights, mu)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            if p_vol == 0:
                return 0.0
            return -(p_ret - self.rf) / p_vol

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        w0 = np.ones(self.num_assets) / self.num_assets

        res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = res.x

        p_ret = float(np.dot(weights, mu))
        p_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
        sharpe = (p_ret - self.rf) / p_vol if p_vol > 0 else 0.0

        return {
            'strategy': 'Markowitz Max Sharpe',
            'weights': {t: round(float(w), 4) for t, w in zip(self.tickers, weights)},
            'expected_annual_return': round(p_ret, 4),
            'annual_volatility': round(p_vol, 4),
            'sharpe_ratio': round(sharpe, 4)
        }

    def optimize_convex_cvar(self, alpha=0.95):
        """
        Solves Rockafellar-Uryasev (2000) Convex Conditional Value-at-Risk (CVaR) minimization via SciPy HiGHS LP.
        Formulation:
            min_{w, zeta, u}  zeta + 1 / ((1 - alpha) * T) * sum_{t=1}^T u_t
            s.t.  u_t >= -w^T r_t - zeta  <=>  -sum_i w_i r_{t, i} - zeta - u_t <= 0,  forall t
                  u_t >= 0,  forall t
                  sum_i w_i = 1
                  w_i >= 0,  forall i
        """
        R = self.daily_returns.values  # (T, N)
        T, N = R.shape

        # Variables: [w_1..w_N (N), zeta (1), u_1..u_T (T)] -> Total N + 1 + T
        num_vars = N + 1 + T
        c = np.zeros(num_vars)
        c[N] = 1.0  # zeta coefficient
        c[N + 1:] = 1.0 / ((1.0 - alpha) * T)  # u_t coefficients

        # Variable Bounds: w_i in [0, 1], zeta in [-inf, inf], u_t in [0, inf)
        bounds = [(0.0, 1.0)] * N + [(None, None)] + [(0.0, None)] * T

        # Inequality constraints: -w^T r_t - zeta - u_t <= 0
        A_ub = np.zeros((T, num_vars))
        b_ub = np.zeros(T)
        for t in range(T):
            A_ub[t, :N] = -R[t, :]
            A_ub[t, N] = -1.0
            A_ub[t, N + 1 + t] = -1.0

        # Equality constraint: sum_i w_i = 1
        A_eq = np.zeros((1, num_vars))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not res.success:
            raise RuntimeError(f"HiGHS Convex CVaR LP Solver failed: {res.message}")

        weights = res.x[:N]
        zeta = float(res.x[N])
        cvar_daily = float(res.fun)
        cvar_annual = cvar_daily * math.sqrt(252.0)

        cov = self.cov_matrix.values
        mu = self.expected_returns.values
        p_ret = float(np.dot(weights, mu))
        p_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))

        return {
            'strategy': 'Rockafellar-Uryasev Convex Min CVaR (HiGHS LP)',
            'solver_status': res.message,
            'weights': {t: round(float(w), 4) for t, w in zip(self.tickers, weights)},
            'expected_annual_return': round(p_ret, 4),
            'annual_volatility': round(p_vol, 4),
            'annualized_cvar_95': round(cvar_annual, 4),
            'var_threshold_zeta': round(zeta * math.sqrt(252.0), 4)
        }

    def optimize_black_litterman(self, P=None, Q=None, delta=2.5, tau=0.05, market_weights=None):
        """
        Solves Black-Litterman Bayesian Asset Allocation blending CAPM equilibrium prior returns
        with investor views using standard Mean-Variance Quadratic Utility optimization.
        Args:
            P: (k x N) Pick matrix linking views to assets (if None, uses default 2 views)
            Q: (k,) Vector of view returns (if None, uses default [0.02, 0.03])
            delta: risk aversion coefficient (default 2.5)
            tau: scaling factor for prior covariance uncertainty (default 0.05)
            market_weights: benchmark market portfolio weights (default [0.25, 0.25, 0.15, 0.10, 0.10, 0.15])
        """
        cov = self.cov_matrix.values
        N = self.num_assets

        if market_weights is None:
            w_mkt = np.array([0.25, 0.25, 0.15, 0.10, 0.10, 0.15])
        else:
            w_mkt = np.array(market_weights)

        # Implied Equilibrium Prior Returns: Pi = delta * Cov @ w_mkt
        Pi = delta * np.dot(cov, w_mkt)

        # Handle zero-views case (collapse to prior)
        if P is None or len(P) == 0:
            mu_BL = Pi
            cov_BL = cov
        else:
            P = np.array(P)
            Q = np.array(Q)

            # Uncertainty Matrix: Omega = diag(P @ (tau * Cov) @ P^T) (He & Litterman specification)
            Omega = np.diag(np.diag(P @ (tau * cov) @ P.T))

            # Posterior Expected Returns & Covariance:
            tau_cov_inv = np.linalg.inv(tau * cov)
            omega_inv = np.linalg.inv(Omega)

            M = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
            mu_BL = M @ (tau_cov_inv @ Pi + P.T @ omega_inv @ Q)
            cov_BL = cov + M

        # Canonical Black-Litterman Quadratic Utility Optimization:
        # min_{w}  0.5 * delta * w^T Cov_BL w - w^T mu_BL
        # s.t.  sum(w) = 1,  w_i >= 0
        def bl_utility_obj(w):
            return 0.5 * delta * np.dot(w.T, np.dot(cov_BL, w)) - np.dot(w, mu_BL)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))

        res = minimize(bl_utility_obj, w_mkt, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = res.x

        p_ret = float(np.dot(weights, mu_BL))
        p_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
        sharpe = (p_ret - self.rf) / p_vol if p_vol > 0 else 0.0

        return {
            'strategy': 'Black-Litterman Bayesian Allocation',
            'weights': {t: round(float(w), 4) for t, w in zip(self.tickers, weights)},
            'expected_annual_return': round(p_ret, 4),
            'annual_volatility': round(p_vol, 4),
            'sharpe_ratio': round(sharpe, 4),
            'prior_equilibrium_returns': {t: round(float(r), 4) for t, r in zip(self.tickers, Pi)},
            'posterior_bl_returns': {t: round(float(r), 4) for t, r in zip(self.tickers, mu_BL)},
            'benchmark_weights': {t: round(float(w), 4) for t, w in zip(self.tickers, w_mkt)}
        }

    def optimize_risk_parity(self):
        """
        Solves Equal Risk Contribution (Risk Parity) allocation via SQP optimization.
        """
        cov = self.cov_matrix.values
        target = np.full(self.num_assets, 1.0 / self.num_assets)

        def risk_parity_obj(weights):
            weights = np.array(weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            if p_vol == 0:
                return 0.0
            mcr = np.dot(cov, weights) / p_vol
            rc = weights * mcr
            rc_pct = rc / np.sum(rc)
            return np.sum((rc_pct - target) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.01, 1.0) for _ in range(self.num_assets))
        w0 = np.ones(self.num_assets) / self.num_assets

        res = minimize(risk_parity_obj, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = res.x

        mu = self.expected_returns.values
        p_ret = float(np.dot(weights, mu))
        p_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
        sharpe = (p_ret - self.rf) / p_vol if p_vol > 0 else 0.0

        return {
            'strategy': 'Equal Risk Parity (ERC)',
            'weights': {t: round(float(w), 4) for t, w in zip(self.tickers, weights)},
            'expected_annual_return': round(p_ret, 4),
            'annual_volatility': round(p_vol, 4),
            'sharpe_ratio': round(sharpe, 4)
        }

    def compute_discrete_allocation(self, weights_dict, total_capital=100000.0):
        """
        Computes exact integer shares to purchase for a given cash portfolio budget.
        """
        latest_prices = self.prices.iloc[-1].to_dict()
        shares = {}
        allocated_cash = 0.0

        for t in self.tickers:
            w = weights_dict.get(t, 0.0)
            target_val = total_capital * w
            price = latest_prices[t]
            num_shares = int(np.floor(target_val / price))
            shares[t] = num_shares
            allocated_cash += num_shares * price

        leftover_cash = total_capital - allocated_cash
        return {
            'total_portfolio_value': total_capital,
            'allocated_shares': shares,
            'allocated_capital': round(allocated_cash, 2),
            'leftover_cash': round(leftover_cash, 2)
        }
