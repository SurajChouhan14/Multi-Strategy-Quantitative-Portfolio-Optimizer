"""
Multi-Strategy Quantitative Portfolio Optimization Engine.

Implements 5 core quantitative portfolio allocation paradigms:
1. Markowitz Modern Portfolio Theory (Max Sharpe Ratio via Quadratic / Non-Linear Programming)
2. Risk Parity (Equal Marginal Risk Contribution)
3. Post-Modern Portfolio Theory (Max Sortino Ratio with Downside Semi-Variance)
4. Conditional Value-at-Risk (Min CVaR / Expected Shortfall Tail Risk)
5. Discrete Capital Allocation for execution
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MultiStrategyPortfolioOptimizer:
    """
    Quantitative Portfolio Optimizer supporting Sharpe, Risk Parity, Sortino, and CVaR strategies.
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
        Solves Markowitz Maximum Sharpe Ratio optimization.
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

    def optimize_risk_parity(self):
        """
        Solves Equal Risk Contribution (Risk Parity) allocation.
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

    def optimize_max_sortino(self):
        """
        Solves Post-Modern Portfolio Theory (Max Sortino Ratio) penalizing downside volatility only.
        """
        daily_rets = self.daily_returns.values
        rf_daily = self.rf / 252.0

        def neg_sortino(weights):
            p_daily = np.dot(daily_rets, weights)
            downside = p_daily[p_daily < rf_daily] - rf_daily
            downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(252.0)
            if downside_dev == 0:
                return -100.0
            p_ret_ann = np.mean(p_daily) * 252.0
            sortino = (p_ret_ann - self.rf) / downside_dev
            return -sortino

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        w0 = np.ones(self.num_assets) / self.num_assets

        res = minimize(neg_sortino, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = res.x

        cov = self.cov_matrix.values
        mu = self.expected_returns.values
        p_ret = float(np.dot(weights, mu))
        p_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))

        p_daily = np.dot(daily_rets, weights)
        downside = p_daily[p_daily < rf_daily] - rf_daily
        downside_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(252.0))
        sortino = (p_ret - self.rf) / downside_dev if downside_dev > 0 else 0.0

        return {
            'strategy': 'PMPT Max Sortino',
            'weights': {t: round(float(w), 4) for t, w in zip(self.tickers, weights)},
            'expected_annual_return': round(p_ret, 4),
            'annual_volatility': round(p_vol, 4),
            'sortino_ratio': round(sortino, 4)
        }

    def optimize_min_cvar(self, alpha=0.95):
        """
        Solves 95% Conditional Value at Risk (Expected Shortfall) minimization.
        """
        daily_rets = self.daily_returns.values

        def cvar_objective(weights):
            p_daily = np.dot(daily_rets, weights)
            var_thresh = np.percentile(p_daily, (1.0 - alpha) * 100.0)
            tail_losses = p_daily[p_daily <= var_thresh]
            cvar = -np.mean(tail_losses) * np.sqrt(252.0)
            return cvar

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        w0 = np.ones(self.num_assets) / self.num_assets

        res = minimize(cvar_objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        weights = res.x

        cov = self.cov_matrix.values
        mu = self.expected_returns.values
        p_ret = float(np.dot(weights, mu))
        p_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))

        p_daily = np.dot(daily_rets, weights)
        var_thresh = np.percentile(p_daily, (1.0 - alpha) * 100.0)
        cvar_val = float(-np.mean(p_daily[p_daily <= var_thresh]) * np.sqrt(252.0))

        return {
            'strategy': 'Tail Risk Min CVaR (95% ES)',
            'weights': {t: round(float(w), 4) for t, w in zip(self.tickers, weights)},
            'expected_annual_return': round(p_ret, 4),
            'annual_volatility': round(p_vol, 4),
            'annualized_cvar_95': round(cvar_val, 4)
        }

    def compute_discrete_allocation(self, weights_dict, total_capital=100000.0):
        """
        Computes exact integer shares to purchase for a given cash portfolio.
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
