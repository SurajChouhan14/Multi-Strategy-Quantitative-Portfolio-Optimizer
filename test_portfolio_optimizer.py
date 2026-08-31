"""
Automated Test Suite for Multi-Strategy Quantitative Portfolio Optimizer.
Tests:
1. Multi-Asset Price Ingestion & Shrinkage Covariance Properties (1,259 days, 6 assets)
2. Markowitz Mean-Variance Max Sharpe Program Convergence (0.91 Sharpe, 19.36% Ret, 16.83% Vol)
3. Rockafellar-Uryasev (2000) Convex CVaR Linear Program (HiGHS Status 7: Optimal, Vol=8.20%, CVaR <= Equal-Weight CVaR)
4. Black-Litterman Bayesian Invariants (Zero-view collapse to benchmark, view-induced return and weight tilts)
5. Discrete Capital Allocation & Budget Feasibility
6. Equal Risk Contribution (Risk Parity ERC Invariants & Marginal Risk Decomposition)
"""

import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import PortfolioDataLoader
from src.portfolio_optimizer import MultiStrategyPortfolioOptimizer


class TestMultiStrategyPortfolioOptimizer(unittest.TestCase):
    """
    Hard unit tests for Markowitz, Rockafellar-Uryasev CVaR LP, Black-Litterman, and Risk Parity.
    """

    @classmethod
    def setUpClass(cls):
        cls.loader = PortfolioDataLoader(data_dir="data")
        cls.data = cls.loader.load_price_data()
        cls.optimizer = MultiStrategyPortfolioOptimizer(cls.data, risk_free_rate=0.04)

    def test_1_data_ingestion_and_covariance_properties(self):
        """Verify 1,259 daily records, 6 institutional assets, and positive-definite covariance."""
        self.assertEqual(len(self.data['tickers']), 6)
        self.assertEqual(len(self.data['daily_returns']), 1259)
        cov = self.data['shrunk_covariance'].values
        eigenvals = np.linalg.eigvalsh(cov)
        self.assertTrue(np.all(eigenvals > 0.0), "Covariance matrix must be strictly positive-definite.")

    def test_2_markowitz_max_sharpe_reproducibility(self):
        """Verify Markowitz Max Sharpe converges to 0.91 Sharpe Ratio, 19.36% return, 16.83% volatility."""
        res = self.optimizer.optimize_max_sharpe()
        self.assertEqual(res['strategy'], 'Markowitz Max Sharpe')
        self.assertAlmostEqual(res['expected_annual_return'], 0.1936, delta=0.005)
        self.assertAlmostEqual(res['annual_volatility'], 0.1683, delta=0.005)
        self.assertAlmostEqual(res['sharpe_ratio'], 0.91, delta=0.02)
        total_w = sum(res['weights'].values())
        self.assertAlmostEqual(total_w, 1.0, places=3)

    def test_3_rockafellar_uryasev_convex_cvar_lp(self):
        """Verify Rockafellar-Uryasev CVaR LP converges in HiGHS to 8.20% volatility with sub-equal-weight tail risk."""
        res = self.optimizer.optimize_convex_cvar(alpha=0.95)
        self.assertIn("Optimal", res['solver_status'])
        self.assertAlmostEqual(res['annual_volatility'], 0.0820, delta=0.005)
        self.assertGreater(res['annualized_cvar_95'], 0.0)

        # Invariant: Min-CVaR portfolio has lower 95% tail loss than Equal-Weight benchmark
        w_eq = np.ones(6) / 6.0
        p_daily_eq = np.dot(self.data['daily_returns'].values, w_eq)
        var_thresh = np.percentile(p_daily_eq, 5.0)
        cvar_eq = -np.mean(p_daily_eq[p_daily_eq <= var_thresh]) * np.sqrt(252.0)
        self.assertLessEqual(res['annualized_cvar_95'], cvar_eq)

    def test_4_black_litterman_bayesian_invariants(self):
        """Verify Black-Litterman Bayesian allocation prior equilibrium and view updates."""
        # 1. Zero-view invariant: Passing no views collapses posterior returns to prior and weights to benchmark
        res_zero = self.optimizer.optimize_black_litterman(P=None, Q=None)
        for t in self.data['tickers']:
            self.assertAlmostEqual(res_zero['prior_equilibrium_returns'][t], res_zero['posterior_bl_returns'][t], places=6)
            self.assertAlmostEqual(res_zero['weights'][t], res_zero['benchmark_weights'][t], places=3)

        # 2. View-induced Bayesian updates:
        N = len(self.data['tickers'])
        P = np.zeros((2, N))
        P[0, self.data['tickers'].index('AAPL')] = 1.0
        P[0, self.data['tickers'].index('MSFT')] = -1.0
        P[1, self.data['tickers'].index('TLT')] = 1.0
        Q = np.array([0.02, 0.03])

        res = self.optimizer.optimize_black_litterman(P=P, Q=Q)
        prior = res['prior_equilibrium_returns']
        post = res['posterior_bl_returns']

        # View 1 (AAPL > MSFT by 2%): posterior return spread expands relative to prior spread
        prior_spread = prior['AAPL'] - prior['MSFT']
        post_spread = post['AAPL'] - post['MSFT']
        self.assertGreater(post_spread, prior_spread)

        # Weight tilt: AAPL weight increases vs benchmark, MSFT weight decreases vs benchmark
        self.assertGreater(res['weights']['AAPL'], res['benchmark_weights']['AAPL'])
        self.assertLess(res['weights']['MSFT'], res['benchmark_weights']['MSFT'])

    def test_5_discrete_capital_allocation(self):
        """Verify discrete integer share execution allocates cash budget without exceeding funds."""
        sharpe_res = self.optimizer.optimize_max_sharpe()
        alloc = self.optimizer.compute_discrete_allocation(sharpe_res['weights'], total_capital=100000.0)
        self.assertEqual(alloc['total_portfolio_value'], 100000.0)
        self.assertLessEqual(alloc['allocated_capital'], 100000.0)
        self.assertGreaterEqual(alloc['leftover_cash'], 0.0)
        self.assertAlmostEqual(alloc['allocated_capital'] + alloc['leftover_cash'], 100000.0, places=2)

    def test_6_risk_parity_erc(self):
        """Verify Equal Risk Contribution (ERC) produces positive weights summing to 1.0 and equalized risk contributions."""
        res = self.optimizer.optimize_risk_parity()
        self.assertEqual(res['strategy'], 'Equal Risk Parity (ERC)')
        weights = np.array([res['weights'][t] for t in self.data['tickers']])
        
        # Invariant 1: Budget constraint & long-only bounds
        self.assertAlmostEqual(np.sum(weights), 1.0, places=3)
        for w in weights:
            self.assertGreater(w, 0.0)

        # Invariant 2: Equal risk contribution property (each asset contributes ~1/6 = 16.67% of total risk)
        cov = self.data['shrunk_covariance'].values
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        mcr = np.dot(cov, weights) / p_vol
        rc = weights * mcr
        rc_pct = rc / np.sum(rc)
        target_pct = 1.0 / len(weights)

        for p in rc_pct:
            self.assertAlmostEqual(p, target_pct, delta=0.03)


if __name__ == '__main__':
    unittest.main()
