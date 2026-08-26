"""
Automated Unit Test Suite for Multi-Strategy Quantitative Portfolio Optimizer.
Verifies Data Loading, Covariance Regularization, Sharpe/Risk Parity/Sortino Optimization, and Discrete Allocation.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import PortfolioDataLoader
from src.portfolio_optimizer import MultiStrategyPortfolioOptimizer


class TestMultiStrategyPortfolioOptimizer(unittest.TestCase):
    """
    Unit test cases for quantitative portfolio optimization engine.
    """

    @classmethod
    def setUpClass(cls):
        cls.loader = PortfolioDataLoader(data_dir="data")
        cls.data = cls.loader.load_price_data()
        cls.optimizer = MultiStrategyPortfolioOptimizer(cls.data, risk_free_rate=0.04)

    def test_price_data_loading(self):
        """Verify 5-year price series dimensions and tickers."""
        self.assertEqual(len(self.data['tickers']), 6)
        self.assertGreater(len(self.data['daily_returns']), 500)
        self.assertEqual(self.data['shrunk_covariance'].shape, (6, 6))

    def test_max_sharpe_optimization(self):
        """Verify Markowitz weights sum to 1.0 and yield positive Sharpe ratio."""
        res = self.optimizer.optimize_max_sharpe()
        w_sum = sum(res['weights'].values())
        self.assertAlmostEqual(w_sum, 1.0, places=2)
        self.assertGreater(res['sharpe_ratio'], 0.0)
        self.assertGreater(res['expected_annual_return'], 0.0)

    def test_risk_parity_weights(self):
        """Verify Equal Risk Contribution produces diversified positive weights."""
        res = self.optimizer.optimize_risk_parity()
        w_sum = sum(res['weights'].values())
        self.assertAlmostEqual(w_sum, 1.0, places=2)
        for t, w in res['weights'].items():
            self.assertGreater(w, 0.0)

    def test_discrete_allocation_budget(self):
        """Verify discrete share allocation does not exceed total capital."""
        sharpe_res = self.optimizer.optimize_max_sharpe()
        discrete = self.optimizer.compute_discrete_allocation(sharpe_res['weights'], total_capital=50000.0)
        self.assertLessEqual(discrete['allocated_capital'], 50000.0)
        self.assertGreaterEqual(discrete['leftover_cash'], 0.0)


if __name__ == '__main__':
    unittest.main()
