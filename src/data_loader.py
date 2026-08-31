"""
Quantitative Multi-Asset Historical Data Loader & Statistical Estimator.
Ingests institutional daily price histories and estimates returns, annualized expected returns,
and diagonal-target shrinkage covariance matrices.
"""

import os
import pandas as pd
import numpy as np


class PortfolioDataLoader:
    """
    Data loader for multi-asset daily price series and statistical estimation.
    """

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.price_path = os.path.join(self.data_dir, "portfolio_asset_prices.csv")

    def load_price_data(self, tickers=None):
        """
        Loads asset price histories from local deterministic CSV and computes daily returns,
        annualized expected returns, and shrinkage covariance.
        """
        if not os.path.exists(self.price_path):
            raise FileNotFoundError(f"Asset price file not found at {self.price_path}")

        df_prices = pd.read_csv(self.price_path, index_col='Date', parse_dates=True)
        if tickers is not None:
            df_prices = df_prices[[t for t in tickers if t in df_prices.columns]]

        daily_returns = df_prices.pct_change().dropna()
        expected_returns = daily_returns.mean() * 252.0
        sample_covariance = daily_returns.cov() * 252.0

        # Shrinkage covariance estimation (delta=0.20 shrinkage toward diagonal variance target)
        n_obs, n_assets = daily_returns.shape
        prior = np.diag(np.diag(sample_covariance))
        shrinkage = 0.20
        shrunk_covariance = pd.DataFrame(
            (1.0 - shrinkage) * sample_covariance.values + shrinkage * prior,
            index=sample_covariance.index,
            columns=sample_covariance.columns
        )

        return {
            'prices': df_prices,
            'daily_returns': daily_returns,
            'expected_returns': expected_returns,
            'sample_covariance': sample_covariance,
            'shrunk_covariance': shrunk_covariance,
            'tickers': list(df_prices.columns)
        }
