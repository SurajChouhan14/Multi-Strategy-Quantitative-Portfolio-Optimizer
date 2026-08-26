# data_processing.py

import yfinance as yf
import pandas as pd

def download_and_prepare_data(tickers, start_date, end_date):
    """
    Downloads historical price data, calculates daily returns,
    and computes the annualized expected returns and covariance matrix.
    """
    print("Downloading historical price data...")
    all_data = yf.download(tickers, start=start_date, end=end_date)
    prices_df = all_data['Close'].copy()
    prices_df.ffill(inplace=True)
    print("Data download and cleaning complete.")

    print("\nCalculating expected returns and covariance matrix...")
    daily_returns = prices_df.pct_change().dropna()
    expected_returns = daily_returns.mean() * 252
    covariance_matrix = daily_returns.cov() * 252
    print("Calculations complete.")

    return prices_df, daily_returns, expected_returns, covariance_matrix