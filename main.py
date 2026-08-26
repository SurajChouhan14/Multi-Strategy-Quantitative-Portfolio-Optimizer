# main.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_processing import download_and_prepare_data
from optimization_models import (
    run_monte_carlo_optimization,
    run_scipy_optimization,
    run_pypfopt_optimization,
    run_risk_parity_optimization,
    run_sortino_optimization,
    run_cvar_optimization,
    calculate_discrete_allocation
)
from backtest import run_backtest

# --- CONFIGURATION ---
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'PG', 'SPY', 'TLT', 'VXUS', 'GLD']
BENCHMARK_TICKER = 'SPY'
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
RISK_FREE_RATE = 0.02
INITIAL_CAPITAL = 100000
NUM_PORTFOLIOS_MC = 20000

def get_portfolio_performance(weights, expected_returns, covariance_matrix, risk_free_rate):
    """ Helper function to calculate theoretical portfolio performance metrics. """
    weights = np.array(list(weights.values())) if isinstance(weights, dict) else np.array(weights)
    returns = np.sum(expected_returns * weights)
    volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    sharpe = (returns - risk_free_rate) / volatility
    return returns, volatility, sharpe

def main():
    """ Main function to run all portfolio optimization and analysis steps. """
    # --- Step 1: Get Data ---
    prices_df, daily_returns, expected_returns, covariance_matrix = download_and_prepare_data(
        TICKERS, START_DATE, END_DATE
    )

    # --- Step 2: Calculate Optimal Weights for All Strategies ---
    print("\n--- Calculating Optimal Weights & Theoretical Performance ---")
    
    # MVO (SciPy)
    mvo_weights = run_scipy_optimization(expected_returns, covariance_matrix, RISK_FREE_RATE, TICKERS)
    
    # Risk Parity
    rp_weights = run_risk_parity_optimization(covariance_matrix, TICKERS)
    
    # PMPT (Sortino Ratio)
    sortino_weights = run_sortino_optimization(daily_returns, RISK_FREE_RATE, TICKERS)

    # CVaR
    cvar_weights = run_cvar_optimization(expected_returns, daily_returns, TICKERS)

    # --- THE FIX IS HERE: Convert all weight arrays to properly indexed Series ---
    strategies = {
        "Max Sharpe (MVO)": pd.Series(mvo_weights, index=TICKERS),
        "Risk Parity": pd.Series(rp_weights, index=TICKERS),
        "Max Sortino (PMPT)": pd.Series(sortino_weights, index=TICKERS),
        "Min CVaR": pd.Series(cvar_weights), # cvar_weights is already a dict, so this works
    }
    # Add benchmark separately for clarity
    benchmark_weights = pd.Series(0.0, index=TICKERS); benchmark_weights[BENCHMARK_TICKER] = 1.0
    strategies[f"Benchmark ({BENCHMARK_TICKER})"] = benchmark_weights
    
    # Calculate theoretical performance for plotting
    mvo_return, mvo_vol, _ = get_portfolio_performance(strategies["Max Sharpe (MVO)"], expected_returns, covariance_matrix, RISK_FREE_RATE)
    rp_return, rp_vol, _ = get_portfolio_performance(strategies["Risk Parity"], expected_returns, covariance_matrix, RISK_FREE_RATE)
    sortino_return, sortino_vol, _ = get_portfolio_performance(strategies["Max Sortino (PMPT)"], expected_returns, covariance_matrix, RISK_FREE_RATE)
    cvar_return, cvar_vol, _ = get_portfolio_performance(strategies["Min CVaR"], expected_returns, covariance_matrix, RISK_FREE_RATE)
    
    # --- Step 3: Run Backtests on All Strategies ---
    print("\n\n--- Running Historical Backtests ---")
    backtest_results = {}
    for name, weights in strategies.items():
        print(f"\nBacktesting {name} Strategy...")
        equity_curve, metrics = run_backtest(prices_df, weights, INITIAL_CAPITAL)
        backtest_results[name] = {'equity_curve': equity_curve, 'metrics': metrics}
        for key, val in metrics.items(): print(f"  - {key}: {val}")

    # --- Step 4: Visualize Backtest Results ---
    print("\nGenerating backtest performance plot...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 8))
    for name, result in backtest_results.items():
        plt.plot(result['equity_curve'], label=name, lw=2, ls='--' if 'Benchmark' in name else '-')
    plt.title(f'Strategy Performance: ${INITIAL_CAPITAL:,.0f} Initial Investment', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Step 5: Visualize Theoretical Efficient Frontier ---
    print("\nGenerating theoretical Efficient Frontier plot...")
    mc_portfolio, portfolios_df = run_monte_carlo_optimization(
        expected_returns, covariance_matrix, NUM_PORTFOLIOS_MC, RISK_FREE_RATE, TICKERS
    )
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(portfolios_df['Volatility'], portfolios_df['Return'], c=portfolios_df['Sharpe Ratio'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.scatter(mvo_vol, mvo_return, c='blue', marker='X', s=250, edgecolors='black', label='Max Sharpe (MVO)')
    plt.scatter(rp_vol, rp_return, c='orange', marker='D', s=250, edgecolors='black', label='Risk Parity')
    plt.scatter(sortino_vol, sortino_return, c='purple', marker='P', s=250, edgecolors='black', label='Max Sortino (PMPT)')
    plt.scatter(cvar_vol, cvar_return, c='green', marker='s', s=250, edgecolors='black', label='Min CVaR')
    
    plt.title('Theoretical Risk-Return Space of Strategies', fontsize=16)
    plt.xlabel('Annual Volatility (Risk)', fontsize=12)
    plt.ylabel('Annual Expected Return', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    print("Script finished.")

if __name__ == '__main__':
    main()