"""
End-to-End Execution Pipeline for Multi-Strategy Quantitative Portfolio Optimizer.
Compares Markowitz Max Sharpe, Equal Risk Parity, PMPT Max Sortino, and Tail Risk Min CVaR strategies.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import PortfolioDataLoader
from src.portfolio_optimizer import MultiStrategyPortfolioOptimizer


def main():
    print("=" * 95)
    print("MULTI-STRATEGY QUANTITATIVE PORTFOLIO OPTIMIZATION ENGINE")
    print("Asset Universe: AAPL, MSFT, JPM, XOM, JNJ, TLT | Benchmark: S&P 500 & US Treasuries")
    print("Formulation: Markowitz Max Sharpe | Risk Parity | PMPT Sortino | Min CVaR (95% ES)")
    print("=" * 95)

    print("\n[1/3] Ingesting 5-year multi-asset price histories and computing Ledoit-Wolf shrinkage...")
    loader = PortfolioDataLoader(data_dir="data")
    data = loader.load_price_data()
    print(f"      Universe Size: {len(data['tickers'])} institutional assets ({', '.join(data['tickers'])})")
    print(f"      Historical Trading Days: {len(data['daily_returns']):,} days")

    print("\n[2/3] Solving 4 quantitative allocation paradigms via SLSQP Non-Linear Optimization...")
    optimizer = MultiStrategyPortfolioOptimizer(data, risk_free_rate=0.04)

    sharpe_res = optimizer.optimize_max_sharpe()
    rp_res = optimizer.optimize_risk_parity()
    sortino_res = optimizer.optimize_max_sortino()
    cvar_res = optimizer.optimize_min_cvar(alpha=0.95)

    print("\n[3/3] Quantitative Strategy Comparison Matrix:")
    print("=" * 95)
    print(f"{'Strategy':<30} | {'Exp Return':<12} | {'Volatility':<12} | {'Sharpe':<10} | {'Sortino':<10}")
    print("-" * 95)
    print(f"{sharpe_res['strategy']:<30} | {sharpe_res['expected_annual_return']:<12.2%} | {sharpe_res['annual_volatility']:<12.2%} | {sharpe_res['sharpe_ratio']:<10.2f} | {'-':<10}")
    print(f"{rp_res['strategy']:<30} | {rp_res['expected_annual_return']:<12.2%} | {rp_res['annual_volatility']:<12.2%} | {rp_res['sharpe_ratio']:<10.2f} | {'-':<10}")
    print(f"{sortino_res['strategy']:<30} | {sortino_res['expected_annual_return']:<12.2%} | {sortino_res['annual_volatility']:<12.2%} | {'-':<10} | {sortino_res['sortino_ratio']:<10.2f}")
    print(f"{cvar_res['strategy']:<30} | {cvar_res['expected_annual_return']:<12.2%} | {cvar_res['annual_volatility']:<12.2%} | {'-':<10} | {'-':<10}")
    print("=" * 95)

    print("\nOptimal Markowitz Asset Allocation Weights:")
    for ticker, weight in sharpe_res['weights'].items():
        print(f"  * {ticker:<6}: {weight:>6.2%}")

    print("\nDiscrete $100,000 Portfolio Execution:")
    discrete = optimizer.compute_discrete_allocation(sharpe_res['weights'], total_capital=100000.0)
    for ticker, count in discrete['allocated_shares'].items():
        print(f"  * Buy {count:>4} shares of {ticker}")
    print(f"  * Allocated: ${discrete['allocated_capital']:,.2f} | Uninvested Cash: ${discrete['leftover_cash']:.2f}")

    print("\n[CONCLUSION] Successfully generated multi-strategy frontier portfolios with covariance regularization,")
    print(f"   maximizing risk-adjusted yield (Sharpe: {sharpe_res['sharpe_ratio']:.2f}).")
    print("=" * 95)


if __name__ == '__main__':
    main()
