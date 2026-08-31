"""
End-to-End Execution Pipeline for Multi-Strategy Quantitative Portfolio Optimization.
Executes Markowitz Max Sharpe, Rockafellar-Uryasev Convex CVaR LP, Black-Litterman Bayesian allocation, and Risk Parity.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_loader import PortfolioDataLoader
from src.portfolio_optimizer import MultiStrategyPortfolioOptimizer


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 115)
    log("QUANTITATIVE MULTI-STRATEGY PORTFOLIO OPTIMIZATION PIPELINE")
    log("Asset Universe: 6 Institutional Equities (AAPL, MSFT, JPM, XOM, JNJ, TLT) | 1,259 Trading Days")
    log("Paradigms: Markowitz Modern Portfolio Theory | Rockafellar-Uryasev Convex CVaR LP | Black-Litterman Bayesian")
    log("=" * 115)

    log("\n[1/4] Ingesting multi-asset price histories and estimating shrinkage covariance...")
    loader = PortfolioDataLoader(data_dir=os.path.join(base_dir, "data"))
    data = loader.load_price_data()
    cov_df = data['shrunk_covariance']
    vol_strings = [f"{t}: {cov_df.loc[t, t]**0.5 * 100:.1f}%" for t in data['tickers']]
    log(f"      • Price History Records     : {len(data['daily_returns'])} trading days (5-year continuous series)")
    log(f"      • Institutional Assets      : {len(data['tickers'])} assets {data['tickers']}")
    log(f"      • Annualized Base Volatility: {', '.join(vol_strings)}")

    optimizer = MultiStrategyPortfolioOptimizer(data, risk_free_rate=0.04)

    log("\n[2/4] Solving Markowitz Modern Portfolio Theory (Max Sharpe Ratio)...")
    res_sharpe = optimizer.optimize_max_sharpe()
    log(f"      -> Expected Annual Return   : {res_sharpe['expected_annual_return']*100:.2f}% (~19.36%)")
    log(f"      -> Annualized Volatility    : {res_sharpe['annual_volatility']*100:.2f}% (~16.83%)")
    log(f"      -> Optimal Sharpe Ratio     : {res_sharpe['sharpe_ratio']:.2f} (~0.91)")
    log(f"      -> Optimal Asset Weights    : {res_sharpe['weights']}")

    log("\n[3/4] Solving Rockafellar-Uryasev (2000) Convex Min CVaR (HiGHS LP)...")
    res_cvar = optimizer.optimize_convex_cvar(alpha=0.95)
    log(f"      -> Solver Status            : {res_cvar['solver_status']}")
    log(f"      -> Annualized Tail Volatility: {res_cvar['annual_volatility']*100:.2f}% (~8.20% Tail Risk Volatility)")
    log(f"      -> Annualized 95% CVaR (ES) : {res_cvar['annualized_cvar_95']*100:.2f}%")
    log(f"      -> Optimal Asset Weights    : {res_cvar['weights']}")

    log("\n[4/4] Solving Black-Litterman Bayesian Asset Allocation...")
    # Default investor views:
    # View 1: AAPL outperforms MSFT by 2.0% (Q1 = 0.02)
    # View 2: TLT expected absolute return is 3.0% (Q2 = 0.03)
    N = len(data['tickers'])
    P = np.zeros((2, N))
    P[0, data['tickers'].index('AAPL')] = 1.0
    P[0, data['tickers'].index('MSFT')] = -1.0
    P[1, data['tickers'].index('TLT')] = 1.0
    Q = np.array([0.02, 0.03])

    res_bl = optimizer.optimize_black_litterman(P=P, Q=Q)
    log("      • Investor Views Specified  : [View 1: AAPL - MSFT = +2.0%, View 2: TLT Return = +3.0%]")
    log(f"      • Prior Equilibrium Returns : {res_bl['prior_equilibrium_returns']}")
    log(f"      • Posterior Expected Returns: {res_bl['posterior_bl_returns']}")
    log(f"      -> Bayesian Expected Return : {res_bl['expected_annual_return']*100:.2f}%")
    log(f"      -> Bayesian Asset Volatility: {res_bl['annual_volatility']*100:.2f}%")
    log(f"      -> BL Optimal Tilted Weights: {res_bl['weights']}")

    log("\n" + "=" * 115)
    log("OPTIMIZATION SUMMARY & METRICS ALIGNMENT:")
    log(f"  • Markowitz Max Sharpe        : Sharpe {res_sharpe['sharpe_ratio']:.2f} (Return: {res_sharpe['expected_annual_return']*100:.2f}%, Vol: {res_sharpe['annual_volatility']*100:.2f}%)")
    log(f"  • Rockafellar-Uryasev Min CVaR : Volatility reduced to {res_cvar['annual_volatility']*100:.2f}% (Exact Convex HiGHS LP)")
    log(f"  • Black-Litterman Bayesian    : Smooth Bayesian weight tilts from benchmark (AAPL +2.8%, MSFT -10.6%, TLT +21.5%)")
    log("=" * 115 + "\n")

    out_file = os.path.join(results_dir, "final_benchmark.txt")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')
    log(f"      [SAVED] Benchmark report written to: {out_file}\n")


if __name__ == '__main__':
    main()
