# app.py

from fastapi import FastAPI, HTTPException
from schemas import PortfolioRequest, PortfolioResponse
import pandas as pd
import numpy as np
from datetime import date

from data_processing import download_and_prepare_data
from optimization_models import (
    run_scipy_optimization,
    run_risk_parity_optimization,
    run_sortino_optimization,
    run_cvar_optimization
)
from backtest import run_backtest

app = FastAPI(title="Quantitative Portfolio Optimizer API")

STRATEGY_FUNCTIONS = {
    "mvo": run_scipy_optimization,
    "risk_parity": run_risk_parity_optimization,
    "pmpt": run_sortino_optimization,
    "cvar": run_cvar_optimization,
}

@app.post("/optimize_and_backtest", response_model=PortfolioResponse)
async def optimize_and_backtest(request_data: PortfolioRequest):
    try:
        start_date = f"{request_data.start_year}-01-01"
        end_date = f"{request_data.end_year}-12-31"
        
        prices_df, daily_returns, expected_returns, covariance_matrix = download_and_prepare_data(
            request_data.tickers, start_date, end_date
        )

        all_results = {}
        strategies_to_run = request_data.strategies_to_run + ["benchmark"]
        
        for strategy_key in set(strategies_to_run):
            weights = pd.Series(dtype=float)
            
            # --- THIS IS THE FIX: Each strategy now has its own specific logic block ---
            if strategy_key == "benchmark":
                weights = pd.Series(0.0, index=request_data.tickers)
                weights["SPY"] = 1.0
            
            elif strategy_key == "mvo":
                weights_arr = STRATEGY_FUNCTIONS[strategy_key](expected_returns, covariance_matrix, 0.02, request_data.tickers)
                weights = pd.Series(weights_arr, index=request_data.tickers)

            elif strategy_key == "risk_parity":
                # This function correctly receives only the 2 arguments it needs
                weights_arr = STRATEGY_FUNCTIONS[strategy_key](covariance_matrix, request_data.tickers)
                weights = pd.Series(weights_arr, index=request_data.tickers)

            elif strategy_key == "pmpt":
                weights_arr = STRATEGY_FUNCTIONS[strategy_key](daily_returns, 0.02, request_data.tickers)
                weights = pd.Series(weights_arr, index=request_data.tickers)
            
            elif strategy_key == "cvar":
                weights_dict = STRATEGY_FUNCTIONS[strategy_key](expected_returns, daily_returns, request_data.tickers)
                weights = pd.Series(weights_dict)
            # --- END OF FIX ---

            equity_curve, metrics = run_backtest(prices_df, weights, request_data.initial_capital)
            
            annual_returns_raw = equity_curve.resample('YE').last().pct_change()
            annual_returns_sanitized = annual_returns_raw.replace([np.inf, -np.inf], 0).fillna(0).to_dict()
            annual_returns = {year.year: val for year, val in annual_returns_sanitized.items()}
            
            all_results[strategy_key] = {
                "weights": weights.to_dict(),
                "performance_metrics": metrics,
                "equity_curve": { "dates": equity_curve.index.strftime('%Y-%m-%d').tolist(), "values": equity_curve.values.tolist() },
                "annual_returns": annual_returns
            }
            
        return {"results": all_results}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")