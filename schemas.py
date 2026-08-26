# schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict

# --- Request Model ---
class PortfolioRequest(BaseModel):
    tickers: List[str]
    start_year: int = Field(..., ge=2010, le=2024)
    end_year: int = Field(..., ge=2011, le=2025)
    initial_capital: int = 100000
    strategies_to_run: List[str] # e.g., ["mvo", "risk_parity"]

# --- Response Models ---
class EquityCurve(BaseModel):
    dates: List[str]
    values: List[float]

class StrategyResult(BaseModel):
    weights: Dict[str, float]
    performance_metrics: Dict[str, str]
    equity_curve: EquityCurve
    annual_returns: Dict[int, float]

class PortfolioResponse(BaseModel):
    # The response will be a dictionary where keys are strategy names (e.g., "mvo")
    # and values are the detailed results for that strategy.
    results: Dict[str, StrategyResult]