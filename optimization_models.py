# optimization_models.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pypfopt import EfficientFrontier
from pypfopt import EfficientCVaR
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

def run_monte_carlo_optimization(expected_returns, covariance_matrix, num_portfolios, risk_free_rate, tickers):
    """
    Runs a Monte Carlo simulation to find the optimal portfolio.
    """
    print("\nRunning Monte Carlo simulation...")
    num_assets = len(tickers)
    portfolio_returns = []
    portfolio_volatility = []
    portfolio_weights_list = []

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        port_return = np.sum(weights * expected_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        portfolio_returns.append(port_return)
        portfolio_volatility.append(port_volatility)
        portfolio_weights_list.append(weights)

    portfolios_df = pd.DataFrame({
        'Return': portfolio_returns,
        'Volatility': portfolio_volatility,
        'Weights': portfolio_weights_list
    })
    portfolios_df['Sharpe Ratio'] = (portfolios_df['Return'] - risk_free_rate) / portfolios_df['Volatility']
    
    max_sharpe_portfolio = portfolios_df.loc[portfolios_df['Sharpe Ratio'].idxmax()]
    
    print("\n--- Optimal Portfolio (Monte Carlo Simulation) ---")
    print(f"Expected Annual Return: {max_sharpe_portfolio['Return']:.2%}")
    print(f"Annual Volatility (Risk): {max_sharpe_portfolio['Volatility']:.2%}")
    print(f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe Ratio']:.2f}")
    
    return max_sharpe_portfolio, portfolios_df

def run_scipy_optimization(expected_returns, covariance_matrix, risk_free_rate, tickers):
    """
    Runs precise optimization using SciPy's SLSQP solver for Max Sharpe.
    NOW IT ONLY RETURNS THE WEIGHTS for consistency.
    """
    print("\n\nRunning precise optimization with SciPy...")
    num_assets = len(tickers)

    def minimize_negative_sharpe(weights, exp_returns, cov_matrix, risk_free_rate):
        port_return = np.sum(weights * exp_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility
        return -sharpe_ratio

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array(num_assets * [1. / num_assets,])

    scipy_result = minimize(
        fun=minimize_negative_sharpe,
        x0=initial_guess,
        args=(expected_returns, covariance_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = scipy_result.x
    print("\n--- Optimal Portfolio (SciPy SLSQP Solver) ---")
    return optimal_weights

def run_pypfopt_optimization(expected_returns, covariance_matrix, risk_free_rate):
    """
    Runs optimization using the PyPortfolioOpt library.
    """
    print("\n\nRunning optimization with PyPortfolioOpt library...")
    ef = EfficientFrontier(expected_returns, covariance_matrix)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()

    print("\n--- Optimal Portfolio (PyPortfolioOpt Library) ---")
    ef.portfolio_performance(verbose=True, risk_free_rate=risk_free_rate)
    return cleaned_weights

def run_risk_parity_optimization(covariance_matrix, tickers):
    """
    Calculates the weights for a Risk Parity portfolio.
    This portfolio aims for equal risk contribution from each asset.
    """
    print("\n\nRunning Risk Parity (Equal Risk Contribution) optimization...")
    num_assets = len(tickers)

    def calculate_risk_contribution(weights, cov_matrix):
        """Calculates the risk contribution of each asset in the portfolio."""
        weights = np.array(weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        # Marginal Contribution to Risk
        mcr = np.dot(cov_matrix, weights) / portfolio_volatility
        # Risk Contribution
        rc = weights * mcr
        # Normalize to percentage
        return rc / np.sum(rc)

    def risk_parity_objective(weights, cov_matrix):
        """Objective function for the optimizer to minimize the variance of risk contributions."""
        # We want all assets to have the same risk contribution.
        # Let's say the target is 1/N for each.
        target_contribution = np.full(num_assets, 1 / num_assets)
        
        # Get the actual risk contributions
        actual_contribution = calculate_risk_contribution(weights, cov_matrix)
        
        # Return the sum of squared differences from the target.
        # This is what the optimizer will try to make zero.
        return np.sum((actual_contribution - target_contribution)**2)

    # Set constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array(num_assets * [1. / num_assets,])

    # Run the optimizer
    rp_result = minimize(
        fun=risk_parity_objective,
        x0=initial_guess,
        args=(covariance_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = rp_result.x
    
    print("\n--- Optimal Portfolio (Risk Parity) ---")
    
    return optimal_weights
#SORTINO RATIO (PMPT)
def run_sortino_optimization(daily_returns, risk_free_rate, tickers):
    """
    Calculates the optimal portfolio that maximizes the Sortino Ratio.
    This is a from-scratch implementation using SciPy.
    """
    print("\n\nRunning PMPT (Sortino Ratio) optimization...")
    num_assets = len(tickers)

    def minimize_negative_sortino(weights):
        """ The objective function to be minimized by SciPy """
        # Calculate portfolio daily returns
        portfolio_returns = daily_returns.dot(weights)
        
        # Calculate downside returns (returns below the risk-free rate)
        target_return = risk_free_rate / 252 # Daily risk-free rate
        downside_returns = portfolio_returns[portfolio_returns < target_return]
        
        # Calculate downside deviation
        downside_deviation = np.std(downside_returns)
        
        # If there are no downside returns, deviation is 0
        if downside_deviation == 0:
            return -np.inf # Return a very large negative number to indicate high preference
            
        # Calculate annualized mean and Sortino Ratio
        annualized_mean_return = portfolio_returns.mean() * 252
        sortino_ratio = (annualized_mean_return - risk_free_rate) / downside_deviation / np.sqrt(252) # Annualize

        return -sortino_ratio

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array(num_assets * [1. / num_assets,])

    sortino_result = minimize(
        fun=minimize_negative_sortino,
        x0=initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = sortino_result.x
    print("\n--- Optimal Portfolio (Sortino Ratio) ---")
    return optimal_weights


# CVAR 
def run_cvar_optimization(expected_returns, daily_returns, tickers):
    """
    Calculates the optimal portfolio by minimizing Conditional Value at Risk (CVaR).
    This uses the PyPortfolioOpt library.
    """
    print("\n\nRunning CVaR (Expected Shortfall) optimization...")
    
    ef_cvar = EfficientCVaR(expected_returns, daily_returns)
    cvar_weights = ef_cvar.min_cvar()
    cleaned_weights = ef_cvar.clean_weights()

    print("\n--- Optimal Portfolio (Minimum CVaR) ---")
    ef_cvar.portfolio_performance(verbose=True)
    
    return cleaned_weights

def calculate_discrete_allocation(weights, prices_df, portfolio_value):
    """
    Calculates the discrete allocation of shares for a given portfolio value.
    """
    print(f"\n--- Discrete Allocation for a ${portfolio_value:,.0f} Portfolio ---")
    latest_prices = get_latest_prices(prices_df)
    da = DiscreteAllocation(
        weights=weights,
        latest_prices=latest_prices,
        total_portfolio_value=portfolio_value
    )
    allocation, leftover_cash = da.greedy_portfolio()
    print("Number of shares to buy:")
    print(allocation)
    print(f"Funds remaining: ${leftover_cash:.2f}")

