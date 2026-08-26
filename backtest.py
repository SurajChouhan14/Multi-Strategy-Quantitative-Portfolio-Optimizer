# backtest.py

import pandas as pd
import numpy as np

def run_backtest(prices_df, weights, initial_capital=100000, rebalance_freq='ME'):
    """
    Runs a historical backtest of a given portfolio strategy.
    (Updated to handle potential zero prices in data and avoid NaN errors)
    """
    weights = pd.Series(weights)
    asset_columns = prices_df.columns.tolist()
    
    holdings = pd.DataFrame(index=prices_df.index, columns=asset_columns + ['Cash', 'Total Value'], dtype=np.float64)
    
    rebalance_dates = prices_df.resample(rebalance_freq).first().index
    
    initial_date = holdings.index[0]
    holdings.loc[initial_date, 'Cash'] = initial_capital
    holdings.loc[initial_date, 'Total Value'] = initial_capital
    holdings.fillna(0, inplace=True)
    
    shares = pd.Series(0.0, index=asset_columns)

    for i in range(1, len(prices_df)):
        current_date = holdings.index[i]
        previous_date = holdings.index[i-1]
        
        current_shares = shares.copy()
        market_values = current_shares * prices_df.iloc[i]
        total_value = market_values.sum() + holdings.loc[previous_date, 'Cash']

        if current_date in rebalance_dates:
            target_dollar_values = weights * total_value
            current_prices = prices_df.iloc[i]
            
            # --- THE FIX IS HERE ---
            # Initialize new shares Series
            new_shares = pd.Series(0.0, index=asset_columns)
            
            # Identify assets with valid, non-zero prices
            valid_prices = current_prices[current_prices > 0]
            
            if not valid_prices.empty:
                # Only calculate shares for assets with valid prices
                valid_targets = target_dollar_values[valid_prices.index]
                calculated_shares = (valid_targets / valid_prices).astype(int)
                new_shares.update(calculated_shares)
            
            shares = new_shares
            # --- END OF FIX ---
            
            cost = (shares * current_prices.fillna(0)).sum() # Use fillna(0) to handle potential NaNs in price
            cash = total_value - cost
        else:
            shares = current_shares
            cash = holdings.loc[previous_date, 'Cash']
            
        holdings.loc[current_date, asset_columns] = shares * prices_df.iloc[i]
        holdings.loc[current_date, 'Cash'] = cash
        holdings.loc[current_date, 'Total Value'] = holdings.loc[current_date, asset_columns].sum() + cash

    equity_curve = holdings['Total Value'].dropna()
    
    # Performance Calculation
    cumulative_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    daily_returns = equity_curve.pct_change().dropna()
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    annualized_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1
    sharpe_ratio = (annualized_return - 0.02) / annualized_volatility if annualized_volatility != 0 else 0
    
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    monthly_returns = equity_curve.resample('ME').last().pct_change()
    best_month = monthly_returns.max()
    worst_month = monthly_returns.min()

    metrics = {
        'Final Portfolio Value': f"${equity_curve.iloc[-1]:,.2f}", 'Cumulative Return': f"{cumulative_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}", 'Annualized Volatility': f"{annualized_volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}", 'Maximum Drawdown': f"{max_drawdown:.2%}",
        'Best Month': f"{best_month:.2%}", 'Worst Month': f"{worst_month:.2%}",
    }
    return equity_curve, metrics