# ui.py

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
from ticker_utils import load_ticker_data

# --- Page Configuration ---
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- Helper Functions (unchanged) ---
def format_weights_for_display(weights_dict):
    weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=['Weight'])
    weights_df['Weight'] = weights_df['Weight'].map('{:.2%}'.format)
    return weights_df[weights_df['Weight'] != '0.00%']

def create_pie_chart(weights_dict):
    weights_df = pd.DataFrame.from_dict(weights_dict, orient='index', columns=['Weight'])
    active_weights = weights_df[weights_df['Weight'] > 0]
    fig = px.pie(active_weights, values='Weight', names=active_weights.index, title='Asset Allocation')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

# --- UI Sidebar for Inputs ---
st.sidebar.header("User Controls")

# --- Load Ticker Data ---
# This now returns three items for more robust handling
formatted_tickers, ticker_mapping, symbol_to_formatted = load_ticker_data()

STRATEGY_OPTIONS = {
    "Maximize Sharpe Ratio (MVO)": "mvo", "Risk Parity": "risk_parity",
    "Maximize Sortino Ratio (PMPT)": "pmpt", "Minimize CVaR (Tail Risk)": "cvar"
}

# --- DYNAMIC DEFAULTS ---
st.sidebar.subheader("1. Select Your Assets")
# Define the tickers we would LIKE to have as defaults
desired_default_tickers = ['SPY', 'AAPL', 'MSFT', 'QQQ', 'GLD', 'TLT']
# Programmatically find the full names that actually exist in our downloaded list
default_selections = []
if symbol_to_formatted: # Check that data loaded successfully
    for ticker in desired_default_tickers:
        if ticker in symbol_to_formatted:
            default_selections.append(symbol_to_formatted[ticker])

selected_formatted_tickers = st.sidebar.multiselect(
    "Type a company name or ticker to search",
    options=formatted_tickers,
    default=default_selections # Use the safe, validated list of defaults
)
# --- END OF DYNAMIC DEFAULTS ---

st.sidebar.subheader("2. Configure Your Analysis")
current_year = datetime.now().year
start_year_input = st.sidebar.selectbox("Start Year", options=range(2010, current_year), index=10)
end_year_input = st.sidebar.selectbox("End Year", options=range(2011, current_year + 1), index=14)
primary_strategy_name = st.sidebar.selectbox("Primary Strategy", options=list(STRATEGY_OPTIONS.keys()))
comparison_strategy_name = st.sidebar.selectbox("Comparison Strategy", options=list(STRATEGY_OPTIONS.keys()), index=1)

# --- Main Page ---
st.title("Multi-Strategy Portfolio Analysis")

if st.sidebar.button("Run Analysis"):
    selected_tickers = [ticker_mapping[formatted] for formatted in selected_formatted_tickers]
    
    if "SPY" not in selected_tickers: selected_tickers.append("SPY")

    if not selected_tickers: st.error("Please select at least one asset.")
    elif start_year_input >= end_year_input: st.error("Start Year must be before End Year.")
    else:
        primary_strategy_key = STRATEGY_OPTIONS[primary_strategy_name]
        comparison_strategy_key = STRATEGY_OPTIONS[comparison_strategy_name]
        
        api_payload = {
            "tickers": selected_tickers, "start_year": start_year_input, "end_year": end_year_input,
            "initial_capital": 100000, "strategies_to_run": list(set([primary_strategy_key, comparison_strategy_key]))
        }
        
        with st.spinner("Running optimizations and backtests... This is computationally intensive and may take a moment."):
            try:
                response = requests.post("http://127.0.0.1:8000/optimize_and_backtest", json=api_payload)
                response.raise_for_status()
                results = response.json()['results']
                # ... The rest of the page logic is identical ...
                st.success("Analysis Complete!")
                st.header("Portfolio Allocations")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(primary_strategy_name)
                    st.dataframe(format_weights_for_display(results[primary_strategy_key]['weights']))
                    st.plotly_chart(create_pie_chart(results[primary_strategy_key]['weights']), use_container_width=True)
                with col2:
                    st.subheader(comparison_strategy_name)
                    st.dataframe(format_weights_for_display(results[comparison_strategy_key]['weights']))
                    st.plotly_chart(create_pie_chart(results[comparison_strategy_key]['weights']), use_container_width=True)
                st.header("Backtest Performance Comparison")
                metrics_to_show = [
                         'Final Portfolio Value',
                         'Cumulative Return',
                         'Annualized Return', 
                         'Annualized Volatility',
                         'Sharpe Ratio',
                         'Maximum Drawdown',
                         'Best Month',       
                         'Worst Month'      
                        ]
                comparison_df = pd.DataFrame({
                    primary_strategy_name: {m: results[primary_strategy_key]['performance_metrics'][m] for m in metrics_to_show},
                    comparison_strategy_name: {m: results[comparison_strategy_key]['performance_metrics'][m] for m in metrics_to_show},
                    "Benchmark (SPY)": {m: results['benchmark']['performance_metrics'][m] for m in metrics_to_show}
                })
                st.dataframe(comparison_df)
                st.header("Portfolio Growth Over Time")
                growth_df = pd.DataFrame({
                    'Date': pd.to_datetime(results['benchmark']['equity_curve']['dates']),
                    primary_strategy_name: results[primary_strategy_key]['equity_curve']['values'],
                    comparison_strategy_name: results[comparison_strategy_key]['equity_curve']['values'],
                    "Benchmark (SPY)": results['benchmark']['equity_curve']['values']
                }).set_index('Date')
                st.line_chart(growth_df)
                st.header("Annual Returns Comparison")
                annual_returns_df = pd.DataFrame({
                    primary_strategy_name: results[primary_strategy_key]['annual_returns'],
                    comparison_strategy_name: results[comparison_strategy_key]['annual_returns'],
                    "Benchmark (SPY)": results['benchmark']['annual_returns']
                }).T
                annual_returns_df.columns = annual_returns_df.columns.astype(str)
                annual_returns_long = annual_returns_df.reset_index().melt(
                    id_vars='index', var_name='Year', value_name='Return'
                ).rename(columns={'index': 'Strategy'})
                fig_bar = px.bar(annual_returns_long, x='Year', y='Return', color='Strategy', barmode='group', labels={'Return': 'Annual Return'}, title='Year-by-Year Performance')
                fig_bar.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig_bar, use_container_width=True)
            except requests.exceptions.RequestException as e: st.error(f"Error connecting to backend API: {e}")
            except Exception as e: st.error(f"An error occurred during analysis: {e}")