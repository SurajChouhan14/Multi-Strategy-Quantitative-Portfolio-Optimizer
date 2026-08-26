# ticker_utils.py

import pandas as pd
import streamlit as st

@st.cache_data
def load_ticker_data():
    """
    Downloads and prepares a list of NASDAQ-traded stocks and ETFs for the UI.
    
    Returns:
        tuple: A list of formatted tickers, a mapping from formatted name back to ticker,
               and a mapping from ticker to formatted name.
    """
    try:
        url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt"
        df = pd.read_csv(url, sep='|')
        
        # Filter out test issues, non-operational stocks, and the file footer.
        df_filtered = df[(df['Test Issue'] == 'N') & (df['Financial Status'] == 'N')]
        df_filtered = df_filtered.iloc[:-1] # Drop the last row which is a file timestamp

        # --- THIS IS THE FIX ---
        # The previous filter for only ETFs was too restrictive. Now we keep both stocks and ETFs.
        
        df_filtered['FormattedTicker'] = df_filtered['Security Name'] + " (" + df_filtered['Symbol'] + ")"
        
        # Create a mapping from Formatted Ticker -> Symbol
        ticker_mapping = pd.Series(df_filtered.Symbol.values, index=df_filtered.FormattedTicker).to_dict()
        
        # Create a mapping from Symbol -> Formatted Ticker (for our dynamic defaults)
        symbol_to_formatted = pd.Series(df_filtered.FormattedTicker.values, index=df_filtered.Symbol).to_dict()
        
        formatted_tickers = df_filtered['FormattedTicker'].sort_values().tolist()
        
        return formatted_tickers, ticker_mapping, symbol_to_formatted
    except Exception as e:
        st.error(f"Failed to load ticker data from NASDAQ: {e}")
        return [], {}, {}