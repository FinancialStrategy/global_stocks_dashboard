"""
Global Data Loader Module
Supports Turkish, US, European, Asian, and Australian markets
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Dict, List
import yaml
from datetime import datetime
import time

@st.cache_data(ttl=3600, show_spinner=False)
def load_config():
    """Load global configuration from config.yaml"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("❌ config.yaml file not found!")
        return {}
    except Exception as e:
        st.error(f"❌ Error loading config: {e}")
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple tickers using yfinance.
    """
    if not tickers:
        return {}
    
    ticker_data = {}
    
    for ticker in tickers:
        try:
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            
            # Download data
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            if not data.empty and 'Close' in data.columns:
                ticker_data[ticker] = data
                
        except Exception as e:
            st.warning(f"Could not load {ticker}: {e}")
            continue
    
    return ticker_data

def get_benchmark_data(benchmark: str, start_date: str, end_date: str) -> pd.Series:
    """Fetch benchmark index data"""
    try:
        bench = yf.download(benchmark, start=start_date, end=end_date, progress=False)
        if not bench.empty and 'Close' in bench.columns:
            return bench['Close']
    except Exception as e:
        st.warning(f"Could not fetch benchmark {benchmark}: {e}")
    
    return pd.Series()
