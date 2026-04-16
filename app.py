"""
Global Equity Analytics Platform
Professional Multi-Market Portfolio Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Global Equity Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple custom CSS
st.markdown("""
<style>
.stMetric {
    background-color: #F8F9FA;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
h1 {
    color: #1E3A5F;
    font-size: 1.8rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 0.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    background-color: #2E86C1;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Global Equity Analytics")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Load config
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("config.yaml file not found")
    st.stop()

# Market selection
available_markets = list(config.get('indices', {}).keys())
if not available_markets:
    st.error("No markets found in config.yaml")
    st.stop()

selected_markets = st.sidebar.multiselect(
    "Select Markets",
    options=available_markets,
    default=available_markets[:2] if len(available_markets) >= 2 else available_markets
)

# Date range
end_date = datetime.now().date()
start_date = end_date - timedelta(days=730)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", start_date, max_value=end_date)
with col2:
    end_date = st.date_input("End Date", end_date, max_value=end_date)

if start_date >= end_date:
    st.sidebar.error("End date must be after start date")

# Stock selection
st.sidebar.subheader("Stock Selection")
selected_tickers = {}
total_selected = 0

for market in selected_markets:
    market_config = config['indices'].get(market, {})
    market_tickers = market_config.get('tickers', [])
    
    if market_tickers:
        with st.sidebar.expander(f"{market}"):
            selected = st.multiselect(
                f"Select stocks",
                options=market_tickers,
                default=market_tickers[:3] if len(market_tickers) >= 3 else market_tickers,
                key=f"{market}_selector"
            )
            if selected:
                selected_tickers[market] = selected
                total_selected += len(selected)

# Supertrend settings
st.sidebar.markdown("---")
st.sidebar.subheader("Supertrend")
enable_supertrend = st.sidebar.checkbox("Enable Supertrend", value=True)

if enable_supertrend:
    supertrend_period = st.sidebar.slider("ATR Period", 5, 20, 10)
    supertrend_multiplier = st.sidebar.slider("ATR Multiplier", 1.0, 5.0, 3.0, 0.5)

# Run button
if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
    st.session_state['run_analysis'] = True
    st.session_state['selected_markets'] = selected_markets
    st.session_state['selected_tickers'] = selected_tickers
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    st.session_state['supertrend_params'] = {
        'enabled': enable_supertrend,
        'period': supertrend_period if enable_supertrend else 10,
        'multiplier': supertrend_multiplier if enable_supertrend else 3.0
    }

# Main content
if st.session_state.get('run_analysis', False):
    from modules.data_loader import fetch_market_data, get_benchmark_data
    
    selected_markets = st.session_state['selected_markets']
    selected_tickers_dict = st.session_state['selected_tickers']
    start_date = st.session_state['start_date']
    end_date = st.session_state['end_date']
    supertrend_params = st.session_state['supertrend_params']
    
    if total_selected == 0:
        st.warning("Please select at least one stock")
        st.stop()
    
    # Load data
    st.info(f"Loading data for {total_selected} stocks...")
    
    all_prices = {}
    all_ohlc_data = {}
    progress_bar = st.progress(0)
    
    for idx, market in enumerate(selected_markets):
        if market in selected_tickers_dict:
            tickers = selected_tickers_dict[market]
            progress_bar.progress((idx + 0.5) / len(selected_markets))
            
            ticker_data = fetch_market_data(tickers, str(start_date), str(end_date))
            
            if ticker_data:
                close_prices = {}
                for ticker, df in ticker_data.items():
                    if 'Close' in df.columns:
                        close_prices[ticker] = df['Close']
                        all_ohlc_data[ticker] = df
                
                if close_prices:
                    prices_df = pd.DataFrame(close_prices)
                    if not prices_df.empty:
                        all_prices[market] = prices_df
    
    progress_bar.progress(1.0)
    progress_bar.empty()
    
    if not all_prices:
        st.error("No data loaded. Please check your selections.")
        st.stop()
    
    # Combine prices
    combined_prices = pd.concat(all_prices.values(), axis=1).dropna(axis=1, how='all')
    combined_returns = combined_prices.pct_change().dropna()
    
    # Create tabs
    tabs_list = ["Overview", "Technical", "Portfolio", "Risk"]
    if supertrend_params['enabled']:
        tabs_list.append("Supertrend")
    
    tabs = st.tabs(tabs_list)
    
    # ========== TAB 1: OVERVIEW ==========
    with tabs[0]:
        st.header("Market Performance")
        
        # Normalized prices
        if not combined_prices.empty:
            normalized = combined_prices / combined_prices.iloc[0] * 100
            
            fig = px.line(
                normalized,
                title="Normalized Prices (Base 100)",
                labels={"value": "Price", "variable": "Ticker"}
            )
            fig.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation
        if len(combined_prices.columns) > 1:
            st.subheader("Correlation Matrix")
            corr = combined_returns.corr()
            
            fig_corr = px.imshow(
                corr,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Summary
        st.subheader("Summary Statistics")
        summary = []
        for market, prices in all_prices.items():
            if not prices.empty:
                rets = prices.pct_change().dropna()
                summary.append({
                    'Market': market,
                    'Stocks': len(prices.columns),
                    'Avg Return': f"{rets.mean().mean():.4%}",
                    'Avg Vol': f"{rets.std().mean():.4%}"
                })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
    
    # ========== TAB 2: TECHNICAL ==========
    with tabs[1]:
        st.header("Technical Analysis")
        
        if all_ohlc_data:
            all_tickers = list(all_ohlc_data.keys())
            selected_ticker = st.selectbox("Select Stock", all_tickers)
            
            if selected_ticker and selected_ticker in all_ohlc_data:
                df = all_ohlc_data[selected_ticker]
                
                if not df.empty and 'Close' in df.columns:
                    # Simple technical indicators
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    df['RSI'] = calculate_rsi(df['Close'])
                    
                    # Price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red')))
                    
                    fig.update_layout(
                        title=f"{selected_ticker} - Price & Indicators",
                        yaxis_title="Price",
                        height=500,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # RSI
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(title="RSI (14)", yaxis_title="RSI", height=300, template='plotly_white')
                    st.plotly_chart(fig_rsi, use_container_width=True)
    
    # ========== TAB 3: PORTFOLIO ==========
    with tabs[2]:
        st.header("Portfolio Optimization")
        
        if len(combined_prices.columns) >= 2:
            from modules.portfolio_optimizer import PortfolioOptimizer
            
            optimizer = PortfolioOptimizer(combined_prices)
            result = optimizer.optimize_max_sharpe()
            
            if result['status'] == 'success' and result['weights']:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Return", f"{result['expected_return']:.2%}")
                with col2:
                    st.metric("Volatility", f"{result['volatility']:.2%}")
                with col3:
                    st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
                
                # Weights chart
                weights_df = pd.DataFrame.from_dict(result['weights'], orient='index', columns=['Weight'])
                weights_df = weights_df[weights_df['Weight'] > 0.01].sort_values('Weight', ascending=False)
                
                if not weights_df.empty:
                    fig = px.bar(weights_df, x=weights_df.index, y='Weight', title="Portfolio Weights")
                    fig.update_yaxis(tickformat='.0%')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Optimization failed. Try different assets.")
        else:
            st.warning("Select at least 2 assets for optimization")
    
    # ========== TAB 4: RISK ==========
    with tabs[3]:
        st.header("Risk Analytics")
        
        if not combined_returns.empty:
            # Equal weight portfolio
            portfolio_returns = combined_returns.mean(axis=1)
            
            # Metrics
            total_return = (1 + portfolio_returns).prod() - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() != 0 else 0
            
            # Drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{total_return:.2%}")
            with col2:
                st.metric("Volatility", f"{volatility:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            with col4:
                st.metric("Max Drawdown", f"{max_dd:.2%}")
            
            # Drawdown chart
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, fill='tozeroy', name='Drawdown'))
            fig_dd.update_layout(title="Drawdown", yaxis_title="%", height=400, template='plotly_white')
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # Returns distribution
            fig_dist = px.histogram(portfolio_returns, nbins=50, title="Daily Returns Distribution")
            fig_dist.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # ========== TAB 5: SUPERTREND ==========
    if supertrend_params['enabled'] and len(tabs) > 4:
        with tabs[4]:
            st.header("Supertrend Strategy")
            
            from modules.supertrend_signals import SupertrendAnalyzer
            
            if all_ohlc_data:
                all_tickers = list(all_ohlc_data.keys())
                selected = st.selectbox("Select Stock", all_tickers, key="st_selector")
                
                if selected and selected in all_ohlc_data:
                    df = all_ohlc_data[selected]
                    
                    if len(df) > supertrend_params['period']:
                        analyzer = SupertrendAnalyzer(
                            period=supertrend_params['period'],
                            multiplier=supertrend_params['multiplier']
                        )
                        signals = analyzer.generate_signals(df)
                        
                        # Current status
                        current_trend = signals['Trend'].iloc[-1]
                        current_price = signals['Close'].iloc[-1]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            trend_text = "UPTREND" if current_trend == 1 else "DOWNTREND"
                            st.metric("Trend", trend_text)
                        with col2:
                            st.metric("Current Price", f"{current_price:.2f}")
                        with col3:
                            last_signal = signals[signals['Signal'] != 0]
                            if not last_signal.empty:
                                sig = last_signal.iloc[-1]['Signal']
                                signal_text = "BUY" if sig == 1 else "SELL"
                                st.metric("Last Signal", signal_text)
                        
                        # Chart
                        fig = analyzer.create_signal_chart(f"{selected}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Performance
                        analyzer.create_performance_dashboard()
                    else:
                        st.warning(f"Insufficient data. Need at least {supertrend_params['period']} days.")

else:
    # Welcome screen
    st.info("Select markets and stocks from the sidebar, then click 'Run Analysis'")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Global Coverage")
        st.markdown("- Turkey\n- USA\n- Europe\n- Asia\n- Australia")
    with col2:
        st.subheader("Features")
        st.markdown("- Portfolio Optimization\n- Risk Analytics\n- Technical Indicators\n- Supertrend Strategy")
    with col3:
        st.subheader("Analysis")
        st.markdown("- Correlation Matrix\n- Drawdown Analysis\n- Performance Metrics\n- Signal Generation")


# Helper function
def calculate_rsi(prices, period=14):
    """Calculate RSI without TA-Lib"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
