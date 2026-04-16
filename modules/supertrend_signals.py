"""
Supertrend Indicator with Buy/Sell Signal Generation
Professional Trading Strategy Module
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit as st
from typing import Optional, Dict, List

class SupertrendAnalyzer:
    """
    Supertrend Indicator Implementation with Signal Generation
    
    Parameters:
    - period: ATR period (default: 10)
    - multiplier: ATR multiplier (default: 3.0)
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        self.df = None
        self.signals = None
        
    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend indicator
        
        Formula:
        Basic Upper Band = (High + Low) / 2 + multiplier * ATR
        Basic Lower Band = (High + Low) / 2 - multiplier * ATR
        """
        
        df_copy = df.copy()
        
        # Calculate ATR (Average True Range)
        high_low = df_copy['High'] - df_copy['Low']
        high_close = abs(df_copy['High'] - df_copy['Close'].shift())
        low_close = abs(df_copy['Low'] - df_copy['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df_copy['ATR'] = true_range.rolling(window=self.period).mean()
        
        # Calculate Basic Bands
        hl_avg = (df_copy['High'] + df_copy['Low']) / 2
        df_copy['Basic_Upper'] = hl_avg + (self.multiplier * df_copy['ATR'])
        df_copy['Basic_Lower'] = hl_avg - (self.multiplier * df_copy['ATR'])
        
        # Calculate Final Bands
        df_copy['Final_Upper'] = 0.0
        df_copy['Final_Lower'] = 0.0
        df_copy['Supertrend'] = 0.0
        df_copy['Trend'] = 1  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(df_copy)):
            # Upper Band
            if (df_copy['Basic_Upper'].iloc[i] < df_copy['Final_Upper'].iloc[i-1]) or \
               (df_copy['Close'].iloc[i-1] > df_copy['Final_Upper'].iloc[i-1]):
                df_copy.loc[df_copy.index[i], 'Final_Upper'] = df_copy['Basic_Upper'].iloc[i]
            else:
                df_copy.loc[df_copy.index[i], 'Final_Upper'] = df_copy['Final_Upper'].iloc[i-1]
            
            # Lower Band
            if (df_copy['Basic_Lower'].iloc[i] > df_copy['Final_Lower'].iloc[i-1]) or \
               (df_copy['Close'].iloc[i-1] < df_copy['Final_Lower'].iloc[i-1]):
                df_copy.loc[df_copy.index[i], 'Final_Lower'] = df_copy['Basic_Lower'].iloc[i]
            else:
                df_copy.loc[df_copy.index[i], 'Final_Lower'] = df_copy['Final_Lower'].iloc[i-1]
            
            # Determine Supertrend and Trend
            if df_copy['Close'].iloc[i] <= df_copy['Final_Upper'].iloc[i]:
                df_copy.loc[df_copy.index[i], 'Supertrend'] = df_copy['Final_Upper'].iloc[i]
                df_copy.loc[df_copy.index[i], 'Trend'] = -1
            else:
                df_copy.loc[df_copy.index[i], 'Supertrend'] = df_copy['Final_Lower'].iloc[i]
                df_copy.loc[df_copy.index[i], 'Trend'] = 1
        
        self.df = df_copy
        return df_copy
    
    def generate_signals(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate Buy and Sell signals based on Supertrend
        
        Buy Signal: When Trend changes from -1 to 1 (downtrend to uptrend)
        Sell Signal: When Trend changes from 1 to -1 (uptrend to downtrend)
        """
        if df is not None:
            self.calculate_supertrend(df)
        elif self.df is None:
            raise ValueError("No data provided. Please provide DataFrame or run calculate_supertrend first.")
        
        signals_df = self.df.copy()
        
        # Generate signals
        signals_df['Signal'] = 0
        signals_df['Buy_Signal'] = np.nan
        signals_df['Sell_Signal'] = np.nan
        
        # Detect trend changes
        signals_df['Trend_Change'] = signals_df['Trend'].diff()
        
        # Buy signal: Trend changes from -1 to 1
        buy_condition = (signals_df['Trend_Change'] == 2) & (signals_df['Trend'] == 1)
        signals_df.loc[buy_condition, 'Signal'] = 1
        signals_df.loc[buy_condition, 'Buy_Signal'] = signals_df['Close']
        
        # Sell signal: Trend changes from 1 to -1
        sell_condition = (signals_df['Trend_Change'] == -2) & (signals_df['Trend'] == -1)
        signals_df.loc[sell_condition, 'Signal'] = -1
        signals_df.loc[sell_condition, 'Sell_Signal'] = signals_df['Close']
        
        # Calculate position (for backtesting)
        signals_df['Position'] = signals_df['Signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        # Calculate returns if position is held
        signals_df['Strategy_Returns'] = signals_df['Close'].pct_change() * signals_df['Position'].shift(1)
        signals_df['Buy_Hold_Returns'] = signals_df['Close'].pct_change()
        
        # Calculate cumulative returns
        signals_df['Cumulative_Strategy'] = (1 + signals_df['Strategy_Returns']).cumprod()
        signals_df['Cumulative_BuyHold'] = (1 + signals_df['Buy_Hold_Returns']).cumprod()
        
        self.signals = signals_df
        return signals_df
    
    def calculate_performance_metrics(self) -> dict:
        """Calculate performance metrics for the strategy"""
        if self.signals is None:
            raise ValueError("Please generate signals first using generate_signals()")
        
        total_trades = len(self.signals[self.signals['Signal'] != 0])
        buy_trades = len(self.signals[self.signals['Signal'] == 1])
        sell_trades = len(self.signals[self.signals['Signal'] == -1])
        
        # Calculate trade returns
        trade_returns = []
        entry_price = 0
        entry_date = None
        
        for i in range(len(self.signals)):
            if self.signals['Signal'].iloc[i] == 1:  # Buy signal
                entry_price = self.signals['Close'].iloc[i]
                entry_date = self.signals.index[i]
            elif self.signals['Signal'].iloc[i] == -1 and entry_price > 0:  # Sell signal
                exit_price = self.signals['Close'].iloc[i]
                trade_return = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_return)
                entry_price = 0
        
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0
        
        # Strategy vs Buy & Hold
        total_strategy_return = self.signals['Cumulative_Strategy'].iloc[-1] - 1
        total_buyhold_return = self.signals['Cumulative_BuyHold'].iloc[-1] - 1
        
        # Sharpe Ratio
        strategy_returns = self.signals['Strategy_Returns'].dropna()
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() != 0 else 0
        
        # Max Drawdown
        cumulative = self.signals['Cumulative_Strategy']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'Total Trades': total_trades,
            'Buy Signals': buy_trades,
            'Sell Signals': sell_trades,
            'Average Trade Return': avg_trade_return,
            'Win Rate': win_rate,
            'Strategy Total Return': total_strategy_return,
            'Buy & Hold Return': total_buyhold_return,
            'Strategy Alpha': total_strategy_return - total_buyhold_return,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Total Days': len(self.signals)
        }
        
        return metrics
    
    def create_signal_chart(self, title: str = "Supertrend Strategy - Buy/Sell Signals") -> go.Figure:
        """Create interactive chart with Supertrend and signals"""
        if self.signals is None:
            raise ValueError("Please generate signals first using generate_signals()")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(title, "Supertrend & Trend", "Cumulative Returns Comparison")
        )
        
        # Price and Candlestick
        fig.add_trace(
            go.Candlestick(
                x=self.signals.index,
                open=self.signals['Open'],
                high=self.signals['High'],
                low=self.signals['Low'],
                close=self.signals['Close'],
                name='OHLC',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Supertrend line
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals['Supertrend'],
                name='Supertrend',
                line=dict(color='purple', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Buy signals (green triangles)
        buy_signals = self.signals[self.signals['Signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=14,
                        color='green',
                        line=dict(color='darkgreen', width=1)
                    ),
                    text=[f'BUY at {price:.2f}' for price in buy_signals['Close']],
                    hoverinfo='text+x+y'
                ),
                row=1, col=1
            )
        
        # Sell signals (red triangles)
        sell_signals = self.signals[self.signals['Signal'] == -1]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=14,
                        color='red',
                        line=dict(color='darkred', width=1)
                    ),
                    text=[f'SELL at {price:.2f}' for price in sell_signals['Close']],
                    hoverinfo='text+x+y'
                ),
                row=1, col=1
            )
        
        # Trend indicator with colored background
        for i in range(len(self.signals) - 1):
            if self.signals['Trend'].iloc[i] == 1:
                fillcolor = 'rgba(46, 204, 113, 0.15)'
            else:
                fillcolor = 'rgba(231, 76, 60, 0.15)'
            
            fig.add_vrect(
                x0=self.signals.index[i],
                x1=self.signals.index[i+1],
                fillcolor=fillcolor,
                opacity=0.3,
                line_width=0,
                row=2, col=1
            )
        
        # Trend line
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals['Supertrend'],
                name='Supertrend',
                line=dict(color='purple', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Trend annotation
        latest_trend = self.signals['Trend'].iloc[-1]
        trend_text = "🟢 UPTREND" if latest_trend == 1 else "🔴 DOWNTREND"
        trend_color = "#2ECC71" if latest_trend == 1 else "#E74C3C"
        
        fig.add_annotation(
            x=self.signals.index[len(self.signals)//2],
            y=self.signals['Supertrend'].max(),
            text=trend_text,
            showarrow=False,
            font=dict(size=14, color="white", weight="bold"),
            bgcolor=trend_color,
            bordercolor="black",
            borderwidth=1,
            borderpad=5,
            row=2, col=1
        )
        
        # Cumulative returns comparison
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals['Cumulative_Strategy'],
                name='Strategy Returns',
                line=dict(color='blue', width=2.5)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals['Cumulative_BuyHold'],
                name='Buy & Hold Returns',
                line=dict(color='gray', width=2, dash='dash')
            ),
            row=3, col=1
        )
        
        # Layout updates
        fig.update_layout(
            title_x=0.5,
            height=900,
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Supertrend", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Returns", tickformat=".0%", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        # Disable rangeslider for candlestick
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        return fig
    
    def create_performance_dashboard(self) -> None:
        """Create Streamlit dashboard for performance metrics"""
        metrics = self.calculate_performance_metrics()
        
        st.subheader("📊 Strategy Performance Dashboard")
        
        #
