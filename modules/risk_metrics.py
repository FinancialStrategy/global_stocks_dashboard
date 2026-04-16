"""
Risk Analytics Module - NO QuantStats Required
Professional risk metrics and performance analysis with manual calculations
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, Optional
from datetime import datetime


class RiskAnalyzer:
    """Professional risk analysis - No external dependencies"""
    
    def __init__(self, returns: pd.Series, benchmark_returns: pd.Series = None):
        """
        Initialize risk analyzer
        
        Args:
            returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns (optional)
        """
        self.returns = returns.dropna()
        self.benchmark = benchmark_returns.dropna() if benchmark_returns is not None else None
        
    def calculate_returns_metrics(self) -> Dict:
        """Calculate return-based metrics"""
        if self.returns.empty:
            return {}
        
        # Basic returns
        total_return = (1 + self.returns).prod() - 1
        
        # Calculate CAGR
        days = len(self.returns)
        years = days / 252
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Annual return
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # Monthly returns
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        return {
            'Total Return': total_return,
            'CAGR': cagr,
            'Annual Return': annual_return,
            'Average Daily Return': self.returns.mean(),
            'Median Daily Return': self.returns.median(),
            'Best Day': self.returns.max(),
            'Worst Day': self.returns.min(),
            'Positive Days': (self.returns > 0).sum(),
            'Negative Days': (self.returns < 0).sum(),
            'Win Rate': (self.returns > 0).sum() / len(self.returns),
            'Best Month': monthly_returns.max() if not monthly_returns.empty else 0,
            'Worst Month': monthly_returns.min() if not monthly_returns.empty else 0
        }
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate risk-based metrics"""
        if self.returns.empty:
            return {}
        
        # Volatility
        daily_vol = self.returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Drawdown
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Find max drawdown period
        drawdown_start = None
        drawdown_end = None
        in_drawdown = False
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] == 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = drawdown.index[i]
            elif drawdown.iloc[i] < 0 and in_drawdown and drawdown.iloc[i] == max_drawdown:
                drawdown_end = drawdown.index[i]
                break
        
        max_dd_duration = None
        if drawdown_start and drawdown_end:
            max_dd_duration = (drawdown_end - drawdown_start).days
        
        # VaR and CVaR
        var_95 = self.returns.quantile(0.05)
        var_99 = self.returns.quantile(0.01)
        cvar_95 = self.returns[self.returns <= var_95].mean()
        
        # Downside metrics
        negative_returns = self.returns[self.returns < 0]
        downside_deviation = negative_returns.std() if len(negative_returns) > 0 else daily_vol
        
        # Semi-deviation (downside volatility)
        below_mean = self.returns[self.returns < self.returns.mean()]
        semi_deviation = below_mean.std() if len(below_mean) > 0 else daily_vol
        
        return {
            'Daily Volatility': daily_vol,
            'Annual Volatility': annual_vol,
            'Max Drawdown': max_drawdown,
            'Max Drawdown Duration (Days)': max_dd_duration,
            'Value at Risk (95%)': var_95,
            'Value at Risk (99%)': var_99,
            'Conditional VaR (95%)': cvar_95,
            'Downside Deviation': downside_deviation,
            'Semi-Deviation': semi_deviation
        }
    
    def calculate_risk_adjusted_metrics(self) -> Dict:
        """Calculate risk-adjusted return metrics"""
        if self.returns.empty:
            return {}
        
        metrics = self.calculate_risk_metrics()
        returns_metrics = self.calculate_returns_metrics()
        
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_rf = risk_free_rate / 252
        
        # Sharpe Ratio
        excess_returns = self.returns - daily_rf
        sharpe = (excess_returns.mean() / self.returns.std() * np.sqrt(252)) if self.returns.std() != 0 else 0
        
        # Sortino Ratio (using downside deviation)
        downside_std = metrics['Downside Deviation']
        sortino = (excess_returns.mean() / downside_std * np.sqrt(252)) if downside_std != 0 else 0
        
        # Calmar Ratio
        calmar = (returns_metrics['Annual Return'] / abs(metrics['Max Drawdown'])) if metrics['Max Drawdown'] != 0 else 0
        
        # Omega Ratio
        threshold = 0
        gains = self.returns[self.returns > threshold].sum()
        losses = abs(self.returns[self.returns < threshold].sum())
        omega = gains / losses if losses != 0 else float('inf')
        
        # Treynor Ratio (needs beta, will calculate if benchmark exists)
        treynor = None
        if self.benchmark is not None:
            beta = self.calculate_beta()
            if beta > 0:
                treynor = (returns_metrics['Annual Return'] - risk_free_rate) / beta
        
        return {
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Omega Ratio': omega,
            'Treynor Ratio': treynor,
            'Risk-Free Rate': risk_free_rate
        }
    
    def calculate_beta(self) -> float:
        """Calculate beta against benchmark"""
        if self.benchmark is None or self.benchmark.empty:
            return 1.0
        
        # Align returns
        common_idx = self.returns.index.intersection(self.benchmark.index)
        if len(common_idx) < 2:
            return 1.0
        
        portfolio_returns = self.returns[common_idx]
        benchmark_returns = self.benchmark[common_idx]
        
        covariance = portfolio_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
        
        return beta
    
    def calculate_market_metrics(self) -> Dict:
        """Calculate market-relative metrics (requires benchmark)"""
        if self.benchmark is None or self.benchmark.empty:
            return {}
        
        # Align returns
        common_idx = self.returns.index.intersection(self.benchmark.index)
        if len(common_idx) < 2:
            return {}
        
        portfolio_returns = self.returns[common_idx]
        benchmark_returns = self.benchmark[common_idx]
        
        # Beta
        beta = self.calculate_beta()
        
        # Alpha (using CAPM)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        portfolio_mean = portfolio_returns.mean()
        benchmark_mean = benchmark_returns.mean()
        
        expected_return = risk_free_rate + beta * (benchmark_mean - risk_free_rate)
        alpha_daily = portfolio_mean - expected_return
        alpha_annual = alpha_daily * 252
        
        # R-squared
        correlation = portfolio_returns.corr(benchmark_returns)
        r_squared = correlation ** 2
        
        # Tracking error
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        
        # Information ratio
        active_return = portfolio_returns - benchmark_returns
        information_ratio = (active_return.mean() / active_return.std() * np.sqrt(252)) if active_return.std() != 0 else 0
        
        # Up/Down capture
        up_months = benchmark_returns[benchmark_returns > 0]
        down_months = benchmark_returns[benchmark_returns < 0]
        
        up_capture = (portfolio_returns[benchmark_returns > 0].mean() / up_months.mean()) if len(up_months) > 0 and up_months.mean() != 0 else 1
        down_capture = (portfolio_returns[benchmark_returns < 0].mean() / down_months.mean()) if len(down_months) > 0 and down_months.mean() != 0 else 1
        
        return {
            'Beta': beta,
            'Alpha (Daily)': alpha_daily,
            'Alpha (Annual)': alpha_annual,
            'R-Squared': r_squared,
            'Tracking Error': tracking_error,
            'Information Ratio': information_ratio,
            'Up Capture Ratio': up_capture,
            'Down Capture Ratio': down_capture
        }
    
    def get_complete_metrics(self) -> Dict:
        """Get all metrics combined"""
        metrics = {}
        metrics.update(self.calculate_returns_metrics())
        metrics.update(self.calculate_risk_metrics())
        metrics.update(self.calculate_risk_adjusted_metrics())
        metrics.update(self.calculate_market_metrics())
        return metrics
    
    def display_metrics_dashboard(self) -> Dict:
        """Display professional metrics dashboard"""
        metrics = self.get_complete_metrics()
        
        if not metrics:
            st.warning("No metrics available. Please check your data.")
            return metrics
        
        st.markdown("### 📊 Performance Dashboard")
        
        # Row 1: Return Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Total Return", f"{metrics.get('Total Return', 0):.2%}")
            st.metric("📊 CAGR", f"{metrics.get('CAGR', 0):.2%}")
        
        with col2:
            st.metric("🎯 Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.3f}")
            st.metric("⚡ Sortino Ratio", f"{metrics.get('Sortino Ratio', 0):.3f}")
        
        with col3:
            st.metric("📉 Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}")
            st.metric("📊 Annual Volatility", f"{metrics.get('Annual Volatility', 0):.2%}")
        
        with col4:
            st.metric("✅ Win Rate", f"{metrics.get('Win Rate', 0):.2%}")
            st.metric("📅 Positive Days", metrics.get('Positive Days', 0))
        
        # Row 2: Additional Metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'Beta' in metrics:
                st.metric("📊 Beta", f"{metrics['Beta']:.3f}")
            if 'Alpha (Annual)' in metrics:
                st.metric("⭐ Alpha", f"{metrics['Alpha (Annual)']:.2%}")
        
        with col2:
            st.metric("📉 VaR (95%)", f"{metrics.get('Value at Risk (95%)', 0):.2%}")
            st.metric("⚠️ CVaR (95%)", f"{metrics.get('Conditional VaR (95%)', 0):.2%}")
        
        with col3:
            st.metric("🏆 Calmar Ratio", f"{metrics.get('Calmar Ratio', 0):.3f}")
            if 'Omega Ratio' in metrics and metrics['Omega Ratio'] != float('inf'):
                st.metric("🔄 Omega Ratio", f"{metrics['Omega Ratio']:.3f}")
        
        with col4:
            st.metric("📈 Best Day", f"{metrics.get('Best Day', 0):.2%}")
            st.metric("📉 Worst Day", f"{metrics.get('Worst Day', 0):.2%}")
        
        return metrics
    
    def create_rolling_metrics_chart(self, window: int = 252) -> go.Figure:
        """Create rolling metrics chart"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rolling Sharpe Ratio (1Y)', 'Rolling Volatility (1Y)',
                           'Rolling Returns (1Y)', 'Rolling Beta (1Y)'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Calculate rolling metrics
        rolling_sharpe = self.returns.rolling(window).apply(
            lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() != 0 and x.std() is not None else 0
        )
        rolling_vol = self.returns.rolling(window).std() * np.sqrt(252)
        rolling_returns = self.returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
        
        # Rolling Beta if benchmark exists
        rolling_beta = pd.Series(index=self.returns.index, dtype=float)
        if self.benchmark is not None:
            aligned = pd.DataFrame({
                'portfolio': self.returns,
                'benchmark': self.benchmark
            }).dropna()
            
            for i in range(window, len(aligned)):
                subset = aligned.iloc[i-window:i]
                cov = subset['portfolio'].cov(subset['benchmark'])
                var = subset['benchmark'].var()
                rolling_beta.iloc[i] = cov / var if var != 0 and cov is not None else 1
        
        # Add traces
        fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, 
                                name='Sharpe Ratio', line=dict(color='blue', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, 
                                name='Volatility', line=dict(color='red', width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=rolling_returns.index, y=rolling_returns, 
                                name='Rolling Returns', line=dict(color='green', width=2)), row=2, col=1)
        
        if not rolling_beta.isna().all():
            fig.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, 
                                    name='Beta', line=dict(color='purple', width=2)), row=2, col=2)
        
        # Update layouts
        fig.update_layout(height=600, showlegend=False, template='plotly_white')
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", tickformat=".2%", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Return", tickformat=".2%", row=2, col=1)
        fig.update_yaxes(title_text="Beta", row=2, col=2)
        
        return fig
    
    def create_drawdown_chart(self) -> go.Figure:
        """Create drawdown chart"""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red', width=1.5),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown Analysis',
            yaxis_title='Drawdown (%)',
            xaxis_title='Date',
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        fig.update_yaxes(tickformat=".0f")
        fig.add_hline(y=-10, line_dash="dash", line_color="orange", 
                     annotation_text="-10%", annotation_position="bottom right")
        fig.add_hline(y=-20, line_dash="dash", line_color="red",
                     annotation_text="-20%", annotation_position="bottom right")
        
        return fig
    
    def create_monthly_heatmap(self) -> go.Figure:
        """Create monthly returns heatmap"""
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        if monthly_returns.empty:
            return go.Figure()
        
        # Create pivot table
        monthly_pivot = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot = monthly_pivot.pivot(index='Year', columns='Month', values='Return')
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = month_names[:len(pivot.columns)]
        
        fig = px.imshow(
            pivot * 100,
            text_auto='.1f',
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title="Monthly Returns (%)",
            labels={"x": "Month", "y": "Year", "color": "Return (%)"},
            zmin=-15, zmax=15
        )
        
        fig.update_layout(height=500, template='plotly_white')
        
        return fig
