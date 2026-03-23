"""
Streamlit dashboard for Derivtex.
Real-time monitoring and analytics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot import DerivBot
from config import load_config
from logger import setup_logger

logger = setup_logger(__name__)

# Page config
st.set_page_config(
    page_title="Derivtex Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bot():
    """Load bot instance (cached)."""
    try:
        config = load_config()
        bot = DerivBot(config)
        return bot
    except Exception as e:
        st.error(f"Failed to load bot: {e}")
        return None

def load_trades():
    """Load trades from file."""
    trades_file = Path(__file__).parent.parent / 'logs' / 'trades.json'
    if trades_file.exists():
        try:
            with open(trades_file, 'r') as f:
                return pd.DataFrame(json.load(f))
        except Exception as e:
            st.error(f"Failed to load trades: {e}")
    return pd.DataFrame()

def load_metrics():
    """Load metrics from file."""
    metrics_file = Path(__file__).parent.parent / 'logs' / 'metrics.json'
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load metrics: {e}")
    return {}

def create_equity_curve(df):
    """Create equity curve chart."""
    if df.empty:
        return None

    df['exit_time_dt'] = pd.to_datetime(df['exit_time'], unit='s')
    df = df.sort_values('exit_time_dt')
    df['cumulative_profit'] = df['profit'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['exit_time_dt'],
        y=df['cumulative_profit'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='#1f77b4', width=2)
    ))

    # Add drawdown shading
    running_max = df['cumulative_profit'].cummax()
    drawdown = (df['cumulative_profit'] - running_max) / running_max * 100
    fig.add_trace(go.Scatter(
        x=df['exit_time_dt'],
        y=drawdown,
        mode='lines',
        name='Drawdown %',
        line=dict(color='red', width=1),
        yaxis='y2',
        fill='tozeroy',
        opacity=0.3
    ))

    fig.update_layout(
        title='Equity Curve & Drawdown',
        xaxis_title='Date',
        yaxis_title='Cumulative Profit ($)',
        yaxis2=dict(
            title='Drawdown %',
            overlaying='y',
            side='right',
            range=[-100, 0]
        ),
        height=400
    )

    return fig

def create_win_loss_pie(df):
    """Create win/loss pie chart."""
    if df.empty:
        return None

    wins = len(df[df['profit'] > 0])
    losses = len(df[df['profit'] < 0])
    total = wins + losses

    if total == 0:
        return None

    fig = go.Figure(data=[go.Pie(
        labels=['Wins', 'Losses'],
        values=[wins, losses],
        hole=.4,
        marker_colors=['#28a745', '#dc3545']
    )])

    fig.update_layout(
        title=f'Win/Loss Distribution (Win Rate: {wins/total*100:.1f}%)',
        height=300
    )

    return fig

def create_profit_distribution(df):
    """Create profit distribution histogram."""
    if df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['profit'],
        nbinsx=50,
        marker_color='#1f77b4',
        opacity=0.7
    ))

    fig.update_layout(
        title='Profit Distribution',
        xaxis_title='Profit ($)',
        yaxis_title='Frequency',
        height=300
    )

    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="red")

    return fig

def main():
    """Main dashboard."""
    st.title("📈 Derivtex Dashboard")
    st.markdown("Real-time monitoring for Deriv Volatility 30 trading bot")

    # Sidebar controls
    st.sidebar.header("Controls")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    # Load data
    trades_df = load_trades()
    metrics = load_metrics()

    # Top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Trades",
            metrics.get('total_trades', 0),
            delta=None
        )

    with col2:
        win_rate = metrics.get('win_rate', 0) * 100
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=None
        )

    with col3:
        net_profit = metrics.get('net_profit', 0)
        delta_color = "normal" if net_profit >= 0 else "inverse"
        st.metric(
            "Net Profit",
            f"${net_profit:.2f}",
            delta=None,
            delta_color=delta_color
        )

    with col4:
        profit_factor = metrics.get('profit_factor', 0)
        st.metric(
            "Profit Factor",
            f"{profit_factor:.2f}",
            delta=None
        )

    with col5:
        max_dd = metrics.get('max_drawdown', 0) * 100
        st.metric(
            "Max Drawdown",
            f"{max_dd:.1f}%",
            delta=None
        )

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        equity_fig = create_equity_curve(trades_df)
        if equity_fig:
            st.plotly_chart(equity_fig, use_container_width=True)
        else:
            st.info("No trade data available for equity curve")

    with col2:
        pie_fig = create_win_loss_pie(trades_df)
        if pie_fig:
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No trade data available for win/loss chart")

    # Second row
    col1, col2 = st.columns(2)

    with col1:
        dist_fig = create_profit_distribution(trades_df)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)
        else:
            st.info("No trade data available for distribution")

    with col2:
        st.subheader("📊 Detailed Statistics")

        if metrics:
            stats_df = pd.DataFrame({
                'Metric': [
                    'Total Trades',
                    'Winning Trades',
                    'Losing Trades',
                    'Total Profit',
                    'Total Loss',
                    'Net Profit',
                    'Win Rate',
                    'Avg Win',
                    'Avg Loss',
                    'Profit Factor',
                    'Sharpe Ratio',
                    'Max Drawdown'
                ],
                'Value': [
                    metrics.get('total_trades', 0),
                    metrics.get('winning_trades', 0),
                    metrics.get('losing_trades', 0),
                    f"${metrics.get('total_profit', 0):.2f}",
                    f"${metrics.get('total_loss', 0):.2f}",
                    f"${metrics.get('net_profit', 0):.2f}",
                    f"{metrics.get('win_rate', 0)*100:.1f}%",
                    f"${metrics.get('avg_win', 0):.2f}",
                    f"${metrics.get('avg_loss', 0):.2f}",
                    f"{metrics.get('profit_factor', 0):.2f}",
                    f"{metrics.get('sharpe_ratio', 0):.2f}",
                    f"{metrics.get('max_drawdown', 0)*100:.1f}%"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No metrics available")

    st.markdown("---")

    # Recent trades table
    st.subheader("📋 Recent Trades")

    if not trades_df.empty:
        display_df = trades_df.tail(50).copy()
        display_df['entry_time'] = pd.to_datetime(display_df['entry_time'], unit='s')
        display_df['exit_time'] = pd.to_datetime(display_df['exit_time'], unit='s')

        # Format columns
        display_df['profit'] = display_df['profit'].apply(lambda x: f"${x:.2f}")
        display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"{x:.5f}")
        display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"{x:.5f}")

        st.dataframe(
            display_df[[
                'id', 'direction', 'entry_time', 'exit_time',
                'entry_price', 'exit_price', 'profit', 'exit_reason'
            ]].sort_values('exit_time', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No trades recorded yet")

    # Footer
    st.markdown("---")
    st.caption("Derivtex Dashboard | Last updated: " + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + " UTC")

if __name__ == "__main__":
    main()