"""
Realistic synthetic data generator for R_30 (Volatility 30) indices.
Mimics actual Deriv synthetic index behavior with realistic volatility, trends, and mean reversion.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_realistic_r30(
    start_date: datetime,
    end_date: datetime,
    initial_price: float = 1000.0,
    base_volatility: float = 0.0008,
    trend_strength: float = 0.0003,
    mean_reversion: float = 0.0001
) -> pd.DataFrame:
    """
    Generate realistic R_30 tick data.

    Market characteristics of R_30:
    - High volatility (1-second ticks)
    - Short-term trends (10-60 seconds)
    - Mean reversion around moving average
    - Bid-ask spread ~0.005%
    - No long-term drift (mean-reverting)

    Args:
        start_date: Start datetime
        end_date: End datetime
        initial_price: Starting price
        base_volatility: Base per-tick volatility (0.08%)
        trend_strength: Strength of short-term trends
        mean_reversion: Strength of mean reversion

    Returns:
        DataFrame with realistic tick data
    """
    total_seconds = int((end_date - start_date).total_seconds())
    timestamps = np.array([
        start_date.timestamp() + i for i in range(total_seconds)
    ])

    np.random.seed(42)  # Reproducible

    prices = np.zeros(total_seconds)
    prices[0] = initial_price

    # Parameters for realistic behavior
    ma_period = 20  # Moving average period for mean reversion
    trend_duration_min = 10
    trend_duration_max = 60

    # Track current trend
    current_trend = 0.0
    trend_ticks_remaining = 0

    for i in range(1, total_seconds):
        # Random walk component
        noise = np.random.normal(0, base_volatility)

        # Trend component (persistent drift for short periods)
        if trend_ticks_remaining <= 0:
            # Start new trend
            trend_direction = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            current_trend = trend_direction * trend_strength
            trend_ticks_remaining = np.random.randint(trend_duration_min, trend_duration_max)
        else:
            trend_ticks_remaining -= 1

        # Mean reversion component
        # Calculate short-term moving average
        if i >= ma_period:
            ma = np.mean(prices[i-ma_period:i])
            reversion = (ma - prices[i-1]) * mean_reversion
        else:
            reversion = 0.0

        # Combine components
        price_change = noise + current_trend + reversion
        prices[i] = prices[i-1] * (1 + price_change)

        # Occasionally add spikes (market shocks)
        if np.random.random() < 0.0001:  # 0.01% chance per tick
            spike = np.random.choice([-1, 1]) * np.random.uniform(0.002, 0.005)
            prices[i] *= (1 + spike)

    # Ensure no negative prices
    prices = np.maximum(prices, 0.01)

    # Create bid/ask spread (typical 0.005% for R_30)
    spread = 0.00005
    bids = prices * (1 - spread/2)
    asks = prices * (1 + spread/2)

    # High/Low (slightly beyond bid/ask)
    high_extra = np.random.uniform(0, 0.00002, total_seconds)
    low_extra = np.random.uniform(0, 0.00002, total_seconds)
    highs = np.maximum(bids, asks) * (1 + high_extra)
    lows = np.minimum(bids, asks) * (1 - low_extra)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'bid': bids,
        'ask': asks,
        'quote': prices,
        'high': highs,
        'low': lows,
        'volume': 1.0
    })

    return df

def generate_trending_r30(
    start_date: datetime,
    end_date: datetime,
    initial_price: float = 1000.0,
    trend_probability: float = 0.3
) -> pd.DataFrame:
    """
    Generate R_30 data with more frequent trends for strategy testing.
    Increases the probability of sustained trends to trigger signals.

    Args:
        trend_probability: Probability of trend starting each tick (higher = more trends)

    Returns:
        DataFrame with trending-heavy data
    """
    total_seconds = int((end_date - start_date).total_seconds())
    timestamps = np.array([
        start_date.timestamp() + i for i in range(total_seconds)
    ])

    np.random.seed(42)

    prices = np.zeros(total_seconds)
    prices[0] = initial_price

    base_vol = 0.0008
    trend_strength = 0.0006  # Stronger trends
    trend_duration_min = 15
    trend_duration_max = 120  # Longer trends

    current_trend = 0.0
    trend_ticks_remaining = 0

    for i in range(1, total_seconds):
        noise = np.random.normal(0, base_vol)

        # More frequent trend initiation
        if trend_ticks_remaining <= 0 and np.random.random() < trend_probability:
            trend_direction = np.random.choice([-1, 1], p=[0.5, 0.5])  # Only up/down, no flat
            current_trend = trend_direction * trend_strength
            trend_ticks_remaining = np.random.randint(trend_duration_min, trend_duration_max)
        elif trend_ticks_remaining > 0:
            trend_ticks_remaining -= 1
        else:
            current_trend = 0.0

        price_change = noise + current_trend
        prices[i] = prices[i-1] * (1 + price_change)

    prices = np.maximum(prices, 0.01)

    spread = 0.00005
    bids = prices * (1 - spread/2)
    asks = prices * (1 + spread/2)
    highs = np.maximum(bids, asks) * (1 + np.random.uniform(0, 0.00002, total_seconds))
    lows = np.minimum(bids, asks) * (1 - np.random.uniform(0, 0.00002, total_seconds))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'bid': bids,
        'ask': asks,
        'quote': prices,
        'high': highs,
        'low': lows,
        'volume': 1.0
    })

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate realistic R_30 synthetic data')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-01-07', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--mode', type=str, default='realistic',
                       choices=['realistic', 'trending'],
                       help='Data generation mode')
    parser.add_argument('--initial-price', type=float, default=1000.0, help='Starting price')

    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')

    print(f"Generating {args.mode} R_30 data...")
    print(f"Period: {start.date()} to {end.date()}")

    if args.mode == 'realistic':
        df = generate_realistic_r30(start, end, args.initial_price)
    else:
        df = generate_trending_r30(start, end, args.initial_price)

    df.to_csv(args.output, index=False)

    print(f"✓ Generated {len(df)} ticks")
    print(f"  Price range: ${df['quote'].min():.2f} - ${df['quote'].max():.2f}")
    print(f"  Saved to: {args.output}")

    # Calculate stats
    returns = np.diff(df['quote']) / df['quote'][:-1]
    print(f"\nStatistics:")
    print(f"  Mean return: {returns.mean():.6f}")
    print(f"  Std dev: {returns.std():.6f}")
    print(f"  Annualized vol: {returns.std() * np.sqrt(86400) * 100:.2f}%")
    print(f"  Max drawdown: {(df['quote'].max() - df['quote'].min()) / df['quote'].max() * 100:.2f}%")