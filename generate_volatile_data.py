"""
Generate more volatile synthetic data for backtesting.
Creates data with clear trends to trigger strategy signals.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_volatile_ticks(start_date, end_date, initial_price=1000.0, volatility=0.001):
    """
    Generate volatile tick data with trends.

    Args:
        start_date: Start datetime
        end_date: End datetime
        initial_price: Starting price
        volatility: Per-tick volatility (0.1% = 0.001)

    Returns:
        DataFrame with tick data
    """
    total_seconds = int((end_date - start_date).total_seconds())
    timestamps = [start_date.timestamp() + i for i in range(total_seconds)]

    np.random.seed(42)
    prices = [initial_price]

    # Create some trends by having periods of positive/negative drift
    drift = 0.0
    for i in range(1, total_seconds):
        # Change drift every ~1000 seconds to create trends
        if i % 1000 == 0:
            drift = np.random.choice([-0.0005, 0.0, 0.0005])

        # Add random noise
        noise = np.random.normal(0, volatility)
        price = prices[-1] * (1 + drift + noise)
        prices.append(price)

    prices = np.array(prices)

    # Create bid/ask spread
    spread = 0.00005
    bids = prices * (1 - spread/2)
    asks = prices * (1 + spread/2)

    # High/Low
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
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 2)

    print("Generating volatile synthetic data...")
    df = generate_volatile_ticks(start, end, volatility=0.001)

    output_file = "data/volatile_r30.csv"
    df.to_csv(output_file, index=False)

    print(f"✓ Generated {len(df)} ticks")
    print(f"  Price range: ${df['quote'].min():.2f} - ${df['quote'].max():.2f}")
    print(f"  Saved to: {output_file}")

    # Show some stats
    returns = np.diff(df['quote']) / df['quote'][:-1]
    print(f"  Avg return: {returns.mean():.6f}")
    print(f"  Std dev: {returns.std():.6f}")
    print(f"  Max drawdown: {(df['quote'].max() - df['quote'].min()) / df['quote'].max() * 100:.2f}%")