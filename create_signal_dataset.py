"""
Create a dataset guaranteed to produce multiple trading signals.
Uses alternating trends to force EMA crossovers.
"""

import numpy as np
import pandas as pd
from datetime import datetime

def create_signal_rich_dataset(total_ticks: int = 10000) -> pd.DataFrame:
    """
    Create data with many clear trends to trigger signals.

    Strategy: Alternating uptrends and downtrends every ~200 ticks
    Each trend is strong enough to cause EMA crossover.
    Realistic R_30 behavior: max 2% move per tick, no gaps.
    """
    timestamps = np.arange(total_ticks) + datetime(2024, 1, 1).timestamp()

    np.random.seed(42)
    prices = np.zeros(total_ticks)
    prices[0] = 1000.0

    # Create alternating trends
    trend_length = 200
    current_idx = 0

    while current_idx < total_ticks:
        # Uptrend or downtrend (alternating)
        direction = 1 if (current_idx // trend_length) % 2 == 0 else -1

        # Strong trend for this segment (0.08% per tick)
        trend_price = direction * 0.0008
        segment_end = min(current_idx + trend_length, total_ticks)

        for i in range(current_idx, segment_end):
            noise = np.random.normal(0, 0.0003)  # Low noise
            price_change = trend_price + noise

            # Cap extreme moves to max 2% per tick (realistic for R_30)
            price_change = np.clip(price_change, -0.02, 0.02)

            prices[i] = prices[i-1] * (1 + price_change) if i > 0 else prices[0] * (1 + price_change)

            # Ensure price stays in realistic range (no gaps)
            if i > 0:
                max_change = 0.02  # 2% max per tick
                min_price = prices[i-1] * (1 - max_change)
                max_price = prices[i-1] * (1 + max_change)
                prices[i] = np.clip(prices[i], min_price, max_price)

        current_idx = segment_end

    # Add some choppy sections to create ranging conditions
    choppy_start = 2000
    choppy_length = 500
    if choppy_start + choppy_length < total_ticks:
        for i in range(choppy_start, choppy_start + choppy_length):
            noise = np.random.normal(0, 0.0005)
            noise = np.clip(noise, -0.015, 0.015)  # Cap choppy moves
            prices[i] = prices[i-1] * (1 + noise)

    # Ensure no negative
    prices = np.maximum(prices, 10.0)

    # Create OHLC
    spread = 0.00005
    bids = prices * (1 - spread/2)
    asks = prices * (1 + spread/2)
    highs = np.maximum(bids, asks) * (1 + np.random.uniform(0, 0.00002, total_ticks))
    lows = np.minimum(bids, asks) * (1 - np.random.uniform(0, 0.00002, total_ticks))

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
    print("Creating signal-rich dataset...")

    # Create 10,000 ticks (should produce multiple signals)
    df = create_signal_rich_dataset(10000)

    output = "data/signal_rich_10k.csv"
    df.to_csv(output, index=False)

    print(f"✓ Generated {len(df)} ticks")
    print(f"  Price range: ${df['quote'].min():.2f} - ${df['quote'].max():.2f}")
    print(f"  Saved to: {output}")

    # Count trend changes
    returns = np.diff(df['quote'])
    sign_changes = ((returns[1:] * returns[:-1]) < 0).sum()
    print(f"  Trend direction changes: {sign_changes}")
    print("\nThis dataset should generate multiple trades!")