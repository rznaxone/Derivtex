"""
Minimal backtest test with controlled data to verify trades are generated.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from indicators import Indicators
from strategy import Strategy, SignalType
from risk_manager import RiskManager
from config import load_config

def create_trending_data(n_ticks=200, trend='up'):
    """Create data with a clear trend to trigger signals."""
    timestamps = [datetime(2024, 1, 1).timestamp() + i for i in range(n_ticks)]

    if trend == 'up':
        # Strong uptrend
        prices = np.linspace(1000, 1050, n_ticks) + np.random.randn(n_ticks) * 0.5
    else:
        # Strong downtrend
        prices = np.linspace(1050, 1000, n_ticks) + np.random.randn(n_ticks) * 0.5

    spread = 0.00005
    bids = prices * (1 - spread/2)
    asks = prices * (1 + spread/2)
    highs = np.maximum(bids, asks) * 1.001
    lows = np.minimum(bids, asks) * 0.999

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

def test_backtest_with_trend():
    """Test that backtest logic can generate trades with trending data."""
    print("=" * 60)
    print("BACKTEST LOGIC TEST WITH TRENDING DATA")
    print("=" * 60)

    config = load_config()
    risk_manager = RiskManager(config, 10000.0)
    strategy = Strategy(config, risk_manager)

    # Test uptrend (should generate LONG signals)
    print("\nTest 1: Uptrend data")
    df_up = create_trending_data(200, 'up')
    signals_up = []

    for idx, row in df_up.iterrows():
        tick = row.to_dict()
        values = strategy.indicators.update(tick)

        # Check risk
        can_trade, reason = risk_manager.can_trade()
        if not can_trade:
            continue

        signal = strategy._generate_signal(values, tick)
        if signal and signal.type not in [SignalType.HOLD, SignalType.NONE]:
            signals_up.append(signal)
            print(f"  Tick {idx}: {signal.type.value} signal (RSI={values.rsi:.1f}, ADX={values.adx:.1f})")

    print(f"  Total signals in uptrend: {len(signals_up)}")

    # Test downtrend (should generate SHORT signals)
    print("\nTest 2: Downtrend data")
    df_down = create_trending_data(200, 'down')
    signals_down = []

    # Reset indicators
    strategy.indicators = Indicators(config)
    risk_manager = RiskManager(config, 10000.0)
    strategy.risk_manager = risk_manager

    for idx, row in df_down.iterrows():
        tick = row.to_dict()
        values = strategy.indicators.update(tick)

        can_trade, reason = risk_manager.can_trade()
        if not can_trade:
            continue

        signal = strategy._generate_signal(values, tick)
        if signal and signal.type not in [SignalType.HOLD, SignalType.NONE]:
            signals_down.append(signal)
            print(f"  Tick {idx}: {signal.type.value} signal (RSI={values.rsi:.1f}, ADX={values.adx:.1f})")

    print(f"  Total signals in downtrend: {len(signals_down)}")

    print("\n" + "=" * 60)
    if len(signals_up) > 0 or len(signals_down) > 0:
        print("✅ SUCCESS: Strategy generates signals with trending data!")
        print("   The backtest should produce trades with real trending data.")
    else:
        print("❌ FAILURE: No signals generated even with clear trends.")
        print("   Check entry conditions and indicator calculations.")

    print("=" * 60)

if __name__ == "__main__":
    test_backtest_with_trend()