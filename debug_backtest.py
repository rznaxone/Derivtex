"""
Debug backtest to see why no signals are generated.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from indicators import Indicators
from strategy import Strategy, SignalType
from risk_manager import RiskManager
from config import load_config

def debug_backtest():
    print("="*60)
    print("DEBUG BACKTEST")
    print("="*60)

    # Load signal-rich data
    df = pd.read_csv('data/signal_rich_10k.csv')
    print(f"Loaded {len(df)} ticks")

    config = load_config()
    risk_manager = RiskManager(config, 10000.0)
    strategy = Strategy(config, risk_manager)

    signals_generated = 0

    for idx, row in df.iterrows():
        tick = row.to_dict()

        # Update indicators
        values = strategy.indicators.update(tick)

        # Check risk
        can_trade, reason = risk_manager.can_trade()
        if not can_trade:
            if idx < 100:  # Only log first few
                print(f"Tick {idx}: Blocked by risk - {reason}")
            continue

        # Check trading hours
        from datetime import time as dt_time
        tick_dt = datetime.fromtimestamp(tick['timestamp'])
        trading_hours_start = dt_time(8, 0)
        trading_hours_end = dt_time(20, 0)
        current_time = tick_dt.time()

        # Simple check: is it within trading hours?
        in_trading_hours = trading_hours_start <= current_time < trading_hours_end
        if not in_trading_hours and idx < 100:
            print(f"Tick {idx}: Outside trading hours {current_time}")
            continue

        # Generate signal using internal method
        signal = strategy._generate_signal(values, tick)

        if signal and signal.type not in [SignalType.HOLD, SignalType.NONE]:
            signals_generated += 1
            print(f"\nTick {idx}: SIGNAL!")
            print(f"  Type: {signal.type.value}")
            print(f"  Price: {tick['quote']:.2f}")
            print(f"  EMA Fast: {values.ema_fast:.2f}, EMA Slow: {values.ema_slow:.2f}")
            print(f"  RSI: {values.rsi:.1f}, ADX: {values.adx:.1f}")
            print(f"  Regime: {values.regime.value}")
            print(f"  Crossover: {strategy.indicators.check_crossover()}")
            print(f"  Reason: {signal.reason}")

            if signals_generated >= 5:
                print("\n... (more signals possible)")
                break

    print("\n" + "="*60)
    print(f"Total signals generated: {signals_generated}")
    print("="*60)

    if signals_generated == 0:
        print("\n⚠️  DIAGNOSIS:")
        print("  1. Check if EMA crossovers are being detected")
        print("  2. Check if RSI conditions are met")
        print("  3. Check if ADX conditions are met")
        print("  4. Check if trading hours are blocking")

        # Print final indicator state
        print(f"\nFinal indicator state:")
        print(f"  EMA Fast: {strategy.indicators._ema_fast:.2f}")
        print(f"  EMA Slow: {strategy.indicators._ema_slow:.2f}")
        print(f"  RSI: {strategy.indicators._rsi:.1f}")
        print(f"  ADX: {strategy.indicators._adx:.1f}")
        print(f"  Regime: {strategy.indicators._determine_regime().value}")

if __name__ == "__main__":
    debug_backtest()