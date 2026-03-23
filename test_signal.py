"""
Test to force a signal by creating ideal market conditions.
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from indicators import Indicators
from strategy import Strategy
from risk_manager import RiskManager
from config import load_config

def force_bullish_signal():
    """Force a bullish signal by creating perfect conditions."""
    print("=" * 60)
    print("FORCE BULLISH SIGNAL TEST")
    print("=" * 60)

    config = load_config()
    risk_manager = RiskManager(config, 10000.0)
    strategy = Strategy(config, risk_manager)
    indicators = strategy.indicators

    # Build initial data (downtrend to set up EMAs)
    print("Phase 1: Building downtrend (EMA fast < EMA slow)...")
    prices = np.linspace(1020, 1000, 50)  # Downtrend
    for price in prices:
        tick = {
            'quote': float(price),
            'bid': float(price * 0.9995),
            'ask': float(price * 1.0005),
            'high': float(price * 1.001),
            'low': float(price * 0.999),
            'timestamp': datetime.now().timestamp(),
            'volume': 1.0
        }
        indicators.update(tick)

    print(f"  EMA Fast: {indicators._ema_fast:.2f}, EMA Slow: {indicators._ema_slow:.2f}")
    print(f"  RSI: {indicators._rsi:.2f}")

    # Now create a sharp uptrend to cause crossover
    print("\nPhase 2: Sharp uptrend (crossover)...")
    crossover_signal = None

    # Rapid price increase
    prices = np.linspace(1000, 1040, 10)
    for i, price in enumerate(prices):
        tick = {
            'quote': float(price),
            'bid': float(price * 0.9995),
            'ask': float(price * 1.0005),
            'high': float(price * 1.001),
            'low': float(price * 0.999),
            'timestamp': datetime.now().timestamp(),
            'volume': 1.0
        }

        # Update indicators
        values = indicators.update(tick)

        # Check crossover
        crossover = indicators.check_crossover(lookback=3)
        if crossover == "bullish":
            print(f"  Tick {i}: Crossover detected! EMA Fast: {values.ema_fast:.2f}, EMA Slow: {values.ema_slow:.2f}")

        # Try to generate signal
        signal = strategy.update(tick)
        if signal and signal.type.value != 'hold':
            crossover_signal = signal
            print(f"\n✅ SIGNAL GENERATED!")
            print(f"  Type: {signal.type.value}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Reason: {signal.reason}")
            print(f"  Entry: ${signal.entry_price:.2f}")
            print(f"  Stop Loss: ${signal.stop_loss:.2f}")
            print(f"  Take Profit: ${signal.take_profit:.2f}")
            print(f"  Position Size: {signal.position_size:.2f}")
            print(f"  High Probability: {signal.high_probability}")
            return signal

    print("\n⚠️  No signal generated even with uptrend")
    print(f"  Final EMA Fast: {indicators._ema_fast:.2f}, EMA Slow: {indicators._ema_slow:.2f}")
    print(f"  Final RSI: {indicators._rsi:.2f}")
    print(f"  Final ADX: {indicators._adx:.2f}")
    print(f"  Regime: {indicators._determine_regime().value}")
    return None

def force_bearish_signal():
    """Force a bearish signal."""
    print("\n" + "=" * 60)
    print("FORCE BEARISH SIGNAL TEST")
    print("=" * 60)

    config = load_config()
    risk_manager = RiskManager(config, 10000.0)
    strategy = Strategy(config, risk_manager)
    indicators = strategy.indicators

    # Build initial data (uptrend)
    print("Phase 1: Building uptrend (EMA fast > EMA slow)...")
    prices = np.linspace(1000, 1020, 50)  # Uptrend
    for price in prices:
        tick = {
            'quote': float(price),
            'bid': float(price * 0.9995),
            'ask': float(price * 1.0005),
            'high': float(price * 1.001),
            'low': float(price * 0.999),
            'timestamp': datetime.now().timestamp(),
            'volume': 1.0
        }
        indicators.update(tick)

    print(f"  EMA Fast: {indicators._ema_fast:.2f}, EMA Slow: {indicators._ema_slow:.2f}")
    print(f"  RSI: {indicators._rsi:.2f}")

    # Sharp downtrend
    print("\nPhase 2: Sharp downtrend (crossover)...")
    prices = np.linspace(1020, 980, 10)
    for i, price in enumerate(prices):
        tick = {
            'quote': float(price),
            'bid': float(price * 0.9995),
            'ask': float(price * 1.0005),
            'high': float(price * 1.001),
            'low': float(price * 0.999),
            'timestamp': datetime.now().timestamp(),
            'volume': 1.0
        }

        signal = strategy.update(tick)
        if signal and signal.type.value != 'hold':
            print(f"\n✅ SIGNAL GENERATED!")
            print(f"  Type: {signal.type.value}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Reason: {signal.reason}")
            print(f"  Entry: ${signal.entry_price:.2f}")
            print(f"  Stop Loss: ${signal.stop_loss:.2f}")
            print(f"  Take Profit: ${signal.take_profit:.2f}")
            return signal

    print("\n⚠️  No signal generated")
    return None

def main():
    print("\n" + "=" * 60)
    print("DERIVTEX SIGNAL GENERATION TEST")
    print("=" * 60 + "\n")

    bullish = force_bullish_signal()
    bearish = force_bearish_signal()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Bullish signal: {'✅ Generated' if bullish else '❌ Not generated'}")
    print(f"Bearish signal: {'✅ Generated' if bearish else '❌ Not generated'}")

    if bullish or bearish:
        print("\n✅ Strategy is capable of generating signals!")
        print("   The backtest with random data didn't produce trades because")
        print("   the synthetic data lacked clear trends. Use real historical")
        print("   data or more volatile synthetic data for backtesting.")
    else:
        print("\n⚠️  Strategy not generating signals. Check entry conditions.")

if __name__ == "__main__":
    main()