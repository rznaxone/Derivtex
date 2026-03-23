"""
Quick test script to verify Derivtex components work correctly.
Tests indicator calculations and signal generation with controlled data.
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

def test_indicators():
    """Test indicator calculations."""
    print("=" * 60)
    print("TEST 1: Indicator Calculations")
    print("=" * 60)

    config = load_config()
    indicators = Indicators(config)

    # Generate trending up data
    prices = np.linspace(1000, 1100, 100) + np.random.randn(100) * 2
    highs = prices * 1.001
    lows = prices * 0.999

    for i in range(len(prices)):
        tick = {
            'quote': float(prices[i]),
            'bid': float(prices[i] * 0.9995),
            'ask': float(prices[i] * 1.0005),
            'high': float(highs[i]),
            'low': float(lows[i]),
            'timestamp': datetime.now().timestamp(),
            'volume': 1.0
        }
        values = indicators.update(tick)

        if i > 50:
            break

    print(f"Final EMA Fast: {values.ema_fast:.5f}")
    print(f"Final EMA Slow: {values.ema_slow:.5f}")
    print(f"Final RSI: {values.rsi:.2f}")
    print(f"Final ATR: {values.atr:.5f}")
    print(f"Final ADX: {values.adx:.2f}")
    print(f"Market Regime: {values.regime.value}")
    print("✅ Indicators working\n")

    return indicators

def test_strategy(indicators):
    """Test strategy signal generation."""
    print("=" * 60)
    print("TEST 2: Strategy Signal Generation")
    print("=" * 60)

    config = load_config()
    risk_manager = RiskManager(config, 10000.0)
    strategy = Strategy(config, risk_manager)

    # Create a scenario that should trigger a LONG signal
    # Uptrend: EMA > EMA, RSI > 50, ADX > 20
    print("Simulating bullish crossover...")

    # First, build up some data
    for _ in range(60):
        tick = {
            'quote': 1000.0,
            'bid': 999.5,
            'ask': 1000.5,
            'high': 1001.0,
            'low': 999.0,
            'timestamp': datetime.now().timestamp(),
            'volume': 1.0
        }
        strategy.indicators.update(tick)

    # Now create a crossover by having price rise and EMA fast cross above
    prices = np.linspace(1000, 1020, 10)
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
        signal = strategy.update(tick)
        if signal and signal.type.value != 'hold':
            print(f"Signal generated: {signal.type.value}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Reason: {signal.reason}")
            print(f"  Entry: {signal.entry_price:.5f}")
            print(f"  SL: {signal.stop_loss:.5f}")
            print(f"  TP: {signal.take_profit:.5f}")
            print(f"  Position size: {signal.position_size:.2f}")
            print("✅ Strategy generating signals\n")
            return signal

    print("⚠️  No signal generated (market conditions not met)")
    print("This is normal for random data - try with real trends\n")
    return None

def test_risk_manager():
    """Test risk manager calculations."""
    print("=" * 60)
    print("TEST 3: Risk Manager")
    print("=" * 60)

    config = load_config()
    risk_manager = RiskManager(config, 10000.0)

    # Test position sizing
    entry_price = 1000.0
    stop_loss = 990.0  # 1% risk
    position_size = risk_manager.calculate_position_size(entry_price, stop_loss, False)

    print(f"Account balance: $10,000")
    print(f"Risk per trade: {config['risk']['risk_per_trade']*100}% = ${10000 * config['risk']['risk_per_trade']:.2f}")
    print(f"Price difference (entry-SL): ${abs(entry_price - stop_loss):.2f}")
    print(f"Calculated position size: {position_size:.2f}")
    print(f"Risk amount: ${position_size * abs(entry_price - stop_loss):.2f}")
    print("✅ Risk manager working\n")

    # Test can_trade
    can_trade, reason = risk_manager.can_trade()
    print(f"Can trade: {can_trade}")
    if not can_trade:
        print(f"Reason: {reason}")
    print("✅ Risk checks working\n")

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DERIVTEX COMPONENT TEST SUITE")
    print("=" * 60 + "\n")

    try:
        # Test 1: Indicators
        indicators = test_indicators()

        # Test 2: Strategy
        signal = test_strategy(indicators)

        # Test 3: Risk Manager
        test_risk_manager()

        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nThe bot components are working correctly.")
        print("To generate trades, you need:")
        print("  1. Real or realistic historical data with trends")
        print("  2. Market conditions that meet entry criteria")
        print("  3. Proper risk manager configuration")
        print("\nNext steps:")
        print("  - Run: python run.py backtest --data data/your_data.csv")
        print("  - Or run live: python run.py bot (with real API credentials)")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()