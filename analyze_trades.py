"""
Analyze individual trades from backtest to understand loss patterns.
"""

import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backtest import Backtester, BacktestConfig
from config import load_config
from datetime import datetime

async def analyze_trades():
    print("="*60)
    print("TRADE ANALYSIS")
    print("="*60)

    # Load dataset
    df = pd.read_csv('data/signal_rich_10k.csv')
    df = df.head(5000)

    # Load config with optimized params
    config = load_config()
    config['strategy']['atr_tp_multiplier'] = 3.0
    config['strategy']['atr_sl_multiplier'] = 1.0

    backtest_config = BacktestConfig(
        start_date=datetime.fromtimestamp(df['timestamp'].iloc[0]),
        end_date=datetime.fromtimestamp(df['timestamp'].iloc[-1]),
        initial_balance=10000.0,
        enable_risk_manager=True
    )

    backtester = Backtester(config, backtest_config)
    result = await backtester.run(df)

    print(f"\nTotal trades: {len(result.trades)}")
    print("\n" + "-"*60)
    print("INDIVIDUAL TRADES:")
    print("-"*60)

    for i, trade in enumerate(result.trades, 1):
        profit = trade['profit']
        entry = trade['entry_price']
        exit_price = trade['exit_price']
        sl = trade['stop_loss']
        tp = trade['take_profit']
        direction = trade['direction']
        reason = trade['exit_reason']

        print(f"\nTrade {i}:")
        print(f"  Direction: {direction}")
        print(f"  Entry: ${entry:.2f}")
        print(f"  SL: ${sl:.2f} (${abs(entry-sl):.2f} away)")
        print(f"  TP: ${tp:.2f} (${abs(tp-entry):.2f} away)")
        print(f"  Exit: ${exit_price:.2f} @ {trade['exit_time']}")
        print(f"  P&L: ${profit:.2f}")
        print(f"  Reason: {reason}")

        # Calculate R:R actually achieved
        if profit > 0:
            rr = profit / abs(entry - sl) if abs(entry - sl) > 0 else 0
            print(f"  Actual R:R: {rr:.2f}")
        else:
            rr = profit / abs(entry - sl) if abs(entry - sl) > 0 else 0
            print(f"  Actual R:R: {rr:.2f} (loss)")

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    profits = [t['profit'] for t in result.trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]

    print(f"Wins: {len(wins)} (avg: ${np.mean(wins):.2f})")
    print(f"Losses: {len(losses)} (avg: ${np.mean(losses):.2f})")
    print(f"Win rate: {len(wins)/len(profits)*100:.1f}%")
    print(f"Avg Win / Avg Loss ratio: {np.mean(wins)/abs(np.mean(losses)):.2f}")

    # Check if losses are hitting full stop
    full_sl_hits = 0
    for trade in result.trades:
        if trade['profit'] < 0:
            # Check if exit price equals stop loss (within tolerance)
            if abs(trade['exit_price'] - trade['stop_loss']) < 0.01:
                full_sl_hits += 1

    print(f"Full stop loss hits: {full_sl_hits}/{len(losses)}")

if __name__ == "__main__":
    import asyncio
    import numpy as np
    asyncio.run(analyze_trades())