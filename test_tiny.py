"""
Test backtest with just first 100 ticks to see position sizing.
"""

import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backtest import Backtester, BacktestConfig
from config import load_config
from datetime import datetime

async def tiny_test():
    df = pd.read_csv('data/signal_rich_10k.csv')
    df = df.head(100)

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

    print(f"\nTrades: {len(result.trades)}")
    for t in result.trades:
        print(f"  {t['id']}: size={t['position_size']:.2f}, profit={t['profit']:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(tiny_test())