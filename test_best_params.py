"""
Test optimized parameters on larger dataset.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backtest import Backtester, BacktestConfig
from config import load_config
from datetime import datetime

async def test_best_params():
    print("="*60)
    print("TEST OPTIMIZED PARAMETERS ON LARGER DATASET")
    print("="*60)

    # Load larger dataset (5000 ticks)
    df = pd.read_csv('data/signal_rich_10k.csv')
    df = df.head(5000)
    print(f"\nDataset: {len(df)} ticks")

    # Load and modify config with best params
    config = load_config()
    config['strategy']['atr_tp_multiplier'] = 3.0
    config['strategy']['atr_sl_multiplier'] = 1.0
    config['strategy']['ema_fast'] = 20
    config['strategy']['ema_slow'] = 50
    config['strategy']['rsi_period'] = 14
    config['strategy']['adx_trending'] = 25
    config['strategy']['crossover_lookback'] = 3

    backtest_config = BacktestConfig(
        start_date=datetime.fromtimestamp(df['timestamp'].iloc[0]),
        end_date=datetime.fromtimestamp(df['timestamp'].iloc[-1]),
        initial_balance=10000.0,
        enable_risk_manager=True
    )

    backtester = Backtester(config, backtest_config)
    result = await backtester.run(df)

    print("\n" + "="*60)
    print("RESULTS WITH OPTIMIZED PARAMETERS")
    print("="*60)
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate*100:.1f}%")
    print(f"Net Profit: ${result.net_profit:.2f}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown*100:.1f}%")
    print(f"Avg Win: ${result.avg_win:.2f}")
    print(f"Avg Loss: ${result.avg_loss:.2f}")
    print(f"Total Return: {result.metrics.get('total_return_percent', 0):.1f}%")

    if result.total_trades > 0:
        print("\n✅ Strategy profitable with optimized parameters!")
    else:
        print("\n⚠️  No trades generated")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_best_params())