"""
Fresh backtest with 5000 ticks to verify position sizing fix.
"""

import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from backtest import Backtester, BacktestConfig
from config import load_config
from datetime import datetime
import asyncio

async def run():
    df = pd.read_csv('data/signal_rich_10k.csv').head(5000)
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

    print(f"\nTotal trades: {result.total_trades}")
    print(f"Net profit: ${result.net_profit:.2f}")
    print(f"Win rate: {result.win_rate*100:.1f}%")

    if result.trades:
        profits = [t['profit'] for t in result.trades]
        losses = [p for p in profits if p < 0]
        if losses:
            print(f"Largest loss: ${min(losses):.2f}")
            print(f"Average loss: ${sum(losses)/len(losses):.2f}")

        # Check for any huge losses (>$500)
        huge_losses = [l for l in losses if l < -500]
        if huge_losses:
            print(f"\n⚠️  Found {len(huge_losses)} huge losses (>$500)")
            for t in result.trades:
                if t['profit'] < -500:
                    print(f"  {t['id']}: profit={t['profit']:.2f}, exit_price={t['exit_price']:.2f}")
        else:
            print("\n✅ No huge losses - max loss limit working!")

if __name__ == "__main__":
    asyncio.run(run())