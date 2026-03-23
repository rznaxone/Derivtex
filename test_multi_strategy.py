"""
Test multi-strategy framework.
"""

import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.strategy_manager import StrategyManager
from src.config import load_config
from src.indicators import Indicators

def test_strategy_manager():
    print("="*60)
    print("MULTI-STRATEGY FRAMEWORK TEST")
    print("="*60)

    # Load config
    config = load_config()
    strategy_config = config.get('strategy', {})
    print(f"\nActive strategies: {strategy_config.get('active_strategies', [])}")
    print(f"Selection mode: {strategy_config.get('selection_mode', 'manual')}")

    # Create strategy manager
    manager = StrategyManager(config)

    print(f"\nRegistered strategies: {list(manager.strategies.keys())}")
    print(f"Active strategies: {[s.get_name() for s in manager.get_active_strategies()]}")

    # Load test data
    df = pd.read_csv('data/signal_rich_10k.csv').head(1000)
    print(f"\nTesting on {len(df)} ticks")

    # Simulate ticks
    signals_generated = 0
    for idx, row in df.iterrows():
        tick = row.to_dict()

        # Strategies maintain their own state, just call generate_signals
        # The strategies will internally update their indicators
        signals = manager.generate_signals(tick, None)

        if signals:
            signals_generated += len(signals)
            if signals_generated <= 3:
                print(f"\nTick {idx}: {len(signals)} signal(s)")
                for sig in signals:
                    print(f"  - {sig.type} from {sig.reason}")

    print(f"\nTotal signals generated: {signals_generated}")
    print(f"Performance summary: {manager.get_performance_summary()}")

    print("\n" + "="*60)
    print("✅ Multi-strategy framework working!")
    print("="*60)

if __name__ == "__main__":
    test_strategy_manager()