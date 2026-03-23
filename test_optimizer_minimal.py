"""
Minimal optimizer test with tiny dataset and 2 parameter combinations.
"""

import sys
from pathlib import Path
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from optimizer import StrategyOptimizer, OptimizationConfig, ParameterRange
from config import load_config

async def minimal_optimize():
    """Test optimizer with minimal data and parameters."""
    print("="*60)
    print("MINIMAL OPTIMIZER TEST")
    print("="*60)

    # Generate tiny dataset (1000 ticks) with clear trend
    print("\nGenerating test data...")
    timestamps = np.arange(1000) + datetime(2024, 1, 1).timestamp()
    prices = np.linspace(1000, 1020, 1000) + np.random.randn(1000) * 0.5

    df = pd.DataFrame({
        'timestamp': timestamps,
        'bid': prices * 0.9995,
        'ask': prices * 1.0005,
        'quote': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'volume': 1.0
    })
    print(f"  Generated {len(df)} ticks with uptrend")

    config = load_config()

    # Just 2 parameter combinations
    param_ranges = [
        ParameterRange("ema_fast", [20]),
        ParameterRange("ema_slow", [50]),
        ParameterRange("rsi_period", [14]),
        ParameterRange("atr_tp_multiplier", [3.0]),
        ParameterRange("atr_sl_multiplier", [1.5]),
        ParameterRange("adx_trending", [25]),
        ParameterRange("crossover_lookback", [3])
    ]

    opt_config = OptimizationConfig(
        param_ranges=param_ranges,
        optimization_metric="sharpe_ratio",
        min_trades=0,  # Allow 0 trades for testing
        walk_forward=False,
        monte_carlo_sims=10,
        output_dir="optimization_minimal"
    )

    optimizer = StrategyOptimizer(config, opt_config)

    print("\nRunning optimization (1 combination)...")
    result = await optimizer.optimize(df)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Total combinations tested: {len(result.all_results)}")
    if result.all_results:
        best = result.all_results[0]
        print(f"Best metrics: {best['metrics']}")
        print(f"Trades generated: {best['metrics']['total_trades']}")
    print(f"\nResults saved to: {opt_config.output_dir}")

    # Check if optimizer infrastructure works
    if result.all_results:
        print("\n✅ Optimizer is working correctly!")
    else:
        print("\n⚠️  No valid results (likely no trades generated)")

if __name__ == "__main__":
    asyncio.run(minimal_optimize())