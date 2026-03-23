"""
Quick optimization test with reduced parameter set.
"""

import sys
from pathlib import Path
import asyncio
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from optimizer import StrategyOptimizer, OptimizationConfig, ParameterRange
from config import load_config

async def quick_optimize():
    """Run optimization with smaller parameter grid for testing."""
    print("="*60)
    print("QUICK OPTIMIZATION TEST")
    print("="*60)

    # Load data - use signal-rich dataset (first 2000 ticks for speed)
    print("\nLoading data...")
    df = pd.read_csv('data/signal_rich_10k.csv')
    df = df.head(2000)  # Use first 2000 ticks
    print(f"  Loaded {len(df)} ticks")

    # Load config
    config = load_config()

    # Focus on ATR multipliers to fix risk-reward
    param_ranges = [
        ParameterRange("ema_fast", [20]),  # Fixed for speed
        ParameterRange("ema_slow", [50]),
        ParameterRange("rsi_period", [14]),
        ParameterRange("atr_tp_multiplier", [3.0, 4.0, 5.0]),  # Test higher TP
        ParameterRange("atr_sl_multiplier", [1.0, 1.25, 1.5]),  # Test tighter SL
        ParameterRange("adx_trending", [25]),
        ParameterRange("crossover_lookback", [3])
    ]

    opt_config = OptimizationConfig(
        param_ranges=param_ranges,
        optimization_metric="sharpe_ratio",
        min_trades=10,
        walk_forward=False,  # Skip for speed
        monte_carlo_sims=100,  # Reduced
        output_dir="optimization_quick"
    )

    optimizer = StrategyOptimizer(config, opt_config)

    print("\nStarting optimization...")
    result = await optimizer.optimize(df)

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest parameters:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest metrics:")
    for k, v in result.best_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\nResults saved to: {opt_config.output_dir}")

if __name__ == "__main__":
    asyncio.run(quick_optimize())