"""
Strategy Optimization & Validation Module for Derivtex
Performs parameter optimization, walk-forward analysis, and Monte Carlo testing.
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import json

from backtest import Backtester, BacktestConfig
from config import load_config

logger = logging.getLogger(__name__)

@dataclass
class ParameterRange:
    """Defines a parameter search range."""
    name: str
    values: List[Any]
    description: str = ""

@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    param_ranges: List[ParameterRange] = field(default_factory=list)
    optimization_metric: str = "sharpe_ratio"  # sharpe_ratio, profit_factor, win_rate, net_profit
    min_trades: int = 20  # Minimum trades for valid result
    walk_forward: bool = True
    walk_forward_window: int = 30  # days
    walk_forward_step: int = 7  # days
    monte_carlo_sims: int = 1000
    monte_carlo_sample: float = 0.8  # Sample 80% of trades each simulation
    output_dir: str = "optimization_results"

@dataclass
class OptimizationResult:
    """Results from optimization."""
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    all_results: List[Dict[str, Any]]
    walk_forward_results: Optional[List[Dict[str, Any]]] = None
    monte_carlo_results: Optional[Dict[str, List[float]]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class StrategyOptimizer:
    """
    Optimizes strategy parameters using historical data.
    Supports grid search, walk-forward analysis, and Monte Carlo simulation.
    """

    def __init__(self, config: Dict[str, Any], opt_config: OptimizationConfig):
        """
        Initialize optimizer.

        Args:
            config: Base bot configuration
            opt_config: Optimization configuration
        """
        self.base_config = config
        self.opt_config = opt_config
        self.output_path = Path(opt_config.output_dir)
        self.output_path.mkdir(exist_ok=True)

    async def optimize(self, data: pd.DataFrame) -> OptimizationResult:
        """
        Run full optimization pipeline.

        Args:
            data: Historical tick data

        Returns:
            OptimizationResult with all findings
        """
        logger.info("Starting strategy optimization...")
        logger.info(f"Parameter ranges: {self.opt_config.param_ranges}")

        # 1. Grid search
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Grid Search")
        logger.info("="*60)
        all_results = await self._grid_search(data)

        if not all_results:
            raise ValueError("No valid parameter combinations found")

        # Sort by optimization metric
        metric = self.opt_config.optimization_metric
        all_results.sort(key=lambda x: x['metrics'][metric], reverse=True)

        best = all_results[0]
        logger.info(f"\nBest parameters ({metric}):")
        for k, v in best['params'].items():
            logger.info(f"  {k}: {v}")
        logger.info(f"Metrics: {best['metrics']}")

        # 2. Walk-forward analysis (if enabled)
        wf_results = None
        if self.opt_config.walk_forward:
            logger.info("\n" + "="*60)
            logger.info("STEP 2: Walk-Forward Analysis")
            logger.info("="*60)
            wf_results = await self._walk_forward_analysis(data, best['params'])

        # 3. Monte Carlo simulation
        mc_results = None
        if self.opt_config.monte_carlo_sims > 0:
            logger.info("\n" + "="*60)
            logger.info("STEP 3: Monte Carlo Simulation")
            logger.info("="*60)
            mc_results = await self._monte_carlo_simulation(data, best['params'])

        # 4. Save results
        result = OptimizationResult(
            best_params=best['params'],
            best_metrics=best['metrics'],
            all_results=all_results,
            walk_forward_results=wf_results,
            monte_carlo_results=mc_results
        )

        self._save_results(result)
        self._generate_report(result)

        return result

    async def _grid_search(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Perform grid search over parameter combinations.

        Returns:
            List of results dicts with params and metrics
        """
        # Generate all parameter combinations
        param_names = [p.name for p in self.opt_config.param_ranges]
        param_values = [p.values for p in self.opt_config.param_ranges]

        combinations = list(itertools.product(*param_values))
        logger.info(f"Testing {len(combinations)} parameter combinations")

        results = []

        for i, combo in enumerate(combinations, 1):
            params = dict(zip(param_names, combo))

            # Create config with these parameters
            test_config = self._apply_params(params)

            # Run backtest
            backtest_config = BacktestConfig(
                start_date=datetime.fromtimestamp(data['timestamp'].iloc[0]),
                end_date=datetime.fromtimestamp(data['timestamp'].iloc[-1]),
                initial_balance=10000.0,
                enable_risk_manager=True
            )

            backtester = Backtester(test_config, backtest_config)
            result = await backtester.run(data)

            # Check minimum trades
            if result.total_trades < self.opt_config.min_trades:
                logger.debug(f"  Combo {i}: Skipped (only {result.total_trades} trades)")
                continue

            # Collect metrics
            metrics = {
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'net_profit': result.net_profit,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'avg_win': result.avg_win,
                'avg_loss': result.avg_loss,
                'total_return_pct': result.metrics.get('total_return_percent', 0)
            }

            results.append({
                'params': params,
                'metrics': metrics,
                'result': result
            })

            logger.info(f"  Combo {i}/{len(combinations)}: {params} -> "
                       f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                       f"Win%={metrics['win_rate']*100:.1f}%, "
                       f"Trades={metrics['total_trades']}")

        logger.info(f"\nCompleted: {len(results)} valid results")
        return results

    def _apply_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply parameter overrides to base config.

        Args:
            params: Parameter dictionary

        Returns:
            Modified config
        """
        config = self.base_config.copy()

        # Deep copy nested dicts
        config['strategy'] = config['strategy'].copy()
        config['risk'] = config['risk'].copy()

        # Apply parameters
        for key, value in params.items():
            if key in config['strategy']:
                config['strategy'][key] = value
            elif key in config['risk']:
                config['risk'][key] = value

        return config

    async def _walk_forward_analysis(self, data: pd.DataFrame, best_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform walk-forward analysis to test robustness.

        Args:
            data: Full dataset
            best_params: Best parameters from grid search

        Returns:
            List of walk-forward period results
        """
        window_days = self.opt_config.walk_forward_window
        step_days = self.opt_config.walk_forward_step

        start_date = datetime.fromtimestamp(data['timestamp'].min())
        end_date = datetime.fromtimestamp(data['timestamp'].max())

        wf_results = []
        current_start = start_date

        while current_start < end_date:
            current_end = current_start + pd.Timedelta(days=window_days)
            if current_end > end_date:
                break

            # Filter data for this window
            mask = (data['timestamp'] >= current_start.timestamp()) & \
                   (data['timestamp'] < current_end.timestamp())
            window_data = data[mask]

            if len(window_data) < 1000:  # Skip if too little data
                current_start += pd.Timedelta(days=step_days)
                continue

            # Run backtest with best params
            test_config = self._apply_params(best_params)
            backtest_config = BacktestConfig(
                start_date=current_start,
                end_date=current_end,
                initial_balance=10000.0,
                enable_risk_manager=True
            )

            backtester = Backtester(test_config, backtest_config)
            result = await backtester.run(window_data)

            wf_results.append({
                'period_start': current_start.isoformat(),
                'period_end': current_end.isoformat(),
                'metrics': {
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'net_profit': result.net_profit,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown
                }
            })

            logger.info(f"  WF Period {current_start.date()} - {current_end.date()}: "
                       f"Sharpe={result.sharpe_ratio:.2f}, Trades={result.total_trades}")

            current_start += pd.Timedelta(days=step_days)

        return wf_results

    async def _monte_carlo_simulation(self, data: pd.DataFrame, best_params: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Run Monte Carlo simulation to test robustness.

        Args:
            data: Full dataset
            best_params: Best parameters

        Returns:
            Dictionary with distribution of metrics
        """
        # First, get baseline result with all trades
        test_config = self._apply_params(best_params)
        backtest_config = BacktestConfig(
            start_date=datetime.fromtimestamp(data['timestamp'].iloc[0]),
            end_date=datetime.fromtimestamp(data['timestamp'].iloc[-1]),
            initial_balance=10000.0,
            enable_risk_manager=True
        )

        baseline_backtester = Backtester(test_config, backtest_config)
        baseline_result = await baseline_backtester.run(data)

        if not baseline_result.trades:
            logger.warning("No trades for Monte Carlo simulation")
            return {}

        # Extract trade profits
        profits = [t['profit'] for t in baseline_result.trades]
        n_trades = len(profits)

        # Run simulations
        n_sims = self.opt_config.monte_carlo_sims
        sample_size = int(n_trades * self.opt_config.monte_carlo_sample)

        sharpe_ratios = []
        profit_factors = []
        max_drawdowns = []
        net_profits = []

        logger.info(f"Running {n_sims} Monte Carlo simulations...")

        for i in range(n_sims):
            # Sample trades with replacement
            sampled_profits = np.random.choice(profits, size=sample_size, replace=True)

            # Calculate metrics
            wins = [p for p in sampled_profits if p > 0]
            losses = [p for p in sampled_profits if p < 0]

            total_profit = sum(wins) if wins else 0
            total_loss = abs(sum(losses)) if losses else 0
            net = total_profit - total_loss

            # Sharpe (simplified)
            returns = np.array(sampled_profits) / 10000  # Normalize by initial balance
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std()
            else:
                sharpe = 0

            # Profit factor
            pf = total_profit / total_loss if total_loss > 0 else float('inf')

            # Max drawdown (simplified)
            equity = 10000 + np.cumsum(sampled_profits)
            running_max = np.maximum.accumulate(equity)
            dd = (equity - running_max) / running_max
            max_dd = dd.min() if len(dd) > 0 else 0

            sharpe_ratios.append(sharpe)
            profit_factors.append(pf)
            max_drawdowns.append(max_dd)
            net_profits.append(net)

            if (i+1) % 100 == 0:
                logger.debug(f"  Completed {i+1}/{n_sims} simulations")

        # Calculate statistics
        mc_results = {
            'sharpe_ratio': {
                'mean': float(np.mean(sharpe_ratios)),
                'std': float(np.std(sharpe_ratios)),
                'percentile_5': float(np.percentile(sharpe_ratios, 5)),
                'percentile_95': float(np.percentile(sharpe_ratios, 95)),
                'distribution': sharpe_ratios[:100]  # Store sample
            },
            'profit_factor': {
                'mean': float(np.mean(profit_factors)),
                'std': float(np.std(profit_factors)),
                'percentile_5': float(np.percentile(profit_factors, 5)),
                'percentile_95': float(np.percentile(profit_factors, 95))
            },
            'max_drawdown': {
                'mean': float(np.mean(max_drawdowns)),
                'std': float(np.std(max_drawdowns)),
                'percentile_95': float(np.percentile(max_drawdowns, 95)),  # Worst case
                'worst_case': float(np.max(max_drawdowns))
            },
            'net_profit': {
                'mean': float(np.mean(net_profits)),
                'std': float(np.std(net_profits)),
                'percentile_5': float(np.percentile(net_profits, 5))
            },
            'n_simulations': n_sims,
            'sample_size': sample_size,
            'baseline_sharpe': baseline_result.sharpe_ratio,
            'baseline_net_profit': baseline_result.net_profit
        }

        logger.info(f"Monte Carlo Results:")
        logger.info(f"  Sharpe: {mc_results['sharpe_ratio']['mean']:.2f} ± {mc_results['sharpe_ratio']['std']:.2f}")
        logger.info(f"  5th percentile Sharpe: {mc_results['sharpe_ratio']['percentile_5']:.2f}")
        logger.info(f"  95th percentile Max DD: {mc_results['max_drawdown']['percentile_95']*100:.1f}%")

        return mc_results

    def _save_results(self, result: OptimizationResult) -> None:
        """Save optimization results to disk."""
        # Save JSON summary
        summary = {
            'best_params': result.best_params,
            'best_metrics': result.best_metrics,
            'timestamp': result.timestamp,
            'config': {
                'optimization_metric': self.opt_config.optimization_metric,
                'min_trades': self.opt_config.min_trades,
                'walk_forward': self.opt_config.walk_forward,
                'monte_carlo_sims': self.opt_config.monte_carlo_sims
            }
        }

        with open(self.output_path / 'optimization_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save all grid search results
        all_results_df = pd.DataFrame([
            {**r['params'], **r['metrics']} for r in result.all_results
        ])
        all_results_df.to_csv(self.output_path / 'grid_search_results.csv', index=False)

        # Save walk-forward results
        if result.walk_forward_results:
            wf_df = pd.DataFrame(result.walk_forward_results)
            wf_df.to_csv(self.output_path / 'walk_forward_results.csv', index=False)

        # Save Monte Carlo results
        if result.monte_carlo_results:
            with open(self.output_path / 'monte_carlo_results.json', 'w') as f:
                json.dump(result.monte_carlo_results, f, indent=2)

        logger.info(f"Results saved to {self.output_path}")

    def _generate_report(self, result: OptimizationResult) -> None:
        """Generate human-readable report."""
        report_path = self.output_path / 'optimization_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DERIVTEX STRATEGY OPTIMIZATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Generated: {result.timestamp}\n\n")

            f.write("BEST PARAMETERS\n")
            f.write("-"*80 + "\n")
            for k, v in result.best_params.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

            f.write("BEST METRICS\n")
            f.write("-"*80 + "\n")
            for k, v in result.best_metrics.items():
                if isinstance(v, float):
                    f.write(f"  {k}: {v:.4f}\n")
                else:
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

            f.write("GRID SEARCH SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total combinations tested: {len(result.all_results)}\n")
            f.write(f"Valid results: {len(result.all_results)}\n")

            # Top 10 results
            f.write("\nTOP 10 PARAMETER SETS:\n")
            for i, r in enumerate(result.all_results[:10], 1):
                f.write(f"\n{i}. {r['params']}\n")
                f.write(f"   Sharpe: {r['metrics']['sharpe_ratio']:.2f}, ")
                f.write(f"Win%: {r['metrics']['win_rate']*100:.1f}%, ")
                f.write(f"Trades: {r['metrics']['total_trades']}\n")

            # Walk-forward summary
            if result.walk_forward_results:
                f.write("\n" + "="*80 + "\n")
                f.write("WALK-FORWARD ANALYSIS\n")
                f.write("-"*80 + "\n")
                wf_sharpes = [r['metrics']['sharpe_ratio'] for r in result.walk_forward_results]
                f.write(f"Periods: {len(result.walk_forward_results)}\n")
                f.write(f"Average Sharpe: {np.mean(wf_sharpes):.2f}\n")
                f.write(f"Min Sharpe: {np.min(wf_sharpes):.2f}\n")
                f.write(f"Max Sharpe: {np.max(wf_sharpes):.2f}\n")
                f.write("\nPeriod breakdown:\n")
                for r in result.walk_forward_results:
                    f.write(f"  {r['period_start'][:10]} - {r['period_end'][:10]}: ")
                    f.write(f"Sharpe={r['metrics']['sharpe_ratio']:.2f}, ")
                    f.write(f"Trades={r['metrics']['total_trades']}\n")

            # Monte Carlo summary
            if result.monte_carlo_results:
                mc = result.monte_carlo_results
                f.write("\n" + "="*80 + "\n")
                f.write("MONTE CARLO SIMULATION\n")
                f.write("-"*80 + "\n")
                f.write(f"Simulations: {mc['n_simulations']}\n")
                f.write(f"Sample size: {mc['sample_size']} trades per sim\n\n")
                f.write("Sharpe Ratio Distribution:\n")
                f.write(f"  Mean: {mc['sharpe_ratio']['mean']:.2f}\n")
                f.write(f"  Std: {mc['sharpe_ratio']['std']:.2f}\n")
                f.write(f"  5th percentile: {mc['sharpe_ratio']['percentile_5']:.2f}\n")
                f.write(f"  95th percentile: {mc['sharpe_ratio']['percentile_95']:.2f}\n\n")
                f.write("Max Drawdown (95th percentile - worst case):\n")
                f.write(f"  {mc['max_drawdown']['percentile_95']*100:.1f}%\n\n")
                f.write("Net Profit Distribution:\n")
                f.write(f"  Mean: ${mc['net_profit']['mean']:.2f}\n")
                f.write(f"  Std: ${mc['net_profit']['std']:.2f}\n")
                f.write(f"  5th percentile: ${mc['net_profit']['percentile_5']:.2f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-"*80 + "\n")

            # Auto-generate recommendations
            best_sharpe = result.best_metrics['sharpe_ratio']
            best_dd = result.best_metrics['max_drawdown']
            best_wr = result.best_metrics['win_rate']

            if best_sharpe >= 1.5:
                f.write("✓ Excellent Sharpe ratio (>1.5). Strategy is robust.\n")
            elif best_sharpe >= 1.0:
                f.write("✓ Good Sharpe ratio (1.0-1.5). Strategy is acceptable.\n")
            else:
                f.write("⚠ Low Sharpe ratio (<1.0). Consider parameter tuning or strategy review.\n")

            if best_dd < 0.10:
                f.write("✓ Low max drawdown (<10%). Good risk management.\n")
            elif best_dd < 0.20:
                f.write("✓ Moderate drawdown (10-20%). Acceptable.\n")
            else:
                f.write("⚠ High drawdown (>20%). May need tighter risk controls.\n")

            if best_wr >= 0.55:
                f.write("✓ High win rate (≥55%). Good signal quality.\n")
            elif best_wr >= 0.45:
                f.write("✓ Moderate win rate (45-55%). Acceptable.\n")
            else:
                f.write("⚠ Low win rate (<45%). Strategy may need refinement.\n")

            if result.monte_carlo_results:
                mc_sharpe_p5 = result.monte_carlo_results['sharpe_ratio']['percentile_5']
                if mc_sharpe_p5 > 0:
                    f.write("✓ Monte Carlo shows positive Sharpe in worst case (5th percentile).\n")
                else:
                    f.write("⚠ Monte Carlo shows negative Sharpe in worst case. Strategy may be fragile.\n")

            f.write("\n")

        logger.info(f"Report saved to {report_path}")

def create_default_parameter_ranges() -> List[ParameterRange]:
    """
    Create default parameter ranges for optimization.
    These are sensible starting points for R_30 volatility 30.
    """
    return [
        ParameterRange(
            name="ema_fast",
            values=[10, 15, 20, 25, 30],
            description="Fast EMA period"
        ),
        ParameterRange(
            name="ema_slow",
            values=[40, 50, 60, 70, 80],
            description="Slow EMA period"
        ),
        ParameterRange(
            name="rsi_period",
            values=[12, 14, 16],
            description="RSI calculation period"
        ),
        ParameterRange(
            name="atr_tp_multiplier",
            values=[2.5, 3.0, 3.5, 4.0],
            description="Take profit ATR multiplier"
        ),
        ParameterRange(
            name="atr_sl_multiplier",
            values=[1.0, 1.25, 1.5, 1.75],
            description="Stop loss ATR multiplier"
        ),
        ParameterRange(
            name="adx_trending",
            values=[20, 22, 25, 28],
            description="ADX threshold for trending regime"
        ),
        ParameterRange(
            name="crossover_lookback",
            values=[2, 3, 4, 5],
            description="Ticks to look back for crossover confirmation"
        )
    ]

async def run_optimization(data_path: str, output_dir: str = "optimization_results") -> OptimizationResult:
    """
    Convenience function to run optimization.

    Args:
        data_path: Path to historical data CSV
        output_dir: Output directory for results

    Returns:
        OptimizationResult
    """
    # Load data
    df = pd.read_csv(data_path)

    # Load config
    config = load_config()

    # Create parameter ranges
    param_ranges = create_default_parameter_ranges()

    # Create optimization config
    opt_config = OptimizationConfig(
        param_ranges=param_ranges,
        optimization_metric="sharpe_ratio",
        min_trades=20,
        walk_forward=True,
        walk_forward_window=7,  # 7 days
        walk_forward_step=1,    # step by 1 day
        monte_carlo_sims=500,   # Reduced for speed
        output_dir=output_dir
    )

    # Run optimization
    optimizer = StrategyOptimizer(config, opt_config)
    result = await optimizer.optimize(df)

    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Derivtex Strategy Optimizer')
    parser.add_argument('--data', type=str, required=True, help='Historical data CSV')
    parser.add_argument('--output', type=str, default='optimization_results', help='Output directory')
    parser.add_argument('--metric', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'profit_factor', 'win_rate', 'net_profit'],
                       help='Optimization metric')

    args = parser.parse_args()

    import asyncio
    asyncio.run(run_optimization(args.data, args.output))