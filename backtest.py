#!/usr/bin/env python3
"""
Derivtex Backtesting Engine
Simulates the trading strategy on historical tick data.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
import argparse
from dataclasses import dataclass, asdict
import logging

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from indicators import Indicators, MarketRegime
from strategy_manager import StrategyManager
from risk_manager import RiskManager
from monitor import Monitor
from logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class BacktestConfig:
    """Backtest configuration."""
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    commission: float = 0.0  # Deriv doesn't charge commission on options
    slippage: float = 0.0  # Slippage in ticks
    use_daily_limits: bool = True
    enable_risk_manager: bool = True

@dataclass
class BacktestResult:
    """Results of a backtest run."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    equity_curve: List[float] = None
    trades: List[Dict[str, Any]] = None
    metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.equity_curve is None:
            self.equity_curve = []
        if self.trades is None:
            self.trades = []
        if self.metrics is None:
            self.metrics = {}

class Backtester:
    """
    Backtesting engine for Derivtex strategy.
    """

    def __init__(self, config: Dict[str, Any], backtest_config: BacktestConfig):
        """
        Initialize backtester.

        Args:
            config: Main bot configuration
            backtest_config: Backtest-specific configuration
        """
        self.config = config
        self.backtest_config = backtest_config

        # Initialize components
        self.indicators = Indicators(config.get('strategy', {}))
        self.risk_manager = RiskManager(config, backtest_config.initial_balance) if backtest_config.enable_risk_manager else None
        # In backtest mode, use strategy manager
        config['backtest_mode'] = True  # Flag for strategies
        self.strategy_manager = StrategyManager(config, self.risk_manager)
        self.monitor = Monitor(config)

        # Log strategy parameters for verification
        logger.info(f"Strategy parameters: EMA({config['strategy']['ema_fast']}/{config['strategy']['ema_slow']}), "
                   f"RSI({config['strategy']['rsi_period']}), "
                   f"ATR TP×{config['strategy']['atr_tp_multiplier']}, SL×{config['strategy']['atr_sl_multiplier']}")

        # State
        self.current_balance = backtest_config.initial_balance
        self.peak_balance = backtest_config.initial_balance
        self.open_trades: List[Dict[str, Any]] = []
        self.closed_trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = [backtest_config.initial_balance]

        # Statistics
        self._trade_counter = 0

    async def run(self, data: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on provided data.

        Args:
            data: DataFrame with columns: timestamp, bid, ask, quote, high, low, volume

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest from {self.backtest_config.start_date} to {self.backtest_config.end_date}")
        logger.info(f"Initial balance: ${self.current_balance:.2f}")
        logger.info(f"Data points: {len(data)}")

        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)

        # Main loop
        for idx, row in data.iterrows():
            tick = row.to_dict()

            # Update indicators
            values = self.indicators.update(tick)

            # Check exits for open trades
            await self._check_exits(tick)

            # Generate signal
            if self.risk_manager:
                can_trade, reason = self.risk_manager.can_trade()
                if not can_trade:
                    # Log why we can't trade occasionally
                    if idx % 1000 == 0:
                        logger.debug(f"Risk manager blocked trading at tick {idx}: {reason}")
                    continue
            else:
                can_trade = True

            if can_trade:
                # Update market regime for strategy manager
                self.strategy_manager.update_market_regime(values)

                # Get signals from active strategies
                signals = self.strategy_manager.generate_signals(tick, values)

                # Combine signals
                if signals:
                    combined_signal = self.strategy_manager.combine_signals(signals)

                    if combined_signal and combined_signal.type not in ["hold"]:
                        logger.info(f"Signal at tick {idx}: {combined_signal.type.value}, "
                                   f"confidence={combined_signal.confidence:.2f}, "
                                   f"reason={combined_signal.reason}")
                        await self._execute_signal(combined_signal, tick)

            # Update equity curve
            self.equity_curve.append(self.current_balance)

            # Log progress
            if idx % 10000 == 0 and idx > 0:
                logger.info(f"Processed {idx}/{len(data)} ticks | "
                           f"Balance: ${self.current_balance:.2f} | "
                           f"Open trades: {len(self.open_trades)}")

        # Close any remaining open trades at last price
        if self.open_trades:
            last_price = data.iloc[-1]['quote']
            for trade in self.open_trades:
                await self._close_trade(trade, last_price, data.iloc[-1]['timestamp'], "end_of_data")

        # Calculate metrics
        result = self._calculate_metrics()
        logger.info(f"Backtest complete. Net profit: ${result.net_profit:.2f}, Win rate: {result.win_rate*100:.1f}%")

        return result

    async def _check_exits(self, tick: Dict[str, Any]) -> None:
        """Check exit conditions for open trades."""
        current_price = tick['quote']
        current_time = tick['timestamp']

        for trade in self.open_trades[:]:  # Copy list to allow modification
            # Check stop loss
            if trade['direction'] == 'long' and current_price <= trade['stop_loss']:
                await self._close_trade(trade, current_price, current_time, "stop_loss")
            elif trade['direction'] == 'short' and current_price >= trade['stop_loss']:
                await self._close_trade(trade, current_price, current_time, "stop_loss")

            # Check take profit
            elif trade['direction'] == 'long' and current_price >= trade['take_profit']:
                await self._close_trade(trade, current_price, current_time, "take_profit")
            elif trade['direction'] == 'short' and current_price <= trade['take_profit']:
                await self._close_trade(trade, current_price, current_time, "take_profit")

            # Check time stop
            elif current_time - trade['entry_time'] > self.config['strategy']['time_stop']:
                await self._close_trade(trade, current_price, current_time, "time_stop")

    async def _execute_signal(self, signal: Signal, tick: Dict[str, Any]) -> None:
        """Execute a trading signal."""
        # Check if we can trade (risk manager)
        if self.risk_manager:
            can_trade, reason = self.risk_manager.can_trade(signal.high_probability)
            if not can_trade:
                logger.debug(f"Signal blocked by risk manager: {reason}")
                return

            # Recalculate position size based on current risk state
            # This ensures max loss limits are applied with current balance
            position_size = self.risk_manager.calculate_position_size(
                signal.entry_price,
                signal.stop_loss,
                signal.high_probability
            )
            logger.info(f"Trade {self._trade_counter}: Using recalculated position_size={position_size:.2f}, "
                       f"entry={signal.entry_price:.2f}, sl={signal.stop_loss:.2f}, "
                       f"balance={self.risk_manager.state.account_balance:.2f}")
        else:
            # Fixed position size if no risk manager
            position_size = 1.0

        # Create trade
        trade = {
            'id': f"bt_{self._trade_counter}",
            'symbol': self.config['trading']['instrument'],
            'direction': signal.type.value.replace('_', ''),
            'entry_price': signal.entry_price,
            'entry_time': signal.timestamp,
            'position_size': position_size,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'status': 'open',
            'signal_confidence': signal.confidence,
            'signal_reason': signal.reason
        }

        self.open_trades.append(trade)
        self._trade_counter += 1

        logger.debug(f"Backtest trade opened: {trade['id']} @ {trade['entry_price']:.5f}")

    async def _close_trade(self, trade: Dict[str, Any], exit_price: float,
                          exit_time: float, reason: str) -> None:
        """Close a trade."""
        # Calculate profit
        if trade['direction'] == 'long':
            profit = (exit_price - trade['entry_price']) * trade['position_size']
        else:
            profit = (trade['entry_price'] - exit_price) * trade['position_size']

        # Apply commission
        if self.backtest_config.commission > 0:
            commission = trade['position_size'] * self.backtest_config.commission
            profit -= commission * 2  # Commission on entry and exit

        # Apply slippage
        if self.backtest_config.slippage > 0:
            slippage_cost = trade['position_size'] * self.backtest_config.slippage
            profit -= slippage_cost

        # Update trade
        trade['exit_price'] = exit_price
        trade['exit_time'] = exit_time
        trade['exit_reason'] = reason
        trade['profit'] = profit
        trade['status'] = 'closed'

        # Update balance
        self.current_balance += profit

        # Update peak balance and drawdown
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        # Record
        self.closed_trades.append(trade)
        if trade in self.open_trades:
            self.open_trades.remove(trade)

        # Record in risk manager if enabled
        if self.risk_manager:
            from risk_manager import TradeResult
            result = TradeResult(
                profit=profit,
                timestamp=datetime.fromtimestamp(exit_time),
                reason=reason
            )
            self.risk_manager.record_trade(result)
            self.risk_manager.update_balance(self.current_balance)

        logger.debug(f"Backtest trade closed: {trade['id']}, profit=${profit:.2f}")

    def _calculate_metrics(self) -> BacktestResult:
        """Calculate performance metrics."""
        if not self.closed_trades:
            logger.warning("No trades closed in backtest")
            return BacktestResult()

        profits = [t['profit'] for t in self.closed_trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]

        total_trades = len(self.closed_trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)

        total_profit = sum(winning_trades) if winning_trades else 0.0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        net_profit = sum(profits)

        win_rate = winning_count / total_trades if total_trades > 0 else 0.0
        avg_win = total_profit / winning_count if winning_count > 0 else 0.0
        avg_loss = total_loss / losing_count if losing_count > 0 else 0.0

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0

        # Sharpe ratio (simplified)
        if len(profits) > 1:
            returns = np.array(profits) / 100  # Simplified
            if returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std()
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

        result = BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=net_profit,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown * 100,
            equity_curve=self.equity_curve,
            trades=[t.copy() for t in self.closed_trades],
            metrics={
                'initial_balance': self.backtest_config.initial_balance,
                'final_balance': self.current_balance,
                'total_return_percent': (self.current_balance / self.backtest_config.initial_balance - 1) * 100,
                'avg_trade_duration': np.mean([t['exit_time'] - t['entry_time'] for t in self.closed_trades]) if self.closed_trades else 0,
                'largest_win': max(profits) if profits else 0,
                'largest_loss': min(profits) if profits else 0,
                'consecutive_wins': self._max_consecutive([p > 0 for p in profits]),
                'consecutive_losses': self._max_consecutive([p < 0 for p in profits])
            }
        )

        return result

    def _max_consecutive(self, conditions: List[bool]) -> int:
        """Calculate maximum consecutive True values."""
        max_consec = 0
        current = 0
        for cond in conditions:
            if cond:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0
        return max_consec

def load_historical_data(filepath: str) -> pd.DataFrame:
    """
    Load historical tick data.

    Expected CSV columns: timestamp, bid, ask, quote, high, low, volume
    """
    df = pd.read_csv(filepath)
    required = ['timestamp', 'bid', 'ask', 'quote']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Ensure numeric types
    for col in ['bid', 'ask', 'quote', 'high', 'low', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values
    df['high'] = df['high'].fillna(df['quote'])
    df['low'] = df['low'].fillna(df['quote'])
    df['volume'] = df['volume'].fillna(1.0)

    return df

def generate_report(result: BacktestResult, output_dir: str = "backtest_results") -> None:
    """
    Generate backtest report with charts and statistics.

    Args:
        result: BacktestResult
        output_dir: Directory to save report files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save JSON report
    report_data = {
        'summary': {
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'net_profit': result.net_profit,
            'profit_factor': result.profit_factor,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'max_drawdown_percent': result.max_drawdown_percent
        },
        'metrics': result.metrics,
        'trades': result.trades
    }

    with open(output_path / 'report.json', 'w') as f:
        json.dump(report_data, f, indent=2)

    # Generate equity curve chart
    if result.equity_curve:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=result.equity_curve,
            mode='lines',
            name='Equity',
            line=dict(color='#1f77b4', width=2)
        ))

        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Trade Number',
            yaxis_title='Balance ($)',
            template='plotly_white'
        )

        fig.write_html(output_path / 'equity_curve.html')

    # Generate trade distribution chart
    if result.trades:
        profits = [t['profit'] for t in result.trades]

        import plotly.express as px
        fig = px.histogram(
            x=profits,
            nbins=50,
            title='Profit Distribution',
            labels={'x': 'Profit ($)', 'y': 'Frequency'}
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.write_html(output_path / 'profit_distribution.html')

    logger.info(f"Backtest report saved to {output_path}")

def print_summary(result: BacktestResult) -> None:
    """Print backtest summary to console."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Trades:     {result.total_trades}")
    print(f"Winning Trades:   {result.winning_trades}")
    print(f"Losing Trades:    {result.losing_trades}")
    print(f"Win Rate:         {result.win_rate*100:.2f}%")
    print(f"Net Profit:       ${result.net_profit:.2f}")
    print(f"Total Profit:     ${result.total_profit:.2f}")
    print(f"Total Loss:       ${result.total_loss:.2f}")
    print(f"Profit Factor:    {result.profit_factor:.2f}")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:     {result.max_drawdown_percent:.2f}%")
    print(f"Avg Win:          ${result.avg_win:.2f}")
    print(f"Avg Loss:         ${result.avg_loss:.2f}")
    print("="*60)

    if result.metrics:
        print("\nADDITIONAL METRICS:")
        print(f"Initial Balance:  ${result.metrics['initial_balance']:.2f}")
        print(f"Final Balance:    ${result.metrics['final_balance']:.2f}")
        print(f"Total Return:     {result.metrics['total_return_percent']:.2f}%")
        print(f"Largest Win:      ${result.metrics['largest_win']:.2f}")
        print(f"Largest Loss:     ${result.metrics['largest_loss']:.2f}")
        print(f"Max Consec. Wins: {result.metrics['consecutive_wins']}")
        print(f"Max Consec. Loss: {result.metrics['consecutive_losses']}")

async def main():
    """Main backtest entry point."""
    parser = argparse.ArgumentParser(description='Derivtex Backtest Engine')
    parser.add_argument('--data', type=str, required=True, help='Path to historical data CSV')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--output', type=str, default='backtest_results', help='Output directory')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--no-risk-manager', action='store_true', help='Disable risk manager')

    args = parser.parse_args()

    # Load config
    from config import load_config
    config = load_config()

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else None
    end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else None

    # Load data
    logger.info(f"Loading data from {args.data}")
    data = load_historical_data(args.data)

    # Filter by date if specified
    if start_date:
        data = data[data['timestamp'] >= start_date.timestamp()]
    if end_date:
        data = data[data['timestamp'] <= end_date.timestamp()]

    if len(data) == 0:
        logger.error("No data available for the specified date range")
        return

    # Create backtest config
    backtest_config = BacktestConfig(
        start_date=start_date or datetime.fromtimestamp(data.iloc[0]['timestamp']),
        end_date=end_date or datetime.fromtimestamp(data.iloc[-1]['timestamp']),
        initial_balance=args.balance,
        enable_risk_manager=not args.no_risk_manager
    )

    # Run backtest
    backtester = Backtester(config, backtest_config)
    result = await backtester.run(data)

    # Print and save results
    print_summary(result)
    generate_report(result, args.output)

if __name__ == "__main__":
    asyncio.run(main())