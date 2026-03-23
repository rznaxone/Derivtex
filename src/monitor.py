"""
Performance monitoring and metrics for Derivtex.
Tracks P&L, trade statistics, and provides dashboard data.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics."""
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
    current_drawdown: float = 0.0
    peak_balance: float = 0.0
    equity_curve: List[float] = field(default_factory=list)

class Monitor:
    """
    Monitors bot performance and provides metrics for dashboard.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.monitoring_config = config.get('monitoring', {})

        # Metrics
        self.metrics = PerformanceMetrics()
        self._trades: List[Dict[str, Any]] = []
        self._balance_history: List[float] = []
        self._timestamps: List[float] = []

        # File paths
        self.logs_dir = Path(__file__).parent.parent / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        self.trades_file = self.logs_dir / 'trades.json'
        self.metrics_file = self.logs_dir / 'metrics.json'

        # Load existing trades if available
        self._load_trades()

        logger.info("Monitor initialized")

    def _load_trades(self) -> None:
        """Load trades from file."""
        if self.trades_file.exists():
            try:
                with open(self.trades_file, 'r') as f:
                    self._trades = json.load(f)
                logger.info(f"Loaded {len(self._trades)} trades from file")
            except Exception as e:
                logger.error(f"Failed to load trades: {e}")

    def _save_trades(self) -> None:
        """Save trades to file."""
        if not self.monitoring_config.get('save_trades', True):
            return

        try:
            with open(self.trades_file, 'w') as f:
                json.dump(self._trades, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save trades: {e}")

    def record_trade(self, trade, profit: float) -> None:
        """
        Record a completed trade.

        Args:
            trade: Trade object
            profit: Profit/loss amount
        """
        trade_data = {
            'id': trade.id,
            'symbol': trade.symbol,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'entry_time': trade.entry_time,
            'exit_price': trade.exit_price,
            'exit_time': trade.exit_time,
            'exit_reason': trade.exit_reason,
            'position_size': trade.position_size,
            'profit': profit,
            'timestamp': trade.exit_time
        }

        self._trades.append(trade_data)
        self._save_trades()

        # Update metrics
        self._update_metrics()

    def update_balance(self, balance: float, timestamp: float) -> None:
        """
        Update balance history.

        Args:
            balance: Current balance
            timestamp: Unix timestamp
        """
        self._balance_history.append(balance)
        self._timestamps.append(timestamp)

        # Keep only recent history (last 10000 points)
        if len(self._balance_history) > 10000:
            self._balance_history = self._balance_history[-10000:]
            self._timestamps = self._timestamps[-10000:]

        # Update peak balance
        if balance > self.metrics.peak_balance:
            self.metrics.peak_balance = balance

        # Calculate current drawdown
        if self.metrics.peak_balance > 0:
            self.metrics.current_drawdown = (self.metrics.peak_balance - balance) / self.metrics.peak_balance

        # Update max drawdown
        if self.metrics.current_drawdown > self.metrics.max_drawdown:
            self.metrics.max_drawdown = self.metrics.current_drawdown

    def _update_metrics(self) -> None:
        """Recalculate all metrics from trade history."""
        if not self._trades:
            return

        # Basic counts
        self.metrics.total_trades = len(self._trades)

        # Separate wins and losses
        profits = [t['profit'] for t in self._trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]

        self.metrics.winning_trades = len(winning_trades)
        self.metrics.losing_trades = len(losing_trades)

        # Totals
        self.metrics.total_profit = sum(winning_trades) if winning_trades else 0.0
        self.metrics.total_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        self.metrics.net_profit = sum(profits)

        # Win rate
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

        # Average win/loss
        self.metrics.avg_win = self.metrics.total_profit / self.metrics.winning_trades if self.metrics.winning_trades > 0 else 0.0
        self.metrics.avg_loss = self.metrics.total_loss / self.metrics.losing_trades if self.metrics.losing_trades > 0 else 0.0

        # Profit factor
        if self.metrics.total_loss > 0:
            self.metrics.profit_factor = self.metrics.total_profit / self.metrics.total_loss
        else:
            self.metrics.profit_factor = float('inf') if self.metrics.total_profit > 0 else 0.0

        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        if len(profits) > 1:
            returns = [p / 100 for p in profits]  # Simplified return calculation
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            if std_return > 0:
                self.metrics.sharpe_ratio = avg_return / std_return
            else:
                self.metrics.sharpe_ratio = 0.0

        # Equity curve
        if self._balance_history:
            self.metrics.equity_curve = self._balance_history.copy()

        # Save metrics
        self._save_metrics()

    def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            metrics_dict = {
                'total_trades': self.metrics.total_trades,
                'winning_trades': self.metrics.winning_trades,
                'losing_trades': self.metrics.losing_trades,
                'total_profit': self.metrics.total_profit,
                'total_loss': self.metrics.total_loss,
                'net_profit': self.metrics.net_profit,
                'win_rate': self.metrics.win_rate,
                'avg_win': self.metrics.avg_win,
                'avg_loss': self.metrics.avg_loss,
                'profit_factor': self.metrics.profit_factor,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'max_drawdown': self.metrics.max_drawdown,
                'current_drawdown': self.metrics.current_drawdown,
                'peak_balance': self.metrics.peak_balance,
                'updated_at': datetime.utcnow().isoformat()
            }

            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for dashboard display.

        Returns:
            Dictionary with metrics and recent trades
        """
        recent_trades = self._trades[-50:] if len(self._trades) > 50 else self._trades

        return {
            'metrics': {
                'total_trades': self.metrics.total_trades,
                'win_rate': round(self.metrics.win_rate * 100, 1),
                'net_profit': round(self.metrics.net_profit, 2),
                'profit_factor': round(self.metrics.profit_factor, 2),
                'sharpe_ratio': round(self.metrics.sharpe_ratio, 2),
                'max_drawdown': round(self.metrics.max_drawdown * 100, 1),
                'avg_win': round(self.metrics.avg_win, 2),
                'avg_loss': round(self.metrics.avg_loss, 2)
            },
            'recent_trades': recent_trades,
            'equity_curve': self.metrics.equity_curve[-1000:] if len(self.metrics.equity_curve) > 1000 else self.metrics.equity_curve,
            'balance_history': self._balance_history[-1000:] if len(self._balance_history) > 1000 else self._balance_history
        }

    def get_stats_summary(self) -> str:
        """Get text summary of statistics."""
        m = self.metrics
        return (
            f"Trades: {m.total_trades} | "
            f"Win Rate: {m.win_rate*100:.1f}% | "
            f"Net P&L: ${m.net_profit:.2f} | "
            f"Profit Factor: {m.profit_factor:.2f} | "
            f"Max DD: {m.max_drawdown*100:.1f}%"
        )