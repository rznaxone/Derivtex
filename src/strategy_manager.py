"""
Strategy Manager for Derivtex.
Manages multiple strategies, selects active ones, and combines signals.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum

from strategies.base import BaseStrategy, Signal
from strategies.ema_rsi import EMARSIStrategy
from strategies.macd import MACDStrategy
from strategies.bollinger_bands import BollingerBandsStrategy
from indicators import Indicators, MarketRegime

logger = logging.getLogger(__name__)

class SelectionMode(Enum):
    """Strategy selection mode."""
    MANUAL = "manual"  # Use only manually selected strategy
    AUTO = "auto"      # Auto-select based on market regime
    ENSEMBLE = "ensemble"  # Combine all active strategies

@dataclass
class StrategyPerformance:
    """Track performance metrics for a strategy."""
    name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_factor(self) -> float:
        if self.total_loss == 0:
            return float('inf') if self.total_profit > 0 else 0.0
        return self.total_profit / self.total_loss

    @property
    def net_profit(self) -> float:
        return self.total_profit - self.total_loss

    @property
    def avg_win(self) -> float:
        return self.total_profit / self.winning_trades if self.winning_trades > 0 else 0.0

    @property
    def avg_loss(self) -> float:
        return self.total_loss / self.losing_trades if self.losing_trades > 0 else 0.0

    @property
    def sharpe_estimate(self) -> float:
        """Rough Sharpe estimate based on recent performance."""
        if self.total_trades < 10:
            return 0.0
        # Simplified: use profit factor and win rate
        if self.profit_factor > 1 and self.win_rate > 0.5:
            return 1.0 + (self.profit_factor - 1) * 0.5
        return 0.0

class StrategyManager:
    """
    Manages multiple trading strategies.
    Handles strategy selection, performance tracking, and signal combination.
    """

    # Registry of available strategies
    STRATEGY_REGISTRY = {
        'ema_rsi': EMARSIStrategy,
        'macd_crossover': MACDStrategy,
        'bollinger_squeeze': BollingerBandsStrategy,
    }

    def __init__(self, config: Dict[str, Any], risk_manager=None):
        """
        Initialize strategy manager.

        Args:
            config: Full bot configuration
            risk_manager: Risk manager instance
        """
        self.config = config
        self.risk_manager = risk_manager

        # Strategy config
        self.strategy_config = config.get('strategy', {})
        self.selection_mode = SelectionMode(
            self.strategy_config.get('selection_mode', 'manual')
        )
        self.active_strategy_names = self.strategy_config.get('active_strategies', ['ema_rsi'])
        self.ensemble_weights = self.strategy_config.get('ensemble_weights', {})

        # Market regime auto-selection mapping
        self.regime_mapping = self.strategy_config.get('auto_regime_mapping', {
            'trending': ['ema_rsi', 'macd_crossover'],
            'ranging': ['bollinger_squeeze'],
            'volatile': []  # No strategies in extreme volatility
        })

        # Initialize strategies
        self.strategies: Dict[str, BaseStrategy] = {}
        self.performance: Dict[str, StrategyPerformance] = {}

        for name in self.active_strategy_names:
            self._register_strategy(name)

        # Market regime detection
        self._current_regime = MarketRegime.TRANSITION
        self._regime_lookback = self.strategy_config.get('regime_lookback', 50)

        logger.info(f"StrategyManager initialized: mode={self.selection_mode.value}, "
                   f"active={self.active_strategy_names}")

    def _register_strategy(self, name: str) -> None:
        """Create and register a strategy instance."""
        if name not in self.STRATEGY_REGISTRY:
            logger.warning(f"Unknown strategy: {name}, skipping")
            return

        strategy_class = self.STRATEGY_REGISTRY[name]
        strategy = strategy_class(self.config, self.risk_manager)
        self.strategies[name] = strategy
        self.performance[name] = StrategyPerformance(name=name)

        logger.info(f"Registered strategy: {name}")

    def update_market_regime(self, indicators: Any) -> None:
        """
        Update current market regime detection.

        Args:
            indicators: IndicatorValues object with ADX
        """
        if hasattr(indicators, 'regime'):
            self._current_regime = indicators.regime
        else:
            # Fallback: calculate from ADX
            adx = getattr(indicators, 'adx', 0)
            if adx > 25:
                self._current_regime = MarketRegime.TRENDING
            elif adx < 20:
                self._current_regime = MarketRegime.RANGING
            else:
                self._current_regime = MarketRegime.TRANSITION

    def get_active_strategies(self) -> List[BaseStrategy]:
        """
        Get list of active strategies based on selection mode and market regime.

        Returns:
            List of strategy instances to use for signal generation
        """
        if self.selection_mode == SelectionMode.MANUAL:
            return list(self.strategies.values())

        elif self.selection_mode == SelectionMode.AUTO:
            # Select strategies based on market regime
            regime_key = self._current_regime.value
            regime_strategies = self.regime_mapping.get(regime_key, [])

            # Filter to only registered strategies
            active = [
                self.strategies[name] for name in regime_strategies
                if name in self.strategies
            ]

            if not active:
                logger.warning(f"No strategies for regime {regime_key}, using all")
                return list(self.strategies.values())

            return active

        elif self.selection_mode == SelectionMode.ENSEMBLE:
            return list(self.strategies.values())

        return list(self.strategies.values())

    def generate_signals(self, tick: Dict[str, Any], indicators: Any) -> List[Signal]:
        """
        Generate signals from all active strategies.

        Args:
            tick: Current tick data
            indicators: Current indicator values

        Returns:
            List of signals from active strategies
        """
        active_strategies = self.get_active_strategies()
        signals = []

        for strategy in active_strategies:
            try:
                signal = strategy.update(tick, indicators)
                if signal and signal.type not in ["hold", "exit_long", "exit_short"]:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error in strategy {strategy.get_name()}: {e}")

        return signals

    def combine_signals(self, signals: List[Signal]) -> Optional[Signal]:
        """
        Combine multiple signals into a single trading decision.

        Args:
            signals: List of signals from different strategies

        Returns:
            Combined signal or None if no consensus
        """
        if not signals:
            return None

        if len(signals) == 1:
            return signals[0]

        if self.selection_mode == SelectionMode.ENSEMBLE:
            return self._ensemble_combine(signals)
        else:
            # AUTO mode: take first signal (strategies already filtered by regime)
            return signals[0] if signals else None

    def _ensemble_combine(self, signals: List[Signal]) -> Optional[Signal]:
        """
        Combine signals using weighted voting.

        Strategy:
        1. Count votes for each direction (long/short)
        2. Weight by strategy performance (recent Sharpe)
        3. Require minimum threshold (e.g., 60% of weight)
        4. Return combined signal with average confidence
        """
        if not signals:
            return None

        # Get weights for each strategy
        weights = {}
        total_weight = 0.0

        for signal in signals:
            strategy_name = self._get_strategy_name_from_signal(signal)
            if strategy_name and strategy_name in self.ensemble_weights:
                weight = self.ensemble_weights[strategy_name]
            else:
                # Default weight based on performance
                perf = self.performance.get(strategy_name)
                if perf and perf.total_trades > 10:
                    weight = max(0.1, perf.sharpe_estimate)
                else:
                    weight = 1.0  # Equal weight for new strategies

            weights[signal.type] = weights.get(signal.type, 0) + weight
            total_weight += weight

        if total_weight == 0:
            return None

        # Find direction with highest weight
        long_weight = weights.get("long", 0)
        short_weight = weights.get("short", 0)

        # Require at least 50% consensus
        if long_weight > short_weight and long_weight / total_weight >= 0.5:
            # Combine long signals
            long_signals = [s for s in signals if s.type == "long"]
            return self._merge_signals(long_signals, "long")

        elif short_weight > long_weight and short_weight / total_weight >= 0.5:
            short_signals = [s for s in signals if s.type == "short"]
            return self._merge_signals(short_signals, "short")

        return None

    def _merge_signals(self, signals: List[Signal], direction: str) -> Signal:
        """
        Merge multiple signals of same direction.

        Uses weighted average of entry prices, stop losses, take profits.
        Confidence is average weighted by strategy performance.
        """
        if not signals:
            raise ValueError("No signals to merge")

        # Get weights
        total_weight = 0.0
        weighted_entry = 0.0
        weighted_sl = 0.0
        weighted_tp = 0.0
        weighted_confidence = 0.0

        for signal in signals:
            strategy_name = self._get_strategy_name_from_signal(signal)
            weight = self.ensemble_weights.get(strategy_name, 1.0)

            weighted_entry += signal.entry_price * weight
            weighted_sl += signal.stop_loss * weight
            weighted_tp += signal.take_profit * weight
            weighted_confidence += signal.confidence * weight
            total_weight += weight

        # Calculate averages
        avg_entry = weighted_entry / total_weight
        avg_sl = weighted_sl / total_weight
        avg_tp = weighted_tp / total_weight
        avg_confidence = weighted_confidence / total_weight

        # Use max position size (conservative)
        position_size = min(s.position_size for s in signals if s.position_size)

        # Combine reasons
        reasons = [s.reason for s in signals]
        combined_reason = f"Ensemble: {'; '.join(reasons)}"

        return Signal(
            type=direction,
            confidence=avg_confidence,
            reason=combined_reason,
            entry_price=avg_entry,
            stop_loss=avg_sl,
            take_profit=avg_tp,
            position_size=position_size,
            high_probability=avg_confidence > 0.7,
            timestamp=signals[0].timestamp,
            metadata={
                'ensemble_size': len(signals),
                'strategies': [self._get_strategy_name_from_signal(s) for s in signals]
            }
        )

    def _get_strategy_name_from_signal(self, signal: Signal) -> Optional[str]:
        """Extract strategy name from signal metadata."""
        if signal.metadata and 'strategy' in signal.metadata:
            return signal.metadata['strategy']

        # Try to infer from reason
        reason = signal.reason.lower()
        for name in self.strategies.keys():
            if name.replace('_', ' ') in reason or name in reason:
                return name

        return None

    def record_trade_result(self, strategy_name: str, profit: float) -> None:
        """
        Record trade result for a strategy's performance tracking.

        Args:
            strategy_name: Name of the strategy
            profit: Profit/loss amount
        """
        if strategy_name not in self.performance:
            return

        perf = self.performance[strategy_name]
        perf.total_trades += 1
        perf.last_update = datetime.utcnow()

        if profit > 0:
            perf.winning_trades += 1
            perf.total_profit += profit
        else:
            perf.losing_trades += 1
            perf.total_loss += abs(profit)

        logger.debug(f"Strategy {strategy_name} performance updated: "
                    f"trades={perf.total_trades}, win_rate={perf.win_rate:.2f}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies."""
        summary = {}
        for name, perf in self.performance.items():
            summary[name] = {
                'total_trades': perf.total_trades,
                'win_rate': round(perf.win_rate * 100, 1),
                'net_profit': round(perf.net_profit, 2),
                'profit_factor': round(perf.profit_factor, 2),
                'sharpe_estimate': round(perf.sharpe_estimate, 2),
                'last_update': perf.last_update.isoformat()
            }
        return summary

    def get_best_strategy(self, metric: str = 'sharpe_estimate', min_trades: int = 10) -> Optional[str]:
        """
        Get best performing strategy based on metric.

        Args:
            metric: Metric to compare ('sharpe_estimate', 'win_rate', 'profit_factor')
            min_trades: Minimum trades required for consideration

        Returns:
            Strategy name or None
        """
        candidates = []

        for name, perf in self.performance.items():
            if perf.total_trades >= min_trades:
                score = getattr(perf, metric, 0)
                candidates.append((name, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def reset_all(self) -> None:
        """Reset all strategies (for backtesting)."""
        for strategy in self.strategies.values():
            strategy.reset()
        for perf in self.performance.values():
            perf.total_trades = 0
            perf.winning_trades = 0
            perf.losing_trades = 0
            perf.total_profit = 0.0
            perf.total_loss = 0.0