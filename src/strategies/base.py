"""
Base strategy interface for Derivtex.
All trading strategies must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Trading signal with metadata."""
    type: str  # "long", "short", "exit_long", "exit_short", "hold"
    confidence: float  # 0.0 to 1.0
    reason: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    high_probability: bool = False
    timestamp: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must implement:
    - generate_signal: Main logic to produce trading signals
    - get_required_data: Minimum number of data points needed
    - get_timeframe: Preferred timeframe (e.g., "1s", "5m")
    - get_name: Unique strategy identifier
    """

    def __init__(self, config: Dict[str, Any], risk_manager=None):
        """
        Initialize strategy.

        Args:
            config: Full bot configuration
            risk_manager: Optional risk manager for position sizing
        """
        self.config = config
        self.risk_manager = risk_manager
        self.strategy_config = config.get('strategy', {})
        self.logger = logging.getLogger(f"{__name__}.{self.get_name()}")

        # State tracking
        self._last_signal: Optional[Signal] = None
        self._signal_counter: int = 0

    @abstractmethod
    def generate_signal(self, tick: Dict[str, Any], indicators: Dict[str, Any]) -> Signal:
        """
        Generate trading signal based on current tick and indicators.

        Args:
            tick: Current tick data with price, volume, etc.
            indicators: Pre-calculated indicator values (EMA, RSI, ATR, etc.)

        Returns:
            Signal object with trade direction and parameters
        """
        pass

    @abstractmethod
    def get_required_data(self) -> int:
        """
        Return minimum number of data points required for this strategy.
        Used by backtest to ensure enough warm-up period.
        """
        pass

    @abstractmethod
    def get_timeframe(self) -> str:
        """
        Return preferred timeframe for this strategy.
        Examples: "1s", "5s", "1m", "5m", "1h"
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return unique strategy name (lowercase, underscores).
        Example: "ema_rsi", "macd_crossover", "bollinger_squeeze"
        """
        pass

    def get_confidence(self, tick: Dict[str, Any], indicators: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the signal (0.0-1.0).
        Can be overridden by strategies for custom logic.
        Default: 0.5 (neutral)
        """
        return 0.5

    def calculate_position_size(self, entry_price: float, stop_loss: float,
                               confidence: float = 0.5) -> float:
        """
        Calculate position size. Uses risk manager if available.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: Signal confidence (0.0-1.0)

        Returns:
            Position size (lots/units)
        """
        if self.risk_manager:
            high_prob = confidence > 0.7
            return self.risk_manager.calculate_position_size(
                entry_price, stop_loss, high_prob
            )
        else:
            # Default: 1 unit
            return 1.0

    def update(self, tick: Dict[str, Any], indicators: Optional[Dict[str, Any]] = None) -> Signal:
        """
        Update strategy with new tick and return signal.
        Includes duplicate signal filtering.

        Args:
            tick: Current tick data
            indicators: Current indicator values (optional, strategies can use own state)

        Returns:
            Signal (or HOLD if no new signal)
        """
        signal = self.generate_signal(tick, indicators)

        # Filter duplicates
        if signal and signal.type not in ["hold", "exit_long", "exit_short"]:
            if self._is_duplicate_signal(signal):
                self.logger.debug(f"Duplicate signal filtered: {signal.type}")
                return Signal(
                    type="hold",
                    confidence=0.0,
                    reason="duplicate",
                    timestamp=tick.get('timestamp', 0)
                )

            self._last_signal = signal
            self._signal_counter += 1

        return signal

    def _is_duplicate_signal(self, signal: Signal) -> bool:
        """
        Check if this signal is essentially the same as the last one.
        Prevents spamming the same signal repeatedly.
        """
        if not self._last_signal:
            return False

        # Same direction
        if signal.type != self._last_signal.type:
            return False

        # Similar entry price (within 0.1%)
        if signal.entry_price and self._last_signal.entry_price:
            price_diff = abs(signal.entry_price - self._last_signal.entry_price)
            if price_diff / self._last_signal.entry_price < 0.001:
                return True

        return False

    def reset(self) -> None:
        """Reset strategy state (useful for backtesting)."""
        self._last_signal = None
        self._signal_counter = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            'name': self.get_name(),
            'signals_generated': self._signal_counter,
            'last_signal': self._last_signal.type if self._last_signal else None,
            'last_signal_time': self._last_signal.timestamp if self._last_signal else 0
        }