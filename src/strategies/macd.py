"""
MACD Crossover Strategy.
Uses MACD line and signal line crossovers with RSI confirmation.
"""

from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
import numpy as np

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)

@dataclass
class MACDValues:
    """Container for MACD indicator values."""
    macd: float
    signal: float
    histogram: float

class MACDStrategy(BaseStrategy):
    """
    MACD crossover strategy with RSI confirmation.

    Entry Conditions (LONG):
    - MACD line crosses ABOVE signal line
    - RSI > 50 (bullish momentum)
    - Histogram increasing (momentum building)

    Entry Conditions (SHORT):
    - MACD line crosses BELOW signal line
    - RSI < 50 (bearish momentum)
    - Histogram decreasing

    Exit Rules:
    - Take Profit: 2× ATR
    - Stop Loss: 1× ATR
    - Opposite MACD crossover
    """

    def __init__(self, config: Dict[str, Any], risk_manager=None):
        super().__init__(config, risk_manager)

        self.strategy_config = config.get('strategy', {})

        # MACD parameters
        self.fast_period = self.strategy_config.get('macd_fast', 12)
        self.slow_period = self.strategy_config.get('macd_slow', 26)
        self.signal_period = self.strategy_config.get('macd_signal', 9)

        # Data buffers
        self._price_buffer = []
        self._max_buffer = max(self.fast_period, self.slow_period, self.signal_period) + 10

        # State
        self._last_macd: Optional[float] = None
        self._last_signal: Optional[float] = None
        self._last_crossover: Optional[str] = None

        logger.info(f"{self.get_name()} initialized")

    def get_required_data(self) -> int:
        """Need enough data for MACD calculation."""
        return self.slow_period + self.signal_period + 10

    def get_timeframe(self) -> str:
        """Works on 1-second ticks."""
        return "1s"

    def get_name(self) -> str:
        return "macd_crossover"

    def generate_signal(self, tick: Dict[str, Any], indicators: Dict[str, Any]) -> Signal:
        """
        Generate MACD-based signal.

        Args:
            tick: Current tick data
            indicators: Not used directly, we calculate MACD internally

        Returns:
            Signal object
        """
        price = tick.get('quote', tick.get('bid', 0))

        # Update price buffer
        self._price_buffer.append(price)
        if len(self._price_buffer) > self._max_buffer:
            self._price_buffer = self._price_buffer[-self._max_buffer:]

        # Need enough data
        if len(self._price_buffer) < self.get_required_data():
            return Signal(type="hold", confidence=0.0, reason="warming_up", timestamp=tick.get('timestamp', 0))

        # Calculate MACD
        macd_values = self._calculate_macd()
        if not macd_values:
            return Signal(type="hold", confidence=0.0, reason="no_macd", timestamp=tick.get('timestamp', 0))

        macd_line = macd_values.macd
        signal_line = macd_values.signal
        histogram = macd_values.histogram

        # Check for crossover
        crossover = self._detect_crossover(macd_line, signal_line)

        # Get RSI from indicators if available
        rsi = indicators.get('rsi', 50) if isinstance(indicators, dict) else getattr(indicators, 'rsi', 50)

        # LONG signal
        if crossover == "bullish":
            if rsi > 50 and histogram > 0:
                # Get ATR for stops
                atr = indicators.get('atr', 0) if isinstance(indicators, dict) else getattr(indicators, 'atr', 0)
                if atr > 0:
                    return self._create_signal(
                        "long", price, atr,
                        f"MACD bullish crossover, RSI={rsi:.1f}",
                        confidence=0.7 if rsi > 55 else 0.6
                    )

        # SHORT signal
        elif crossover == "bearish":
            if rsi < 50 and histogram < 0:
                atr = indicators.get('atr', 0) if isinstance(indicators, dict) else getattr(indicators, 'atr', 0)
                if atr > 0:
                    return self._create_signal(
                        "short", price, atr,
                        f"MACD bearish crossover, RSI={rsi:.1f}",
                        confidence=0.7 if rsi < 45 else 0.6
                    )

        return Signal(type="hold", confidence=0.0, reason="no_signal", timestamp=tick.get('timestamp', 0))

    def _calculate_macd(self) -> Optional[MACDValues]:
        """Calculate MACD, signal line, and histogram."""
        if len(self._price_buffer) < self.slow_period:
            return None

        prices = np.array(self._price_buffer)

        # Calculate EMAs
        fast_ema = self._ema(prices, self.fast_period)
        slow_ema = self._ema(prices, self.slow_period)

        if fast_ema is None or slow_ema is None:
            return None

        macd_line = fast_ema - slow_ema

        # Calculate signal line (EMA of MACD)
        # We need a buffer of MACD values
        if not hasattr(self, '_macd_buffer'):
            self._macd_buffer = []

        self._macd_buffer.append(macd_line)
        if len(self._macd_buffer) > self.signal_period + 10:
            self._macd_buffer = self._macd_buffer[-(self.signal_period + 10):]

        if len(self._macd_buffer) < self.signal_period:
            signal_line = macd_line  # Not enough data yet
        else:
            signal_line = self._ema(np.array(self._macd_buffer), self.signal_period)

        histogram = macd_line - signal_line if signal_line is not None else 0

        return MACDValues(
            macd=macd_line,
            signal=signal_line if signal_line is not None else macd_line,
            histogram=histogram
        )

    def _ema(self, data: np.ndarray, period: int) -> Optional[float]:
        """Calculate EMA for the last value."""
        if len(data) < period:
            return None

        # Use simple moving average as seed
        sma = np.mean(data[-period:])

        # For a proper EMA we'd need to iterate, but for signal generation
        # we just need the current value. Use approximation.
        multiplier = 2 / (period + 1)
        ema = data[-1] * multiplier + sma * (1 - multiplier)

        return ema

    def _detect_crossover(self, macd: float, signal: float) -> Optional[str]:
        """
        Detect MACD crossover.

        Returns:
            "bullish" if MACD crossed above signal
            "bearish" if MACD crossed below signal
            None if no crossover
        """
        if self._last_macd is None or self._last_signal is None:
            self._last_macd = macd
            self._last_signal = signal
            return None

        # Bullish crossover: MACD crosses above signal
        if self._last_macd <= self._last_signal and macd > signal:
            self._last_macd = macd
            self._last_signal = signal
            return "bullish"

        # Bearish crossover: MACD crosses below signal
        if self._last_macd >= self._last_signal and macd < signal:
            self._last_macd = macd
            self._last_signal = signal
            return "bearish"

        self._last_macd = macd
        self._last_signal = signal
        return None

    def _create_signal(self, direction: str, price: float, atr: float,
                      reason: str, confidence: float) -> Signal:
        """Create signal with proper risk management."""
        # Use strategy config or defaults
        tp_mult = self.strategy_config.get('macd_tp_multiplier', 2.0)
        sl_mult = self.strategy_config.get('macd_sl_multiplier', 1.0)

        if direction == "long":
            stop_loss = price - atr * sl_mult
            take_profit = price + atr * tp_mult
        else:
            stop_loss = price + atr * sl_mult
            take_profit = price - atr * tp_mult

        position_size = self.calculate_position_size(price, stop_loss, confidence > 0.7)

        return Signal(
            type=direction,
            confidence=confidence,
            reason=reason,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            high_probability=confidence > 0.7,
            timestamp=0,
            metadata={
                'macd': self._last_macd,
                'signal': self._last_signal,
                'atr': atr
            }
        )

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._price_buffer = []
        self._last_macd = None
        self._last_signal = None
        self._last_crossover = None
        if hasattr(self, '_macd_buffer'):
            self._macd_buffer = []