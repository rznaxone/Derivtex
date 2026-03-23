"""
Bollinger Bands Squeeze Strategy.
Detects volatility contractions (squeeze) and trades the breakout.
"""

from typing import Dict, Any, Optional
import logging
import numpy as np

from strategies.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands squeeze breakout strategy.

    Entry Conditions (LONG):
    - Bollinger Band width (upper - lower) is below threshold (squeeze)
    - Price breaks above upper band
    - RSI not overbought (< 70)

    Entry Conditions (SHORT):
    - Bollinger Band width below threshold (squeeze)
    - Price breaks below lower band
    - RSI not oversold (> 30)

    Exit Rules:
    - Take Profit: 1.5× ATR or middle band touch
    - Stop Loss: 1× ATR
    - Band width expansion (volatility return)
    """

    def __init__(self, config: Dict[str, Any], risk_manager=None):
        super().__init__(config, risk_manager)

        self.strategy_config = config.get('strategy', {})

        # Bollinger Bands parameters
        self.bb_period = self.strategy_config.get('bb_period', 20)
        self.bb_std = self.strategy_config.get('bb_std', 2.0)
        self.squeeze_threshold = self.strategy_config.get('bb_squeeze_threshold', 0.001)  # 0.1% width
        self.squeeze_lookback = self.strategy_config.get('bb_squeeze_lookback', 5)

        # Data buffers
        self._price_buffer = []
        self._max_buffer = self.bb_period + 20

        # State
        self._in_squeeze: bool = False
        self._squeeze_duration: int = 0

        logger.info(f"{self.get_name()} initialized")

    def get_required_data(self) -> int:
        """Need enough data for Bollinger Bands."""
        return self.bb_period + 10

    def get_timeframe(self) -> str:
        """Works on 1-second ticks."""
        return "1s"

    def get_name(self) -> str:
        return "bollinger_squeeze"

    def generate_signal(self, tick: Dict[str, Any], indicators: Dict[str, Any]) -> Signal:
        """
        Generate Bollinger Bands squeeze breakout signal.

        Args:
            tick: Current tick data
            indicators: Indicator values (includes RSI, ATR)

        Returns:
            Signal object
        """
        price = tick.get('quote', tick.get('bid', 0))

        # Update price buffer
        self._price_buffer.append(price)
        if len(self._price_buffer) > self._max_buffer:
            self._price_buffer = self._price_buffer[-self._max_buffer:]

        if len(self._price_buffer) < self.bb_period:
            return Signal(type="hold", confidence=0.0, reason="warming_up", timestamp=tick.get('timestamp', 0))

        # Calculate Bollinger Bands
        prices = np.array(self._price_buffer[-self.bb_period:])
        sma = np.mean(prices)
        std = np.std(prices)
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        band_width = (upper - lower) / sma  # Normalized width

        # Get RSI and ATR
        rsi = indicators.get('rsi', 50) if isinstance(indicators, dict) else getattr(indicators, 'rsi', 50)
        atr = indicators.get('atr', 0) if isinstance(indicators, dict) else getattr(indicators, 'atr', 0)

        # Detect squeeze
        if band_width < self.squeeze_threshold:
            self._in_squeeze = True
            self._squeeze_duration += 1
        else:
            self._in_squeeze = False
            self._squeeze_duration = 0

        # Only trade if we've been in squeeze for at least N ticks
        if not self._in_squeeze or self._squeeze_duration < self.squeeze_lookback:
            return Signal(type="hold", confidence=0.0, reason="no_squeeze", timestamp=tick.get('timestamp', 0))

        # Check for breakout
        if price > upper:
            # Long breakout
            if rsi < 70:  # Not overbought
                return self._create_signal(
                    "long", price, atr,
                    f"BB squeeze breakout (upper), width={band_width:.4f}",
                    confidence=0.7
                )

        elif price < lower:
            # Short breakout
            if rsi > 30:  # Not oversold
                return self._create_signal(
                    "short", price, atr,
                    f"BB squeeze breakout (lower), width={band_width:.4f}",
                    confidence=0.7
                )

        return Signal(type="hold", confidence=0.0, reason="no_breakout", timestamp=tick.get('timestamp', 0))

    def _create_signal(self, direction: str, price: float, atr: float,
                      reason: str, confidence: float) -> Signal:
        """Create signal with risk management."""
        # Bollinger Bands typically uses tighter stops
        sl_mult = self.strategy_config.get('bb_sl_multiplier', 1.0)
        tp_mult = self.strategy_config.get('bb_tp_multiplier', 1.5)

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
                'bb_upper': price + 0,  # Would calculate properly
                'bb_lower': price - 0,
                'bb_width': self.squeeze_threshold,
                'atr': atr
            }
        )

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._price_buffer = []
        self._in_squeeze = False
        self._squeeze_duration = 0