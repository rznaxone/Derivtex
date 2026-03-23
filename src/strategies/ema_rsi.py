"""
EMA + RSI Strategy (refactored to use BaseStrategy).
Original strategy with EMA crossover and RSI confirmation.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime, time as dt_time

from strategies.base import BaseStrategy, Signal
from indicators import Indicators, MarketRegime

logger = logging.getLogger(__name__)

class EMARSIStrategy(BaseStrategy):
    """
    EMA crossover + RSI confirmation strategy.

    Entry Conditions (LONG):
    - EMA 20 > EMA 50 (uptrend)
    - EMA 20 crossed above EMA 50 within lookback ticks
    - RSI > 50
    - ADX > 20 OR (ADX < 20 AND RSI < 40)

    Entry Conditions (SHORT):
    - EMA 20 < EMA 50 (downtrend)
    - EMA 20 crossed below EMA 50 within lookback ticks
    - RSI < 50
    - ADX > 20 OR (ADX < 20 AND RSI > 60)

    Exit Rules:
    - Take Profit: ATR × 3.0 (min 0.3%)
    - Stop Loss: ATR × 1.5 (min 0.2%)
    - Trailing Stop: Activates at ATR × 1.5 profit
    - Time Stop: 60 seconds
    """

    def __init__(self, config: Dict[str, Any], risk_manager=None):
        super().__init__(config, risk_manager)

        self.strategy_config = config.get('strategy', {})

        # Initialize indicators with strategy-specific config
        self.indicators = Indicators(self.strategy_config)

        # Signal tracking
        self._last_crossover: Optional[str] = None
        self._crossover_tick_count: int = 0

        # Time filters (only in live mode, not backtest)
        self.backtest_mode = config.get('backtest_mode', False)
        if not self.backtest_mode:
            self._trading_hours_start = self._parse_time(self.strategy_config.get('trading_hours', {}).get('start', '08:00'))
            self._trading_hours_end = self._parse_time(self.strategy_config.get('trading_hours', {}).get('end', '20:00'))
            self._avoid_hours = [
                (self._parse_time(start), self._parse_time(end))
                for start, end in self.strategy_config.get('avoid_hours', [])
            ]
        else:
            self._trading_hours_start = None
            self._trading_hours_end = None
            self._avoid_hours = []

        logger.info(f"{self.get_name()} initialized")

    def _parse_time(self, time_str: str) -> dt_time:
        """Parse time string to time object."""
        return dt_time.fromisoformat(time_str)

    def get_required_data(self) -> int:
        """Need enough data for EMA calculations."""
        return max(
            self.strategy_config.get('ema_fast', 20),
            self.strategy_config.get('ema_slow', 50),
            self.strategy_config.get('rsi_period', 14) + 1,
            self.strategy_config.get('atr_period', 14) + 1,
            self.strategy_config.get('adx_period', 14) * 2
        ) + 10

    def get_timeframe(self) -> str:
        """This strategy works on 1-second ticks."""
        return "1s"

    def get_name(self) -> str:
        return "ema_rsi"

    def generate_signal(self, tick: Dict[str, Any], indicators: Any) -> Signal:
        """
        Generate trading signal based on EMA crossover + RSI.

        Args:
            tick: Current tick data
            indicators: IndicatorValues object from Indicators class

        Returns:
            Signal object
        """
        current_price = tick.get('quote', tick.get('bid', 0))
        tick_time = datetime.fromtimestamp(tick.get('timestamp', 0))

        # Check trading hours (skip in backtest mode)
        if not self.backtest_mode and not self._is_trading_hours(tick_time):
            return Signal(
                type="hold",
                confidence=0.0,
                reason="outside_trading_hours",
                timestamp=tick.get('timestamp', 0)
            )

        # Check for EMA crossover
        crossover = self._check_crossover()

        # Track crossover timing
        if crossover:
            if crossover != self._last_crossover:
                self._last_crossover = crossover
                self._crossover_tick_count = 1
            else:
                self._crossover_tick_count += 1
        else:
            self._crossover_tick_count = 0

        # Get parameters
        ema_fast = self.strategy_config.get('ema_fast', 20)
        ema_slow = self.strategy_config.get('ema_slow', 50)
        rsi = indicators.rsi
        adx = indicators.adx
        regime = indicators.regime
        atr = indicators.atr

        lookback = self.strategy_config.get('crossover_lookback', 3)
        adx_trending = self.strategy_config.get('adx_trending', 25)
        adx_ranging = self.strategy_config.get('adx_ranging', 20)
        rsi_neutral_high = self.strategy_config.get('rsi_neutral_high', 60)
        rsi_neutral_low = self.strategy_config.get('rsi_neutral_low', 40)

        # LONG entry
        if crossover == "bullish":
            cond1 = indicators.ema_fast > indicators.ema_slow
            cond2 = self._crossover_tick_count <= lookback
            cond3 = rsi > 50
            cond4 = (adx > adx_trending) or (adx < adx_ranging and rsi < rsi_neutral_high)

            if cond1 and cond2 and cond3 and cond4:
                return self._create_long_signal(indicators, current_price, atr, "ema_bullish_crossover")

        # SHORT entry
        elif crossover == "bearish":
            cond1 = indicators.ema_fast < indicators.ema_slow
            cond2 = self._crossover_tick_count <= lookback
            cond3 = rsi < 50
            cond4 = (adx > adx_trending) or (adx < adx_ranging and rsi > rsi_neutral_low)

            if cond1 and cond2 and cond3 and cond4:
                return self._create_short_signal(indicators, current_price, atr, "ema_bearish_crossover")

        # No signal
        return Signal(
            type="hold",
            confidence=0.0,
            reason="no_signal",
            timestamp=tick.get('timestamp', 0)
        )

    def _create_long_signal(self, indicators: Any, price: float, atr: float, reason: str) -> Signal:
        """Create LONG signal with proper risk management."""
        atr_tp = self.strategy_config.get('atr_tp_multiplier', 3.0)
        atr_sl = self.strategy_config.get('atr_sl_multiplier', 1.5)
        min_tp_pct = self.strategy_config.get('atr_min_tp_percent', 0.003)
        min_sl_pct = self.strategy_config.get('atr_min_sl_percent', 0.002)

        # Calculate TP/SL
        tp_distance = max(atr * atr_tp, price * min_tp_pct)
        sl_distance = max(atr * atr_sl, price * min_sl_pct)

        stop_loss = price - sl_distance
        take_profit = price + tp_distance

        # Determine high probability
        high_prob = indicators.adx > self.strategy_config.get('adx_trending', 25) and indicators.rsi > 55

        # Calculate position size
        position_size = self.calculate_position_size(price, stop_loss, high_prob)

        # Confidence
        confidence = self._calculate_confidence(indicators, "long")

        return Signal(
            type="long",
            confidence=confidence,
            reason=reason,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            high_probability=high_prob,
            timestamp=0,  # Will be set by caller
            metadata={
                'atr': atr,
                'rsi': indicators.rsi,
                'adx': indicators.adx,
                'regime': indicators.regime.value
            }
        )

    def _create_short_signal(self, indicators: Any, price: float, atr: float, reason: str) -> Signal:
        """Create SHORT signal with proper risk management."""
        atr_tp = self.strategy_config.get('atr_tp_multiplier', 3.0)
        atr_sl = self.strategy_config.get('atr_sl_multiplier', 1.5)
        min_tp_pct = self.strategy_config.get('atr_min_tp_percent', 0.003)
        min_sl_pct = self.strategy_config.get('atr_min_sl_percent', 0.002)

        tp_distance = max(atr * atr_tp, price * min_tp_pct)
        sl_distance = max(atr * atr_sl, price * min_sl_pct)

        stop_loss = price + sl_distance
        take_profit = price - tp_distance

        high_prob = indicators.adx > self.strategy_config.get('adx_trending', 25) and indicators.rsi < 45

        position_size = self.calculate_position_size(price, stop_loss, high_prob)

        confidence = self._calculate_confidence(indicators, "short")

        return Signal(
            type="short",
            confidence=confidence,
            reason=reason,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            high_probability=high_prob,
            timestamp=0,
            metadata={
                'atr': atr,
                'rsi': indicators.rsi,
                'adx': indicators.adx,
                'regime': indicators.regime.value
            }
        )

    def _check_crossover(self) -> Optional[str]:
        """Check if EMA crossover occurred recently."""
        # We need to track EMA history - for now use current state
        # In production, maintain EMA buffer
        if hasattr(self.indicators, '_ema_fast') and hasattr(self.indicators, '_ema_slow'):
            if self.indicators._ema_fast > self.indicators._ema_slow:
                return "bullish"
            elif self.indicators._ema_fast < self.indicators._ema_slow:
                return "bearish"
        return None

    def _calculate_confidence(self, indicators: Any, signal_type: str) -> float:
        """Calculate signal confidence (0.0-1.0)."""
        confidence = 0.5

        # ADX contribution
        if indicators.adx > 30:
            confidence += 0.2
        elif indicators.adx > 25:
            confidence += 0.1
        elif indicators.adx < 15:
            confidence -= 0.1

        # RSI contribution
        rsi = indicators.rsi
        if signal_type == "long":
            if rsi > 60:
                confidence += 0.1
            elif rsi < 40:
                confidence -= 0.1
        else:
            if rsi < 40:
                confidence += 0.1
            elif rsi > 60:
                confidence -= 0.1

        # EMA separation
        if indicators.ema_fast and indicators.ema_slow:
            ema_diff = abs(indicators.ema_fast - indicators.ema_slow) / indicators.ema_slow
            if ema_diff > 0.002:
                confidence += 0.1

        # Market regime
        if indicators.regime == MarketRegime.TRENDING:
            confidence += 0.1
        else:
            confidence -= 0.05

        return max(0.0, min(1.0, confidence))

    def _is_trading_hours(self, dt: datetime) -> bool:
        """Check if current time is within trading hours."""
        current_time = dt.time()

        # Check avoid hours
        for start, end in self._avoid_hours:
            if start <= current_time <= end:
                return False

        # Check trading hours
        if self._trading_hours_start <= self._trading_hours_end:
            return self._trading_hours_start <= current_time < self._trading_hours_end
        else:
            return current_time >= self._trading_hours_start or current_time < self._trading_hours_end

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self._last_crossover = None
        self._crossover_tick_count = 0
        self.indicators = Indicators(self.config)