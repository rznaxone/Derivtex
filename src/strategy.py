"""
Trading strategy for Derivtex.
Implements EMA crossover + RSI confirmation with ADX market regime detection.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, time

from indicators import Indicators, MarketRegime, IndicatorValues
from risk_manager import RiskManager, CircuitBreakerReason

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types."""
    LONG = "long"
    SHORT = "short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
    NONE = "none"

@dataclass
class Signal:
    """Trading signal with metadata."""
    type: SignalType
    confidence: float  # 0.0 to 1.0
    reason: str
    indicator_values: IndicatorValues
    timestamp: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    high_probability: bool = False

class Strategy:
    """
    EMA Crossover + RSI strategy with ADX market regime detection.
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager, backtest_mode: bool = False):
        """
        Initialize strategy.

        Args:
            config: Configuration dictionary
            risk_manager: Risk manager instance
            backtest_mode: If True, skip time filters (for backtesting)
        """
        self.config = config
        self.strategy_config = config['strategy']
        self.trading_config = config['trading']
        self.risk_manager = risk_manager
        self.backtest_mode = backtest_mode

        # Initialize indicators
        self.indicators = Indicators(config)

        # Signal tracking
        self._last_signal: Optional[Signal] = None
        self._last_crossover: Optional[str] = None
        self._crossover_tick_count: int = 0

        # Time filters (only if not in backtest mode)
        if not backtest_mode:
            self._trading_hours_start = self._parse_time(self.trading_config['trading_hours']['start'])
            self._trading_hours_end = self._parse_time(self.trading_config['trading_hours']['end'])
            self._avoid_hours = [
                self._parse_time_range(hr) for hr in self.trading_config.get('avoid_hours', [])
            ]
        else:
            self._trading_hours_start = None
            self._trading_hours_end = None
            self._avoid_hours = []

        logger.info("Strategy initialized")

    def _parse_time(self, time_str: str) -> time:
        """Parse time string to time object."""
        return time.fromisoformat(time_str)

    def _parse_time_range(self, range_str: str) -> Tuple[time, time]:
        """Parse time range string like '00:00-02:00'."""
        start_str, end_str = range_str.split('-')
        return (self._parse_time(start_str), self._parse_time(end_str))

    def _is_trading_hours(self, dt: datetime) -> bool:
        """Check if current time is within trading hours."""
        # In backtest mode, always allow trading
        if self.backtest_mode:
            return True

        current_time = dt.time()

        # Check avoid hours first
        for start, end in self._avoid_hours:
            if start <= current_time <= end:
                return False

        # Check trading hours
        if self._trading_hours_start <= self._trading_hours_end:
            return self._trading_hours_start <= current_time < self._trading_hours_end
        else:
            # Overnight hours (e.g., 22:00-06:00)
            return current_time >= self._trading_hours_start or current_time < self._trading_hours_end

    def update(self, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        Process new tick and generate trading signal.

        Args:
            tick: Tick data dictionary

        Returns:
            Signal if a trade should be executed, None otherwise
        """
        # Update indicators
        values = self.indicators.update(tick)

        # Check if we can trade (risk manager)
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.debug(f"Trading blocked: {reason}")
            return None

        # Check trading hours
        tick_time = datetime.utcnow()
        if not self._is_trading_hours(tick_time):
            logger.debug(f"Outside trading hours: {tick_time.time()}")
            return None

        # Generate signal
        signal = self._generate_signal(values, tick)

        if signal and signal.type != SignalType.HOLD and signal.type != SignalType.NONE:
            # Check if this is a new signal (avoid duplicates)
            if self._is_new_signal(signal):
                self._last_signal = signal
                logger.info(f"Signal generated: {signal.type.value}, confidence={signal.confidence:.2f}, reason={signal.reason}")
                return signal

        return None

    def _generate_signal(self, values: IndicatorValues, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        Generate trading signal based on indicator values.

        Args:
            values: Current indicator values
            tick: Current tick data

        Returns:
            Signal or None
        """
        current_price = tick.get('quote', tick.get('bid', 0))

        # Check for EMA crossover
        crossover = self.indicators.check_crossover(
            lookback=self.strategy_config['crossover_lookback']
        )

        # Track crossover timing
        if crossover:
            if crossover != self._last_crossover:
                self._last_crossover = crossover
                self._crossover_tick_count = 1
            else:
                self._crossover_tick_count += 1
        else:
            self._crossover_tick_count = 0

        # Get strategy parameters
        ema_fast = self.strategy_config['ema_fast']
        ema_slow = self.strategy_config['ema_slow']
        rsi = values.rsi
        adx = values.adx
        regime = values.regime

        # Entry conditions
        if crossover == "bullish":
            # LONG entry conditions
            cond1 = values.ema_fast > values.ema_slow  # Uptrend
            cond2 = self._crossover_tick_count <= self.strategy_config['crossover_lookback']  # Recent crossover
            cond3 = rsi > 50  # Bullish momentum
            cond4 = (adx > self.strategy_config['adx_trending']) or \
                   (adx < self.strategy_config['adx_ranging'] and rsi < self.strategy_config['rsi_neutral_high'])

            if cond1 and cond2 and cond3 and cond4:
                # Calculate stop loss and take profit
                atr = values.atr
                if atr > 0:
                    stop_distance = atr * self.strategy_config['atr_sl_multiplier']
                    tp_distance = atr * self.strategy_config['atr_tp_multiplier']

                    # Apply minimums
                    min_sl = current_price * self.strategy_config['atr_min_sl_percent']
                    min_tp = current_price * self.strategy_config['atr_min_tp_percent']

                    stop_distance = max(stop_distance, min_sl)
                    tp_distance = max(tp_distance, min_tp)

                    stop_loss = current_price - stop_distance
                    take_profit = current_price + tp_distance

                    # Determine if high probability
                    high_prob = adx > self.strategy_config['adx_trending'] and rsi > 55

                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        current_price, stop_loss, high_prob
                    )

                    confidence = self._calculate_confidence(values, SignalType.LONG)

                    return Signal(
                        type=SignalType.LONG,
                        confidence=confidence,
                        reason=f"EMA bullish crossover, RSI={rsi:.1f}, ADX={adx:.1f}, regime={regime.value}",
                        indicator_values=values,
                        timestamp=tick.get('timestamp', 0),
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size=position_size,
                        high_probability=high_prob
                    )

        elif crossover == "bearish":
            # SHORT entry conditions
            cond1 = values.ema_fast < values.ema_slow  # Downtrend
            cond2 = self._crossover_tick_count <= self.strategy_config['crossover_lookback']
            cond3 = rsi < 50  # Bearish momentum
            cond4 = (adx > self.strategy_config['adx_trending']) or \
                   (adx < self.strategy_config['adx_ranging'] and rsi > self.strategy_config['rsi_neutral_low'])

            if cond1 and cond2 and cond3 and cond4:
                atr = values.atr
                if atr > 0:
                    stop_distance = atr * self.strategy_config['atr_sl_multiplier']
                    tp_distance = atr * self.strategy_config['atr_tp_multiplier']

                    min_sl = current_price * self.strategy_config['atr_min_sl_percent']
                    min_tp = current_price * self.strategy_config['atr_min_tp_percent']

                    stop_distance = max(stop_distance, min_sl)
                    tp_distance = max(tp_distance, min_tp)

                    stop_loss = current_price + stop_distance
                    take_profit = current_price - tp_distance

                    high_prob = adx > self.strategy_config['adx_trending'] and rsi < 45

                    position_size = self.risk_manager.calculate_position_size(
                        current_price, stop_loss, high_prob
                    )

                    confidence = self._calculate_confidence(values, SignalType.SHORT)

                    return Signal(
                        type=SignalType.SHORT,
                        confidence=confidence,
                        reason=f"EMA bearish crossover, RSI={rsi:.1f}, ADX={adx:.1f}, regime={regime.value}",
                        indicator_values=values,
                        timestamp=tick.get('timestamp', 0),
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size=position_size,
                        high_probability=high_prob
                    )

        # Exit signal logic would go here (trailing stop, time stop, etc.)
        # These are typically managed by the trade executor

        return None

    def _calculate_confidence(self, values: IndicatorValues, signal_type: SignalType) -> float:
        """
        Calculate signal confidence score (0.0 to 1.0).

        Factors:
        - ADX strength (trending = higher confidence)
        - RSI positioning
        - EMA separation
        - Market regime alignment
        """
        confidence = 0.5  # Base confidence

        # ADX contribution
        adx = values.adx
        if adx > 30:
            confidence += 0.2
        elif adx > 25:
            confidence += 0.1
        elif adx < 15:
            confidence -= 0.1

        # RSI contribution
        rsi = values.rsi
        if signal_type == SignalType.LONG:
            if rsi > 60:
                confidence += 0.1
            elif rsi < 40:
                confidence -= 0.1
        else:  # SHORT
            if rsi < 40:
                confidence += 0.1
            elif rsi > 60:
                confidence -= 0.1

        # EMA separation
        ema_diff = abs(values.ema_fast - values.ema_slow) / values.ema_slow
        if ema_diff > 0.002:  # 0.2%
            confidence += 0.1

        # Market regime
        if values.regime == MarketRegime.TRENDING:
            confidence += 0.1
        elif values.regime == MarketRegime.RANGING and self._is_ranging_suitable(signal_type, rsi):
            confidence += 0.05
        else:
            confidence -= 0.05

        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return confidence

    def _is_ranging_suitable(self, signal_type: SignalType, rsi: float) -> bool:
        """Check if ranging market conditions are suitable for the signal."""
        if signal_type == SignalType.LONG:
            return rsi < self.strategy_config['rsi_neutral_high']
        else:
            return rsi > self.strategy_config['rsi_neutral_low']

    def _is_new_signal(self, signal: Signal) -> bool:
        """
        Check if this is a new signal (different from last or enough time passed).

        Returns:
            True if signal is new
        """
        if not self._last_signal:
            return True

        # Different signal type
        if signal.type != self._last_signal.type:
            return True

        # Same signal type but different price (significant move)
        if signal.entry_price and self._last_signal.entry_price:
            price_diff = abs(signal.entry_price - self._last_signal.entry_price)
            if price_diff > signal.entry_price * 0.001:  # 0.1% difference
                return True

        return False

    def get_exit_signal(self, trade: Dict[str, Any], current_price: float,
                       current_time: float) -> Optional[Signal]:
        """
        Check if an open position should be exited.

        Args:
            trade: Open trade information
            current_price: Current market price
            current_time: Current timestamp

        Returns:
            Exit signal if conditions met
        """
        entry_time = trade.get('entry_time', 0)
        entry_price = trade.get('entry_price', 0)
        direction = trade.get('direction', '')
        stop_loss = trade.get('stop_loss', 0)
        take_profit = trade.get('take_profit', 0)
        trailing_active = trade.get('trailing_active', False)
        trailing_stop = trade.get('trailing_stop', 0)

        # Time stop
        time_stop = self.strategy_config['time_stop']
        if current_time - entry_time > time_stop:
            return Signal(
                type=SignalType.EXIT_LONG if direction == 'long' else SignalType.EXIT_SHORT,
                confidence=1.0,
                reason=f"Time stop ({time_stop}s)",
                indicator_values=self.indicators.get_values(),
                timestamp=current_time
            )

        # Stop loss hit
        if direction == 'long' and current_price <= stop_loss:
            return Signal(
                type=SignalType.EXIT_LONG,
                confidence=1.0,
                reason="Stop loss hit",
                indicator_values=self.indicators.get_values(),
                timestamp=current_time
            )
        elif direction == 'short' and current_price >= stop_loss:
            return Signal(
                type=SignalType.EXIT_SHORT,
                confidence=1.0,
                reason="Stop loss hit",
                indicator_values=self.indicators.get_values(),
                timestamp=current_time
            )

        # Take profit hit
        if direction == 'long' and current_price >= take_profit:
            return Signal(
                type=SignalType.EXIT_LONG,
                confidence=1.0,
                reason="Take profit hit",
                indicator_values=self.indicators.get_values(),
                timestamp=current_time
            )
        elif direction == 'short' and current_price <= take_profit:
            return Signal(
                type=SignalType.EXIT_SHORT,
                confidence=1.0,
                reason="Take profit hit",
                indicator_values=self.indicators.get_values(),
                timestamp=current_time
            )

        # Trailing stop logic
        if trailing_active:
            if direction == 'long':
                if current_price > trailing_stop + (self.indicators.state.current_atr * self.strategy_config['atr_trailing_distance']):
                    # Update trailing stop
                    return Signal(
                        type=SignalType.EXIT_LONG,
                        confidence=0.8,
                        reason="Trailing stop hit",
                        indicator_values=self.indicators.get_values(),
                        timestamp=current_time
                    )
            elif direction == 'short':
                if current_price < trailing_stop - (self.indicators.state.current_atr * self.strategy_config['atr_trailing_distance']):
                    return Signal(
                        type=SignalType.EXIT_SHORT,
                        confidence=0.8,
                        reason="Trailing stop hit",
                        indicator_values=self.indicators.get_values(),
                        timestamp=current_time
                    )

        return None