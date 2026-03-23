"""
Risk management for Derivtex.
Handles position sizing, daily limits, circuit breakers.
"""

from datetime import datetime, time, timedelta
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitBreakerReason(Enum):
    """Reasons for circuit breaker activation."""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    HIGH_VOLATILITY = "high_volatility"
    API_ERROR = "api_error"
    MANUAL = "manual"

@dataclass
class TradeResult:
    """Result of a completed trade."""
    profit: float  # Positive or negative P&L
    timestamp: datetime
    reason: str  # "tp", "sl", "timeout", etc.

@dataclass
class RiskState:
    """Current risk state."""
    account_balance: float
    daily_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    is_paused: bool = False
    pause_until: Optional[datetime] = None
    pause_reason: Optional[CircuitBreakerReason] = None
    last_trade_time: Optional[datetime] = None
    historical_atr: float = 0.0
    current_atr: float = 0.0

class RiskManager:
    """
    Manages trading risk with position sizing, limits, and circuit breakers.
    """

    def __init__(self, config: Dict[str, Any], initial_balance: float = 0.0):
        """
        Initialize risk manager.

        Args:
            config: Configuration dictionary
            initial_balance: Starting account balance
        """
        self.config = config
        self.risk_config = config.get('risk', {})

        # Initialize state
        self.state = RiskState(account_balance=initial_balance)

        # Daily reset tracking
        self._last_reset_date = datetime.utcnow().date()

        # Load limits
        self.risk_per_trade = self.risk_config.get('risk_per_trade', 0.01)
        self.risk_per_trade_high_prob = self.risk_config.get('risk_per_trade_high_prob', 0.02)
        self.daily_loss_limit = self.risk_config.get('daily_loss_limit', 0.05)
        self.daily_trade_limit = self.risk_config.get('daily_trade_limit', 40)
        self.consecutive_loss_limit = self.risk_config.get('consecutive_loss_limit', 3)
        self.pause_after_losses = self.risk_config.get('pause_after_losses_seconds', 900)

        # Per-trade loss limits
        per_trade_config = self.risk_config.get('per_trade_limits', {})
        self.max_loss_percent = per_trade_config.get('max_loss_percent', 0.02)  # 2% default
        self.max_loss_absolute = per_trade_config.get('max_loss_absolute')

        # ATR volatility settings
        self.atr_volatility_threshold = self.risk_config.get('atr_volatility_threshold', 2.0)
        self.atr_increase_threshold = self.risk_config.get('atr_increase_threshold', 0.20)
        self.atr_increase_reduction = self.risk_config.get('atr_increase_reduction', 0.25)

        # Position size limits
        self.max_position_size = self.risk_config.get('max_position_size')
        self.min_position_size = self.risk_config.get('min_position_size', 1.0)

        logger.info("RiskManager initialized")

    def update_balance(self, balance: float) -> None:
        """Update account balance."""
        self.state.account_balance = balance
        self._check_daily_reset()

    def record_trade(self, result: TradeResult) -> None:
        """
        Record a completed trade result.

        Args:
            result: Trade result with profit/loss
        """
        self._check_daily_reset()

        # Update daily P&L
        self.state.daily_pnl += result.profit
        self.state.daily_trades += 1
        self.state.last_trade_time = datetime.utcnow()

        # Update consecutive losses
        if result.profit < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        logger.info(f"Trade recorded: P&L={result.profit:.2f}, "
                   f"Daily P&L={self.state.daily_pnl:.2f}, "
                   f"Consecutive losses={self.state.consecutive_losses}")

        # Check circuit breakers
        self._check_circuit_breakers()

    def update_atr(self, current_atr: float) -> None:
        """Update ATR values for volatility adjustments."""
        self.state.current_atr = current_atr

        # Initialize historical ATR if not set
        if self.state.historical_atr == 0.0:
            self.state.historical_atr = current_atr

    def can_trade(self, high_probability: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is allowed based on risk rules.

        Args:
            high_probability: Whether this is a high-probability setup

        Returns:
            (can_trade, reason_if_not)
        """
        self._check_daily_reset()
        self._check_pause_expiry()

        # Check if paused
        if self.state.is_paused:
            if self.state.pause_until and datetime.utcnow() < self.state.pause_until:
                return False, f"Paused: {self.state.pause_reason.value}"
            # Pause expired
            self.state.is_paused = False
            self.state.pause_until = None
            self.state.pause_reason = None

        # Check daily loss limit
        if self.state.daily_pnl < -self.state.account_balance * self.daily_loss_limit:
            return False, CircuitBreakerReason.DAILY_LOSS_LIMIT.value

        # Check daily trade limit
        if self.state.daily_trades >= self.daily_trade_limit:
            return False, "daily_trade_limit"

        # Check consecutive losses
        if self.state.consecutive_losses >= self.consecutive_loss_limit:
            return False, CircuitBreakerReason.CONSECUTIVE_LOSSES.value

        # Check volatility
        if self._is_high_volatility():
            return False, CircuitBreakerReason.HIGH_VOLATILITY.value

        return True, None

    def calculate_position_size(self, entry_price: float, stop_loss_price: float,
                                high_probability: bool = False) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            high_probability: Whether this is a high-probability setup

        Returns:
            Position size (in lots/units)
        """
        if self.state.account_balance <= 0:
            logger.warning("Account balance not set, using minimum position size")
            return self.min_position_size

        # Calculate risk per trade
        risk_percent = self.risk_per_trade_high_prob if high_probability else self.risk_per_trade
        risk_amount = self.state.account_balance * risk_percent

        # Calculate risk per unit (pip value)
        # For Deriv options, this is the amount risked per $1 of payout
        # Simplified: risk per point difference
        price_diff = abs(entry_price - stop_loss_price)

        if price_diff == 0:
            logger.warning("Zero price difference, using minimum position size")
            return self.min_position_size

        # Position size = risk amount / risk per unit
        position_size = risk_amount / price_diff

        # Apply volatility adjustment
        position_size = self._apply_volatility_adjustment(position_size)

        # Apply per-trade max loss limit
        # Ensure that even if stop loss is hit, loss doesn't exceed max_loss_percent
        if self.max_loss_percent > 0:
            max_loss_amount = self.state.account_balance * self.max_loss_percent
            # The actual loss if stop hit = position_size * price_diff
            # So we need: position_size * price_diff <= max_loss_amount
            max_position_by_loss = max_loss_amount / price_diff if price_diff > 0 else position_size
            if position_size > max_position_by_loss:
                logger.info(f"Position size reduced by max loss limit: {position_size:.2f} -> {max_position_by_loss:.2f}")
                position_size = max_position_by_loss

        # Apply absolute max loss if set
        if self.max_loss_absolute is not None:
            max_position_by_abs = self.max_loss_absolute / price_diff if price_diff > 0 else position_size
            if position_size > max_position_by_abs:
                logger.info(f"Position size reduced by absolute max loss: {position_size:.2f} -> {max_position_by_abs:.2f}")
                position_size = max_position_by_abs

        # Apply limits
        if self.max_position_size and position_size > self.max_position_size:
            position_size = self.max_position_size

        if position_size < self.min_position_size:
            position_size = self.min_position_size

        potential_loss = position_size * price_diff
        logger.info(f"Position size: {position_size:.2f}, risk_amount: ${risk_amount:.2f}, "
                   f"price_diff: {price_diff:.5f}, potential_loss: ${potential_loss:.2f}, "
                   f"account_balance: ${self.state.account_balance:.2f}")

        return position_size

    def _apply_volatility_adjustment(self, base_size: float) -> float:
        """
        Adjust position size based on ATR volatility.

        Returns:
            Adjusted position size
        """
        if self.state.historical_atr == 0 or self.state.current_atr == 0:
            return base_size

        # Calculate ATR change
        atr_change = (self.state.current_atr - self.state.historical_atr) / self.state.historical_atr

        if atr_change > self.atr_increase_threshold:
            reduction = self.atr_increase_reduction
            adjusted = base_size * (1 - reduction)
            logger.info(f"High volatility detected (ATR +{atr_change:.1%}), "
                       f"reducing position size by {reduction:.0%}")
            return adjusted

        return base_size

    def _is_high_volatility(self) -> bool:
        """Check if current volatility exceeds threshold."""
        if self.state.historical_atr == 0 or self.state.current_atr == 0:
            return False

        volatility_ratio = self.state.current_atr / self.state.historical_atr
        return volatility_ratio > self.atr_volatility_threshold

    def _check_circuit_breakers(self) -> None:
        """Check and activate circuit breakers if needed."""
        # Daily loss limit
        if self.state.daily_pnl < -self.state.account_balance * self.daily_loss_limit:
            self._activate_pause(CircuitBreakerReason.DAILY_LOSS_LIMIT, duration=24*3600)
            logger.warning(f"Daily loss limit hit: {self.state.daily_pnl:.2f}")

        # Consecutive losses
        if self.state.consecutive_losses >= self.consecutive_loss_limit:
            self._activate_pause(CircuitBreakerReason.CONSECUTIVE_LOSSES, duration=self.pause_after_losses)
            logger.warning(f"{self.consecutive_loss_limit} consecutive losses, pausing for {self.pause_after_losses}s")

        # High volatility
        if self._is_high_volatility():
            self._activate_pause(CircuitBreakerReason.HIGH_VOLATILITY, duration=300)
            logger.warning(f"High volatility (ATR ratio > {self.atr_volatility_threshold}), pausing for 5 minutes")

    def _activate_pause(self, reason: CircuitBreakerReason, duration: int) -> None:
        """Activate trading pause."""
        self.state.is_paused = True
        self.state.pause_until = datetime.utcnow() + timedelta(seconds=duration)
        self.state.pause_reason = reason
        logger.warning(f"Circuit breaker activated: {reason.value}, paused until {self.state.pause_until}")

    def _check_pause_expiry(self) -> None:
        """Check if pause has expired and deactivate."""
        if self.state.is_paused and self.state.pause_until:
            if datetime.utcnow() >= self.state.pause_until:
                self.state.is_paused = False
                self.state.pause_until = None
                logger.info("Pause expired, trading resumed")

    def _check_daily_reset(self) -> None:
        """Reset daily counters if new day."""
        today = datetime.utcnow().date()
        if today != self._last_reset_date:
            logger.info(f"Daily reset: date changed from {self._last_reset_date} to {today}")
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self._last_reset_date = today

    def get_stats(self) -> Dict[str, Any]:
        """Get current risk statistics."""
        return {
            'account_balance': self.state.account_balance,
            'daily_pnl': self.state.daily_pnl,
            'daily_pnl_percent': (self.state.daily_pnl / self.state.account_balance * 100) if self.state.account_balance > 0 else 0,
            'daily_trades': self.state.daily_trades,
            'daily_trades_remaining': max(0, self.daily_trade_limit - self.state.daily_trades),
            'consecutive_losses': self.state.consecutive_losses,
            'is_paused': self.state.is_paused,
            'pause_reason': self.state.pause_reason.value if self.state.pause_reason else None,
            'pause_remaining': (self.state.pause_until - datetime.utcnow()).total_seconds() if self.state.pause_until else 0,
            'current_atr': self.state.current_atr,
            'historical_atr': self.state.historical_atr,
            'volatility_ratio': (self.state.current_atr / self.state.historical_atr) if self.state.historical_atr > 0 else 0
        }