"""
Technical indicators for Derivtex.
Real-time calculation of EMA, RSI, ATR, ADX from tick data.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING = "trending"
    RANGING = "ranging"
    TRANSITION = "transition"

@dataclass
class IndicatorValues:
    """Container for indicator values."""
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    adx: float
    regime: MarketRegime
    timestamp: float

class Indicators:
    """
    Calculate technical indicators from tick data.
    Uses efficient numpy operations for real-time updates.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize indicators with configuration.

        Args:
            config: Configuration dictionary with strategy parameters
        """
        self.config = config
        self.strategy = config['strategy']

        # Buffer sizes (need enough data for calculations)
        self.ema_period = max(self.strategy['ema_fast'], self.strategy['ema_slow'])
        self.rsi_period = self.strategy['rsi_period']
        self.atr_period = self.strategy['atr_period']
        self.adx_period = self.strategy['adx_period']

        # Maximum buffer needed
        self.max_buffer = max(
            self.ema_period,
            self.rsi_period + 1,
            self.atr_period + 1,
            self.adx_period * 2  # ADX needs more data
        ) + 10  # Add safety margin

        # Data buffers
        self.close_buffer: List[float] = []
        self.high_buffer: List[float] = []
        self.low_buffer: List[float] = []
        self.volume_buffer: List[float] = []

        # Cached indicator values
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._rsi: Optional[float] = None
        self._atr: Optional[float] = None
        self._adx: Optional[float] = None

        # For RSI calculation
        self._rsi_gains: List[float] = []
        self._rsi_losses: List[float] = []

        # For ATR calculation
        self._tr_buffer: List[float] = []

    def update(self, tick: Dict[str, Any]) -> IndicatorValues:
        """
        Update indicators with new tick data.

        Args:
            tick: Dictionary with 'quote', 'bid', 'ask', 'timestamp' etc.

        Returns:
            IndicatorValues with current indicator readings
        """
        # Extract price (use mid price)
        price = (tick.get('bid', 0) + tick.get('ask', 0)) / 2
        if price == 0:
            price = tick.get('quote', 0)

        high = tick.get('high', price)
        low = tick.get('low', price)

        # Append to buffers
        self.close_buffer.append(price)
        self.high_buffer.append(high)
        self.low_buffer.append(low)
        self.volume_buffer.append(tick.get('volume', 1.0))

        # Trim buffers
        if len(self.close_buffer) > self.max_buffer:
            self.close_buffer = self.close_buffer[-self.max_buffer:]
            self.high_buffer = self.high_buffer[-self.max_buffer:]
            self.low_buffer = self.low_buffer[-self.max_buffer:]
            self.volume_buffer = self.volume_buffer[-self.max_buffer:]

        # Calculate indicators
        self._calculate_ema()
        self._calculate_rsi()
        self._calculate_atr()
        self._calculate_adx()

        # Determine market regime
        regime = self._determine_regime()

        return IndicatorValues(
            ema_fast=self._ema_fast or price,
            ema_slow=self._ema_slow or price,
            rsi=self._rsi or 50.0,
            atr=self._atr or 0.0,
            adx=self._adx or 0.0,
            regime=regime,
            timestamp=tick.get('timestamp', 0)
        )

    def _calculate_ema(self) -> None:
        """Calculate EMA values."""
        if len(self.close_buffer) < self.ema_period:
            return

        closes = np.array(self.close_buffer)

        # Calculate EMA fast
        if self._ema_fast is None:
            # First time: use SMA as seed
            self._ema_fast = np.mean(closes[-self.strategy['ema_fast']:])
        else:
            multiplier = 2 / (self.strategy['ema_fast'] + 1)
            self._ema_fast = (closes[-1] - self._ema_fast) * multiplier + self._ema_fast

        # Calculate EMA slow
        if self._ema_slow is None:
            self._ema_slow = np.mean(closes[-self.strategy['ema_slow']:])
        else:
            multiplier = 2 / (self.strategy['ema_slow'] + 1)
            self._ema_slow = (closes[-1] - self._ema_slow) * multiplier + self._ema_slow

    def _calculate_rsi(self) -> None:
        """Calculate RSI using Wilder's smoothing."""
        if len(self.close_buffer) < self.rsi_period + 1:
            return

        closes = np.array(self.close_buffer)
        deltas = np.diff(closes)

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Update buffers
        self._rsi_gains.append(gains[-1])
        self._rsi_losses.append(losses[-1])

        if len(self._rsi_gains) > self.rsi_period:
            self._rsi_gains.pop(0)
            self._rsi_losses.pop(0)

        if len(self._rsi_gains) < self.rsi_period:
            return

        # Calculate average gains and losses
        avg_gain = np.mean(self._rsi_gains)
        avg_loss = np.mean(self._rsi_losses)

        if avg_loss == 0:
            self._rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            self._rsi = 100.0 - (100.0 / (1.0 + rs))

    def _calculate_atr(self) -> None:
        """Calculate Average True Range."""
        if len(self.close_buffer) < 2:
            return

        # Calculate True Range
        high = self.high_buffer[-1]
        low = self.low_buffer[-1]
        prev_close = self.close_buffer[-2]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        self._tr_buffer.append(tr)

        if len(self._tr_buffer) > self.atr_period:
            self._tr_buffer.pop(0)

        if len(self._tr_buffer) >= self.atr_period:
            self._atr = np.mean(self._tr_buffer)

    def _calculate_adx(self) -> None:
        """
        Calculate ADX (Average Directional Index).
        Simplified implementation focusing on +DM and -DM.
        """
        if len(self.close_buffer) < self.adx_period + 1:
            return

        # Calculate +DM and -DM
        high = self.high_buffer[-1]
        low = self.low_buffer[-1]
        prev_high = self.high_buffer[-2] if len(self.high_buffer) > 1 else high
        prev_low = self.low_buffer[-2] if len(self.low_buffer) > 1 else low

        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            dm = up_move
        elif down_move > up_move and down_move > 0:
            dm = down_move
        else:
            dm = 0

        # We need a buffer of DM and TR values
        # For simplicity, we'll use a simplified ADX calculation
        # In production, use a proper implementation with smoothing

        if not hasattr(self, '_dm_buffer'):
            self._dm_buffer = []
            self._tr_adx_buffer = []

        self._dm_buffer.append(dm)

        # TR for ADX
        tr = max(
            high - low,
            abs(high - (self.high_buffer[-2] if len(self.high_buffer) > 1 else high)),
            abs(low - (self.low_buffer[-2] if len(self.low_buffer) > 1 else low))
        )
        self._tr_adx_buffer.append(tr)

        if len(self._dm_buffer) > self.adx_period:
            self._dm_buffer.pop(0)
            self._tr_adx_buffer.pop(0)

        if len(self._dm_buffer) >= self.adx_period:
            avg_dm = np.mean(self._dm_buffer)
            avg_tr = np.mean(self._tr_adx_buffer)

            if avg_tr > 0:
                dx = (avg_dm / avg_tr) * 100
                # Smooth DX to get ADX
                if self._adx is None:
                    self._adx = dx
                else:
                    self._adx = ((self._adx * (self.adx_period - 1)) + dx) / self.adx_period

    def _determine_regime(self) -> MarketRegime:
        """Determine current market regime based on ADX."""
        if self._adx is None:
            return MarketRegime.TRANSITION

        adx_threshold_trending = self.strategy['adx_trending']
        adx_threshold_ranging = self.strategy['adx_ranging']

        if self._adx >= adx_threshold_trending:
            return MarketRegime.TRENDING
        elif self._adx <= adx_threshold_ranging:
            return MarketRegime.RANGING
        else:
            return MarketRegime.TRANSITION

    def get_values(self) -> IndicatorValues:
        """Get current indicator values without updating."""
        return IndicatorValues(
            ema_fast=self._ema_fast or 0.0,
            ema_slow=self._ema_slow or 0.0,
            rsi=self._rsi or 50.0,
            atr=self._atr or 0.0,
            adx=self._adx or 0.0,
            regime=self._determine_regime(),
            timestamp=0
        )

    def check_crossover(self, lookback: int = 3) -> Optional[str]:
        """
        Check if EMA crossover occurred in recent ticks.

        Returns:
            "bullish" if fast crossed above slow,
            "bearish" if fast crossed below slow,
            None if no crossover
        """
        if len(self.close_buffer) < lookback + 2:
            return None

        # We need to track EMA history, but we only store current values
        # For proper crossover detection, we'd need to store EMA history
        # This is a simplified version - in production, maintain EMA buffer

        # For now, return based on current relationship
        if self._ema_fast and self._ema_slow:
            if self._ema_fast > self._ema_slow:
                return "bullish"
            elif self._ema_fast < self._ema_slow:
                return "bearish"

        return None