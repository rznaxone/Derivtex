"""
Data buffer for tick history and indicator calculations.
Manages rolling window of tick data.
"""

from collections import deque
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class Tick:
    """Single tick data point."""
    symbol: str
    quote: float
    bid: float
    ask: float
    timestamp: float
    high: float
    low: float
    volume: float

class DataBuffer:
    """
    Rolling buffer for tick data.
    Stores recent ticks for indicator calculations.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize data buffer.

        Args:
            max_size: Maximum number of ticks to store
        """
        self.max_size = max_size
        self._ticks: deque = deque(maxlen=max_size)
        self._symbols: Dict[str, deque] = {}

    def add_tick(self, tick_data: Dict[str, Any]) -> None:
        """
        Add a new tick to the buffer.

        Args:
            tick_data: Dictionary with tick information
        """
        tick = Tick(
            symbol=tick_data.get('symbol', ''),
            quote=tick_data.get('quote', 0),
            bid=tick_data.get('bid', 0),
            ask=tick_data.get('ask', 0),
            timestamp=tick_data.get('timestamp', time.time()),
            high=tick_data.get('high', tick_data.get('quote', 0)),
            low=tick_data.get('low', tick_data.get('quote', 0)),
            volume=tick_data.get('volume', 1.0)
        )

        self._ticks.append(tick)

        # Maintain per-symbol buffers
        symbol = tick.symbol
        if symbol not in self._symbols:
            self._symbols[symbol] = deque(maxlen=self.max_size)
        self._symbols[symbol].append(tick)

        logger.debug(f"Added tick for {symbol}: {tick.quote:.5f}")

    def get_recent_ticks(self, count: int, symbol: Optional[str] = None) -> List[Tick]:
        """
        Get recent ticks.

        Args:
            count: Number of ticks to retrieve
            symbol: Filter by symbol (None for all)

        Returns:
            List of Tick objects
        """
        if symbol:
            ticks = list(self._symbols.get(symbol, []))
        else:
            ticks = list(self._ticks)

        return ticks[-count:] if count > 0 else ticks

    def get_ticks_since(self, timestamp: float, symbol: Optional[str] = None) -> List[Tick]:
        """
        Get ticks since a specific timestamp.

        Args:
            timestamp: Unix timestamp
            symbol: Filter by symbol

        Returns:
            List of Tick objects
        """
        if symbol:
            ticks = self._symbols.get(symbol, [])
        else:
            ticks = self._ticks

        return [t for t in ticks if t.timestamp >= timestamp]

    def get_latest_tick(self, symbol: Optional[str] = None) -> Optional[Tick]:
        """Get the most recent tick."""
        if symbol:
            buffer = self._symbols.get(symbol, [])
            return buffer[-1] if buffer else None
        else:
            return self._ticks[-1] if self._ticks else None

    def get_price_series(self, count: int, price_type: str = 'quote',
                         symbol: Optional[str] = None) -> List[float]:
        """
        Get price series for indicator calculations.

        Args:
            count: Number of prices to retrieve
            price_type: 'quote', 'bid', 'ask', 'high', 'low'
            symbol: Filter by symbol

        Returns:
            List of prices
        """
        ticks = self.get_recent_ticks(count, symbol)

        if price_type == 'quote':
            return [t.quote for t in ticks]
        elif price_type == 'bid':
            return [t.bid for t in ticks]
        elif price_type == 'ask':
            return [t.ask for t in ticks]
        elif price_type == 'high':
            return [t.high for t in ticks]
        elif price_type == 'low':
            return [t.low for t in ticks]
        else:
            raise ValueError(f"Invalid price_type: {price_type}")

    def get_high_low_volume_series(self, count: int, symbol: Optional[str] = None) -> tuple:
        """
        Get high, low, and volume series.

        Returns:
            Tuple of (highs, lows, volumes)
        """
        ticks = self.get_recent_ticks(count, symbol)
        highs = [t.high for t in ticks]
        lows = [t.low for t in ticks]
        volumes = [t.volume for t in ticks]
        return highs, lows, volumes

    def clear(self, symbol: Optional[str] = None) -> None:
        """Clear buffer."""
        if symbol:
            if symbol in self._symbols:
                self._symbols[symbol].clear()
        else:
            self._ticks.clear()
            self._symbols.clear()

    def size(self, symbol: Optional[str] = None) -> int:
        """Get buffer size."""
        if symbol:
            return len(self._symbols.get(symbol, []))
        return len(self._ticks)

    def get_time_range(self, symbol: Optional[str] = None) -> tuple:
        """Get earliest and latest timestamps."""
        ticks = self.get_recent_ticks(-1, symbol)
        if not ticks:
            return (0, 0)

        timestamps = [t.timestamp for t in ticks]
        return (min(timestamps), max(timestamps))