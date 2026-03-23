"""
Trade executor for Derivtex.
Handles order placement, modification, and tracking.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

from risk_manager import RiskManager  # For type hint

class TradeStatus(Enum):
    """Trade status."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Trade:
    """Represents an open or closed trade."""
    id: str
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_time: float
    position_size: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    trailing_active: bool = False
    status: TradeStatus = TradeStatus.PENDING
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    exit_reason: Optional[str] = None
    profit: Optional[float] = None
    contract_id: Optional[str] = None
    proposal_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'trailing_active': self.trailing_active,
            'status': self.status.value,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'exit_reason': self.exit_reason,
            'profit': self.profit,
            'contract_id': self.contract_id,
            'proposal_id': self.proposal_id
        }

class TradeExecutor:
    """
    Executes trades via Deriv API and manages open positions.
    """

    def __init__(self, deriv_client, config: Dict[str, Any], risk_manager: RiskManager):
        """
        Initialize trade executor.

        Args:
            deriv_client: DerivClient instance
            config: Configuration dictionary
            risk_manager: Risk manager instance
        """
        self.client = deriv_client
        self.config = config
        self.trading_config = config['trading']
        self.risk_manager = risk_manager

        self.trades: Dict[str, Trade] = {}
        self._trade_counter = 0

        # Callbacks
        self._on_trade_open: Optional[callable] = None
        self._on_trade_close: Optional[callable] = None
        self._on_trade_update: Optional[callable] = None

    def set_callbacks(self, on_open: callable = None, on_close: callable = None,
                     on_update: callable = None) -> None:
        """Set event callbacks."""
        self._on_trade_open = on_open
        self._on_trade_close = on_close
        self._on_trade_update = on_update

    async def execute_signal(self, signal, tick: Dict[str, Any]) -> Optional[Trade]:
        """
        Execute a trading signal.

        Args:
            signal: Signal object with trade parameters
            tick: Current tick data

        Returns:
            Trade object if successful, None otherwise
        """
        try:
            # Get proposal from Deriv
            proposal = await self.client.get_proposal(
                contract_type=self.trading_config['contract_type'],
                symbol=self.trading_config['instrument'],
                duration=self.trading_config['duration'],
                duration_unit=self.trading_config['duration_unit'],
                amount=signal.position_size
            )

            if not proposal or 'proposal' not in proposal:
                logger.error(f"Failed to get proposal: {proposal}")
                return None

            proposal_id = proposal['proposal']['id']
            price = proposal['proposal']['ask_price']  # or 'bid_price' for sell

            # Buy contract
            buy_result = await self.client.buy_contract(proposal_id, price)

            if not buy_result or 'buy' not in buy_result:
                logger.error(f"Failed to buy contract: {buy_result}")
                return None

            contract_id = buy_result['buy']['contract_id']

            # Create trade record
            trade_id = f"trade_{self._trade_counter}"
            self._trade_counter += 1

            trade = Trade(
                id=trade_id,
                symbol=self.trading_config['instrument'],
                direction=signal.type.value.replace('_', ''),  # "long" or "short"
                entry_price=signal.entry_price,
                entry_time=signal.timestamp,
                position_size=signal.position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                trailing_stop=signal.entry_price,  # Initialize at entry
                trailing_active=False,
                status=TradeStatus.OPEN,
                contract_id=contract_id,
                proposal_id=proposal_id
            )

            self.trades[trade_id] = trade

            logger.info(f"Trade executed: {trade_id}, direction={trade.direction}, "
                       f"size={trade.position_size}, entry={trade.entry_price}")

            # Trigger callback
            if self._on_trade_open:
                self._on_trade_open(trade)

            return trade

        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return None

    async def update_open_trades(self, current_price: float, current_time: float) -> List[Trade]:
        """
        Update all open trades and check for exit conditions.

        Args:
            current_price: Current market price
            current_time: Current timestamp

        Returns:
            List of trades that were closed in this update
        """
        closed_trades = []

        for trade_id, trade in list(self.trades.items()):
            if trade.status != TradeStatus.OPEN:
                continue

            # Check stop loss
            if trade.direction == 'long' and current_price <= trade.stop_loss:
                await self._close_trade(trade_id, current_price, current_time, "stop_loss")
                closed_trades.append(trade)

            elif trade.direction == 'short' and current_price >= trade.stop_loss:
                await self._close_trade(trade_id, current_price, current_time, "stop_loss")
                closed_trades.append(trade)

            # Check take profit
            elif trade.direction == 'long' and current_price >= trade.take_profit:
                await self._close_trade(trade_id, current_price, current_time, "take_profit")
                closed_trades.append(trade)

            elif trade.direction == 'short' and current_price <= trade.take_profit:
                await self._close_trade(trade_id, current_price, current_time, "take_profit")
                closed_trades.append(trade)

            # Check trailing stop
            elif trade.trailing_active:
                if trade.direction == 'long' and current_price <= trade.trailing_stop:
                    await self._close_trade(trade_id, current_price, current_time, "trailing_stop")
                    closed_trades.append(trade)
                elif trade.direction == 'short' and current_price >= trade.trailing_stop:
                    await self._close_trade(trade_id, current_price, current_time, "trailing_stop")
                    closed_trades.append(trade)

            # Update trailing stop if in profit
            if trade.status == TradeStatus.OPEN:
                await self._update_trailing_stop(trade, current_price)

        return closed_trades

    async def _update_trailing_stop(self, trade: Trade, current_price: float) -> None:
        """
        Update trailing stop for a trade.

        Args:
            trade: Trade object
            current_price: Current market price
        """
        atr_multiplier = self.config['strategy']['atr_trailing_distance']
        atr = self.risk_manager.state.current_atr

        if atr <= 0:
            return

        activation_threshold = self.config['strategy']['atr_trailing_activation']
        activation_price = trade.entry_price + (atr * activation_threshold) if trade.direction == 'long' else trade.entry_price - (atr * activation_threshold)

        # Check if we should activate trailing stop
        if not trade.trailing_active:
            if trade.direction == 'long' and current_price >= activation_price:
                trade.trailing_active = True
                trade.trailing_stop = current_price - (atr * atr_multiplier)
                logger.info(f"Trailing stop activated for {trade.id} at {trade.trailing_stop:.5f}")
            elif trade.direction == 'short' and current_price <= activation_price:
                trade.trailing_active = True
                trade.trailing_stop = current_price + (atr * atr_multiplier)
                logger.info(f"Trailing stop activated for {trade.id} at {trade.trailing_stop:.5f}")
        else:
            # Update trailing stop
            if trade.direction == 'long':
                new_stop = current_price - (atr * atr_multiplier)
                if new_stop > trade.trailing_stop:
                    trade.trailing_stop = new_stop
                    logger.debug(f"Trailing stop updated for {trade.id} to {trade.trailing_stop:.5f}")
            else:
                new_stop = current_price + (atr * atr_multiplier)
                if new_stop < trade.trailing_stop:
                    trade.trailing_stop = new_stop
                    logger.debug(f"Trailing stop updated for {trade.id} to {trade.trailing_stop:.5f}")

    async def _close_trade(self, trade_id: str, exit_price: float, exit_time: float, reason: str) -> None:
        """
        Close a trade.

        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Exit reason
        """
        trade = self.trades.get(trade_id)
        if not trade:
            logger.warning(f"Trade {trade_id} not found for closing")
            return

        # Calculate profit
        if trade.direction == 'long':
            profit = (exit_price - trade.entry_price) * trade.position_size
        else:
            profit = (trade.entry_price - exit_price) * trade.position_size

        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.exit_reason = reason
        trade.profit = profit
        trade.status = TradeStatus.CLOSED

        logger.info(f"Trade closed: {trade_id}, profit={profit:.2f}, reason={reason}")

        # Record in risk manager
        from risk_manager import TradeResult
        result = TradeResult(
            profit=profit,
            timestamp=datetime.fromtimestamp(exit_time),
            reason=reason
        )
        self.risk_manager.record_trade(result)

        # Trigger callback
        if self._on_trade_close:
            self._on_trade_close(trade)

        # Remove from open trades
        del self.trades[trade_id]

    async def cancel_trade(self, trade_id: str) -> bool:
        """
        Cancel a pending trade.

        Args:
            trade_id: Trade ID

        Returns:
            True if cancelled successfully
        """
        trade = self.trades.get(trade_id)
        if not trade or trade.status != TradeStatus.PENDING:
            return False

        # In production, send cancel request to Deriv API
        trade.status = TradeStatus.CANCELLED

        if trade.proposal_id:
            # Send cancel request
            pass

        logger.info(f"Trade cancelled: {trade_id}")
        return True

    def get_open_trades(self) -> List[Trade]:
        """Get all open trades."""
        return [t for t in self.trades.values() if t.status == TradeStatus.OPEN]

    def get_closed_trades(self) -> List[Trade]:
        """Get all closed trades."""
        return [t for t in self.trades.values() if t.status == TradeStatus.CLOSED]

    def get_all_trades(self) -> List[Trade]:
        """Get all trades."""
        return list(self.trades.values())

    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get trade by ID."""
        return self.trades.get(trade_id)