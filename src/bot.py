"""
Main bot orchestration for Derivtex.
Coordinates all components: Deriv client, strategy, risk manager, trade executor, monitor.
"""

import asyncio
import signal
from typing import Dict, Any, Optional
from datetime import datetime, time as dt_time
import logging
from collections import deque

from deriv_client import DerivClient, TickData
from strategy import Strategy, Signal, SignalType
from risk_manager import RiskManager
from trade_executor import TradeExecutor, TradeStatus
from monitor import Monitor
from notifier import Notifier
from logger import setup_logger
from config import load_config

logger = setup_logger(__name__)

class DerivBot:
    """
    Main bot class that orchestrates all components.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize bot.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False

        # Initialize components
        self.deriv_client = DerivClient(config)
        self.risk_manager = RiskManager(config)
        self.strategy = Strategy(config, self.risk_manager)
        self.trade_executor = TradeExecutor(self.deriv_client, config, self.risk_manager)
        self.monitor = Monitor(config)
        self.notifier = Notifier(config)

        # Set up trade executor callbacks
        self.trade_executor.set_callbacks(
            on_open=self._on_trade_open,
            on_close=self._on_trade_close,
            on_update=self._on_trade_update
        )

        # State
        self._current_price: float = 0.0
        self._last_tick_time: float = 0.0
        self._start_time: Optional[datetime] = None
        self._shutdown_requested = False

        # Statistics
        self._ticks_processed: int = 0
        self._signals_generated: int = 0
        self._trades_executed: int = 0

        logger.info("DerivBot initialized")

    async def start(self) -> None:
        """Start the bot."""
        logger.info("Starting Derivtex bot...")
        self.running = True
        self._start_time = datetime.utcnow()

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Connect to Deriv
            await self.deriv_client.connect()

            # Subscribe to ticks
            instrument = self.config['trading']['instrument']
            self.deriv_client.subscribe_ticks(instrument, self._on_tick)

            # Send startup notification
            await self.notifier.send_status(
                f"🤖 Derivtex bot started\n"
                f"Instrument: {instrument}\n"
                f"Time: {self._start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )

            # Main loop
            await self._main_loop()

        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
            await self.notifier.send_error(f"Bot crashed: {e}")
            raise

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self.running = False
        self._shutdown_requested = True

        # Close open trades? (configurable)
        # await self._close_all_trades()

        await self.deriv_client.disconnect()
        logger.info("Bot stopped")

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_requested = True

    async def _on_tick(self, tick: TickData) -> None:
        """
        Handle incoming tick data.

        Args:
            tick: TickData object
        """
        self._ticks_processed += 1
        self._current_price = tick.quote
        self._last_tick_time = tick.timestamp

        # Convert tick to dict for strategy
        tick_dict = {
            'symbol': tick.symbol,
            'quote': tick.quote,
            'bid': tick.bid,
            'ask': tick.ask,
            'timestamp': tick.timestamp,
            'high': tick.high,
            'low': tick.low,
            'volume': tick.volume
        }

        # Update risk manager with current balance
        balance = await self.deriv_client.get_balance()
        self.risk_manager.update_balance(balance)
        self.monitor.update_balance(balance, tick.timestamp)

        # Update ATR in risk manager
        current_atr = self.strategy.indicators._atr
        if current_atr:
            self.risk_manager.update_atr(current_atr)

        # Check for exit signals on open trades
        open_trades = self.trade_executor.get_open_trades()
        for trade in open_trades:
            exit_signal = self.strategy.get_exit_signal(
                trade.to_dict(),
                self._current_price,
                tick.timestamp
            )
            if exit_signal:
                await self._execute_exit(trade, exit_signal, self._current_price, tick.timestamp)

        # Update open trades (check SL/TP)
        closed_trades = await self.trade_executor.update_open_trades(
            self._current_price,
            tick.timestamp
        )

        # Record closed trades in monitor
        for trade in closed_trades:
            self.monitor.record_trade(trade, trade.profit)

        # Generate new signal
        signal = self.strategy.update(tick_dict)

        if signal and signal.type not in [SignalType.HOLD, SignalType.NONE]:
            self._signals_generated += 1
            await self._execute_signal(signal, tick_dict)

        # Periodic logging
        if self._ticks_processed % 1000 == 0:
            logger.info(f"Processed {self._ticks_processed} ticks | "
                       f"Open trades: {len(open_trades)} | "
                       f"Balance: ${balance:.2f}")

    async def _execute_signal(self, signal: Signal, tick: Dict[str, Any]) -> None:
        """
        Execute a trading signal.

        Args:
            signal: Signal object
            tick: Current tick data
        """
        logger.info(f"Executing signal: {signal.type.value}, confidence={signal.confidence:.2f}")

        # Check if we can trade (risk manager)
        can_trade, reason = self.risk_manager.can_trade(signal.high_probability)
        if not can_trade:
            logger.warning(f"Trade blocked by risk manager: {reason}")
            return

        # Execute trade
        trade = await self.trade_executor.execute_signal(signal, tick)

        if trade:
            self._trades_executed += 1
            logger.info(f"Trade executed: {trade.id}")

            # Send notification
            await self.notifier.send_trade_notification(trade)

    async def _execute_exit(self, trade, exit_signal: Signal, price: float, timestamp: float) -> None:
        """
        Execute an exit signal.

        Args:
            trade: Trade to exit
            exit_signal: Exit signal
            price: Exit price
            timestamp: Exit timestamp
        """
        logger.info(f"Exit signal for {trade.id}: {exit_signal.reason}")

        # For now, we rely on the trade executor's update_open_trades to handle exits
        # In a more sophisticated implementation, we could send a sell order here
        # For Deriv digital options, we might need to sell the contract early
        pass

    def _on_trade_open(self, trade) -> None:
        """Callback when trade opens."""
        logger.info(f"Trade opened: {trade.id}")

    def _on_trade_close(self, trade) -> None:
        """Callback when trade closes."""
        logger.info(f"Trade closed: {trade.id}, P&L: ${trade.profit:.2f}")

        # Send notification
        asyncio.create_task(
            self.notifier.send_trade_notification(trade, trade.profit)
        )

    def _on_trade_update(self, trade) -> None:
        """Callback when trade is updated."""
        pass

    async def _main_loop(self) -> None:
        """Main bot loop - handles reconnection and monitoring."""
        while self.running and not self._shutdown_requested:
            try:
                # Check connection health
                if not self.deriv_client.ws or self.deriv_client.ws.closed:
                    logger.warning("WebSocket disconnected, reconnecting...")
                    await self.deriv_client.connect()
                    instrument = self.config['trading']['instrument']
                    self.deriv_client.subscribe_ticks(instrument, self._on_tick)

                # Periodic status update
                await self._send_periodic_status()

                # Sleep to avoid tight loop
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)

        # Graceful shutdown
        await self.stop()

    async def _send_periodic_status(self) -> None:
        """Send periodic status updates (every 30 minutes)."""
        if not self._start_time:
            return

        uptime = datetime.utcnow() - self._start_time
        if uptime.total_seconds() < 60:
            return

        # Send every 30 minutes
        if int(uptime.total_seconds() / 1800) > int((uptime.total_seconds() - 60) / 1800):
            balance = await self.deriv_client.get_balance()
            stats = self.monitor.get_stats_summary()
            status_msg = (
                f"📊 *Status Update*\n"
                f"Uptime: {str(uptime).split('.')[0]}\n"
                f"Ticks: {self._ticks_processed:,}\n"
                f"Signals: {self._signals_generated:,}\n"
                f"Trades: {self._trades_executed:,}\n"
                f"Balance: ${balance:.2f}\n"
                f"Stats: {stats}"
            )
            await self.notifier.send_status(status_msg)

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status for dashboard."""
        balance = self.risk_manager.state.account_balance
        risk_stats = self.risk_manager.get_stats()
        dashboard_data = self.monitor.get_dashboard_data()

        return {
            'running': self.running,
            'uptime': str(datetime.utcnow() - self._start_time) if self._start_time else "0:00:00",
            'ticks_processed': self._ticks_processed,
            'signals_generated': self._signals_generated,
            'trades_executed': self._trades_executed,
            'current_price': self._current_price,
            'balance': balance,
            'risk': risk_stats,
            'performance': dashboard_data['metrics'],
            'open_trades': [t.to_dict() for t in self.trade_executor.get_open_trades()],
            'recent_trades': dashboard_data['recent_trades'][-10:]
        }