"""
Deriv API client for WebSocket and REST operations.
Handles authentication, tick streaming, and trade execution.
"""

import asyncio
import json
import hmac
import hashlib
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import websockets
import aiohttp
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TickData:
    """Container for tick data."""
    symbol: str
    quote: float
    bid: float
    ask: float
    timestamp: float
    raw: Dict[str, Any]

@dataclass
class AccountInfo:
    """Account information."""
    balance: float
    currency: str
    id: str
    is_demo: bool

class DerivClient:
    """
    Client for Deriv API using WebSocket for ticks and REST for trading.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Deriv client.

        Args:
            config: Configuration dictionary with deriv credentials
        """
        self.config = config
        self.deriv_config = config['deriv']

        self.app_id = self.deriv_config['app_id']
        self.api_token = self.deriv_config['api_token']
        self.ws_url = self.deriv_config['ws_url']
        self.rest_url = self.deriv_config['rest_url']

        # WebSocket connection
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.ws_task: Optional[asyncio.Task] = None
        self._tick_callbacks: List[Callable[[TickData], None]] = []
        self._running = False

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

        # Authentication
        self._auth_token: Optional[str] = None
        self._account_info: Optional[AccountInfo] = None

    async def connect(self) -> None:
        """Establish WebSocket connection and authenticate."""
        logger.info("Connecting to Deriv WebSocket...")

        try:
            # Build WebSocket URL with app_id
            ws_url = f"{self.ws_url}&app_id={self.app_id}"

            self.ws = await websockets.connect(ws_url)
            logger.info("WebSocket connected")

            # Start message handler
            self.ws_task = asyncio.create_task(self._message_handler())

            # Authenticate
            await self._authenticate()

            # Get account info
            await self._get_account_info()

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False

        if self.ws_task:
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass

        if self.ws:
            await self.ws.close()

        if self.session:
            await self.session.close()

        logger.info("Disconnected from Deriv")

    async def _message_handler(self) -> None:
        """Handle incoming WebSocket messages."""
        self._running = True

        while self._running and self.ws:
            try:
                message = await self.ws.recv()
                data = json.loads(message)

                # Handle different message types
                msg_type = data.get('msg_type')

                if msg_type == 'tick':
                    await self._handle_tick(data)
                elif msg_type == 'buy':
                    await self._handle_buy_response(data)
                elif msg_type == 'proposal':
                    await self._handle_proposal(data)
                elif msg_type == 'error':
                    logger.error(f"Deriv error: {data.get('error', {}).get('message')}")
                elif msg_type == 'pong':
                    pass  # Keepalive
                else:
                    logger.debug(f"Unhandled message type: {msg_type}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message handler: {e}", exc_info=True)
                await asyncio.sleep(1)  # Avoid tight loop on error

    async def _handle_tick(self, data: Dict[str, Any]) -> None:
        """Process tick data."""
        tick = TickData(
            symbol=data.get('symbol', ''),
            quote=data.get('quote', 0),
            bid=data.get('bid', 0),
            ask=data.get('ask', 0),
            timestamp=data.get('timestamp', time.time()),
            raw=data
        )

        # Call registered callbacks
        for callback in self._tick_callbacks:
            try:
                callback(tick)
            except Exception as e:
                logger.error(f"Error in tick callback: {e}")

    async def _handle_buy_response(self, data: Dict[str, Any]) -> None:
        """Handle buy contract response."""
        # Implementation for buy response
        pass

    async def _handle_proposal(self, data: Dict[str, Any]) -> None:
        """Handle proposal response."""
        pass

    async def _authenticate(self) -> None:
        """Authenticate with API token."""
        auth_msg = {
            "authorize": self.api_token
        }

        await self._send(auth_msg)
        logger.info("Authentication request sent")

        # Wait for response (handled in message handler)
        # In production, implement proper response correlation

    async def _get_account_info(self) -> None:
        """Get account information via REST."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        try:
            async with self.session.get(
                f"{self.rest_url}/balance",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self._account_info = AccountInfo(
                        balance=float(data.get('balance', 0)),
                        currency=data.get('currency', 'USD'),
                        id=data.get('id', ''),
                        is_demo=data.get('is_demo', True)
                    )
                    logger.info(f"Account info: balance={self._account_info.balance} {self._account_info.currency}")
                else:
                    logger.error(f"Failed to get account info: {response.status}")
        except Exception as e:
            logger.error(f"Error getting account info: {e}")

    async def _send(self, data: Dict[str, Any]) -> None:
        """Send message to WebSocket."""
        if not self.ws:
            raise ConnectionError("WebSocket not connected")

        message = json.dumps(data)
        await self.ws.send(message)

    def subscribe_ticks(self, symbol: str, callback: Callable[[TickData], None]) -> None:
        """
        Subscribe to tick stream for a symbol.

        Args:
            symbol: Trading symbol (e.g., "R_30")
            callback: Function to call on each tick
        """
        self._tick_callbacks.append(callback)

        # Send subscription request
        asyncio.create_task(self._send({
            "ticks": symbol
        }))
        logger.info(f"Subscribed to ticks for {symbol}")

    def unsubscribe_ticks(self, symbol: str) -> None:
        """Unsubscribe from tick stream."""
        asyncio.create_task(self._send({
            "forget": "ticks"
        }))
        # Remove callbacks for this symbol (simplified)
        logger.info(f"Unsubscribed from ticks for {symbol}")

    async def get_proposal(self, contract_type: str, symbol: str,
                          duration: int, duration_unit: str,
                          amount: float) -> Dict[str, Any]:
        """
        Get proposal for a contract.

        Args:
            contract_type: "DIGITAL" or "CFD"
            symbol: Trading symbol
            duration: Duration value
            duration_unit: "s", "m", "h"
            amount: Stake amount

        Returns:
            Proposal details
        """
        proposal_msg = {
            "proposal": {
                "contract_type": contract_type,
                "symbol": symbol,
                "duration": duration,
                "duration_unit": duration_unit,
                "amount": amount,
                "basis": "stake"
            }
        }

        await self._send(proposal_msg)

        # In production, implement proper response waiting
        # For now, return empty dict
        return {}

    async def buy_contract(self, proposal_id: str, price: float) -> Dict[str, Any]:
        """
        Buy a contract.

        Args:
            proposal_id: Proposal ID from get_proposal
            price: Price to pay (stake)

        Returns:
            Contract details
        """
        buy_msg = {
            "buy": proposal_id,
            "price": price
        }

        await self._send(buy_msg)

        # In production, implement proper response waiting
        return {}

    async def get_balance(self) -> float:
        """Get current account balance."""
        if self._account_info:
            return self._account_info.balance

        # Fetch fresh
        await self._get_account_info()
        return self._account_info.balance if self._account_info else 0.0

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get cached account info."""
        return self._account_info

    async def ping(self) -> bool:
        """Send ping to keep connection alive."""
        try:
            await self._send({"ping": 1})
            return True
        except:
            return False