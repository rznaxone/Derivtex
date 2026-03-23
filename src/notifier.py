"""
Notification system for Derivtex.
Supports Telegram and Discord alerts.
"""

from typing import Dict, Any, Optional
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Notifier:
    """
    Sends notifications via Telegram and Discord.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize notifier.

        Args:
            config: Configuration dictionary with notification settings
        """
        self.config = config
        self.notif_config = config.get('notifications', {})

        self.telegram_enabled = self.notif_config.get('telegram', {}).get('enabled', False)
        self.telegram_token = self.notif_config.get('telegram', {}).get('bot_token', '')
        self.telegram_chat_id = self.notif_config.get('telegram', {}).get('chat_id', '')

        self.discord_enabled = self.notif_config.get('discord', {}).get('enabled', False)
        self.discord_webhook = self.notif_config.get('discord', {}).get('webhook_url', '')

        logger.info(f"Notifier initialized: Telegram={self.telegram_enabled}, Discord={self.discord_enabled}")

    async def send_message(self, message: str, level: str = "info") -> bool:
        """
        Send notification to all enabled channels.

        Args:
            message: Message text
            level: Message level (info, warning, error, critical)

        Returns:
            True if sent successfully to at least one channel
        """
        tasks = []

        if self.telegram_enabled:
            tasks.append(self._send_telegram(message, level))

        if self.discord_enabled:
            tasks.append(self._send_discord(message, level))

        if not tasks:
            return False

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return any(isinstance(r, bool) and r for r in results)

    async def send_trade_notification(self, trade, profit: Optional[float] = None) -> None:
        """
        Send notification about a trade.

        Args:
            trade: Trade object
            profit: Profit/loss (if trade is closed)
        """
        if trade.status.value == 'open':
            message = (
                f"🟢 *Trade Opened*\n"
                f"Symbol: {trade.symbol}\n"
                f"Direction: {trade.direction.upper()}\n"
                f"Entry: {trade.entry_price:.5f}\n"
                f"Size: {trade.position_size:.2f}\n"
                f"SL: {trade.stop_loss:.5f}\n"
                f"TP: {trade.take_profit:.5f}\n"
                f"Time: {datetime.fromtimestamp(trade.entry_time).strftime('%H:%M:%S')}"
            )
        else:
            emoji = "✅" if profit and profit > 0 else "❌"
            message = (
                f"{emoji} *Trade Closed*\n"
                f"Symbol: {trade.symbol}\n"
                f"Direction: {trade.direction.upper()}\n"
                f"Entry: {trade.entry_price:.5f}\n"
                f"Exit: {trade.exit_price:.5f}\n"
                f"P&L: ${profit:.2f}\n"
                f"Reason: {trade.exit_reason}\n"
                f"Time: {datetime.fromtimestamp(trade.exit_time).strftime('%H:%M:%S')}"
            )

        await self.send_message(message, level="info" if not profit or profit > 0 else "warning")

    async def send_error(self, error_msg: str, exc_info: bool = False) -> None:
        """Send error notification."""
        message = f"🚨 *Error*\n```\n{error_msg}\n```"
        await self.send_message(message, level="error")

    async def send_status(self, status_msg: str) -> None:
        """Send status update."""
        message = f"ℹ️ *Status*\n{status_msg}"
        await self.send_message(message, level="info")

    async def _send_telegram(self, message: str, level: str) -> bool:
        """Send message to Telegram."""
        if not self.telegram_token or not self.telegram_chat_id:
            return False

        try:
            # Format message with Markdown
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"

            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_notification": level in ["debug", "info"]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        logger.debug(f"Telegram message sent: {message[:50]}...")
                        return True
                    else:
                        logger.error(f"Telegram send failed: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    async def _send_discord(self, message: str, level: str) -> bool:
        """Send message to Discord via webhook."""
        if not self.discord_webhook:
            return False

        try:
            # Discord color based on level
            colors = {
                "info": 0x3498db,      # Blue
                "warning": 0xf39c12,   # Orange
                "error": 0xe74c3c,     # Red
                "critical": 0x8e44ad   # Purple
            }
            color = colors.get(level, 0x3498db)

            payload = {
                "embeds": [{
                    "title": "Derivtex Bot",
                    "description": message,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook, json=payload, timeout=10) as response:
                    if response.status in [200, 204]:
                        logger.debug(f"Discord message sent: {message[:50]}...")
                        return True
                    else:
                        logger.error(f"Discord send failed: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Discord error: {e}")
            return False