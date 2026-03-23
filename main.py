#!/usr/bin/env python3
"""
Derivtex - Deriv Volatility 30 Trading Bot
Entry point for the application.
"""

import asyncio
import sys
import signal
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    print("Warning: .env file not found. Using environment variables.")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from bot import DerivBot
from config import load_config
from logger import setup_logger

logger = setup_logger(__name__)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Shutdown signal received, stopping bot...")
    sys.exit(0)

async def main():
    """Main entry point."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Initialize and start bot
        bot = DerivBot(config)
        await bot.start()

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())