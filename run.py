#!/usr/bin/env python3
"""
Derivtex CLI Runner
Simple script to run the bot with different modes.
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from bot import DerivBot
from config import load_config
from logger import setup_logger
import asyncio

logger = setup_logger(__name__)

async def run_bot():
    """Run the trading bot."""
    try:
        config = load_config()
        bot = DerivBot(config)
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot failed: {e}", exc_info=True)
        sys.exit(1)

def run_backtest(args):
    """Run backtest."""
    from backtest import main as backtest_main
    sys.argv = ['backtest.py'] + args
    asyncio.run(backtest_main())

def fetch_data(args):
    """Fetch historical data."""
    from data_fetcher import main as fetch_main
    sys.argv = ['data_fetcher.py'] + args
    asyncio.run(fetch_main())

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Derivtex - Deriv Volatility 30 Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s bot              # Run the trading bot
  %(prog)s backtest --data historical_ticks.csv --start 2024-01-01 --end 2024-12-31
  %(prog)s fetch --start 2024-01-01 --end 2024-01-31 --output data.csv
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Bot command
    bot_parser = subparsers.add_parser('bot', help='Run the trading bot')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--data', type=str, required=True, help='Historical data CSV')
    backtest_parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    backtest_parser.add_argument('--output', type=str, default='backtest_results', help='Output directory')
    backtest_parser.add_argument('--no-risk-manager', action='store_true', help='Disable risk manager')

    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch historical data')
    fetch_parser.add_argument('--symbol', type=str, default='R_30', help='Trading symbol')
    fetch_parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    fetch_parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    fetch_parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    fetch_parser.add_argument('--max-ticks', type=int, default=100000, help='Maximum ticks')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'bot':
        asyncio.run(run_bot())
    elif args.command == 'backtest':
        # Pass only backtest-related args
        backtest_args = [
            '--data', args.data,
            '--output', args.output,
            '--balance', str(args.balance)
        ]
        if args.start:
            backtest_args.extend(['--start', args.start])
        if args.end:
            backtest_args.extend(['--end', args.end])
        if args.no_risk_manager:
            backtest_args.append('--no-risk-manager')
        run_backtest(backtest_args)
    elif args.command == 'fetch':
        fetch_args = [
            '--symbol', args.symbol,
            '--start', args.start,
            '--end', args.end,
            '--output', args.output,
            '--max-ticks', str(args.max_ticks)
        ]
        fetch_data(fetch_args)

if __name__ == "__main__":
    main()