"""
Test risk manager max loss limit.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import load_config
from risk_manager import RiskManager

config = load_config()
print(f"Max loss percent: {config['risk'].get('per_trade_limits', {}).get('max_loss_percent')}")

risk_manager = RiskManager(config, 10000.0)

entry = 1025.0
sl = 1027.0  # 2 point difference
position = risk_manager.calculate_position_size(entry, sl, False)

print(f"\nTest: entry=${entry}, sl=${sl}, diff=${abs(entry-sl)}")
print(f"Position size: {position:.2f}")
print(f"Potential loss: {position * abs(entry-sl):.2f}")
print(f"Max allowed loss (2%): ${10000 * 0.02:.2f}")