"""
Configuration loader for Derivtex.
Loads YAML config files and merges with environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML files and environment variables.

    Returns:
        Complete configuration dictionary
    """
    # Load .env file
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path, override=True)

    # Load main config
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load risk config
    risk_config_path = Path(__file__).parent.parent / 'config' / 'risk_config.yaml'
    with open(risk_config_path, 'r') as f:
        risk_config = yaml.safe_load(f)

    # Merge configs
    config['risk'] = {**config.get('risk', {}), **risk_config}

    # Override with environment variables
    _override_from_env(config)

    # Validate required fields
    _validate_config(config)

    return config

def _override_from_env(config: Dict[str, Any]) -> None:
    """Override configuration with environment variables."""
    # Deriv credentials
    if os.getenv('DERIV_APP_ID'):
        config['deriv']['app_id'] = os.getenv('DERIV_APP_ID')
    if os.getenv('DERIV_API_TOKEN'):
        config['deriv']['api_token'] = os.getenv('DERIV_API_TOKEN')

    # Optional overrides
    if os.getenv('DERIV_INSTRUMENT'):
        config['trading']['instrument'] = os.getenv('DERIV_INSTRUMENT')
    if os.getenv('DERIV_CONTRACT_TYPE'):
        config['trading']['contract_type'] = os.getenv('DERIV_CONTRACT_TYPE')
    if os.getenv('DERIV_DURATION'):
        config['trading']['duration'] = int(os.getenv('DERIV_DURATION'))
    if os.getenv('DERIV_DURATION_UNIT'):
        config['trading']['duration_unit'] = os.getenv('DERIV_DURATION_UNIT')

    # Logging
    if os.getenv('LOG_LEVEL'):
        config['monitoring']['log_level'] = os.getenv('LOG_LEVEL')

    # Notifications
    if os.getenv('TELEGRAM_BOT_TOKEN'):
        config['notifications']['telegram']['bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN')
        config['notifications']['telegram']['enabled'] = True
    if os.getenv('TELEGRAM_CHAT_ID'):
        config['notifications']['telegram']['chat_id'] = os.getenv('TELEGRAM_CHAT_ID')

    if os.getenv('DISCORD_WEBHOOK_URL'):
        config['notifications']['discord']['webhook_url'] = os.getenv('DISCORD_WEBHOOK_URL')
        config['notifications']['discord']['enabled'] = True

def _validate_config(config: Dict[str, Any]) -> None:
    """Validate required configuration fields."""
    required = [
        ('deriv', 'app_id'),
        ('deriv', 'api_token'),
        ('trading', 'instrument'),
    ]

    for section, key in required:
        if not config.get(section, {}).get(key):
            raise ValueError(f"Missing required config: {section}.{key}")

    # Validate risk parameters
    risk = config.get('risk', {})
    if risk.get('risk_per_trade', 0) <= 0 or risk.get('risk_per_trade', 0) > 0.1:
        raise ValueError("risk_per_trade must be between 0 and 0.1 (10%)")

    if risk.get('daily_loss_limit', 0) <= 0 or risk.get('daily_loss_limit', 0) > 0.2:
        raise ValueError("daily_loss_limit must be between 0 and 0.2 (20%)")

    if risk.get('min_rr_ratio', 0) < 1:
        raise ValueError("min_rr_ratio must be at least 1")
