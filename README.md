# Derivtex 🚀

Automated trading bot for Deriv synthetic volatility 30 (1s) indices.

**Status:** Core components implemented | Ready for testing

> ⚠️ **Warning**: This bot is for educational purposes. Trading involves significant risk. Always test in demo mode first.

## Features

- **Strategy**: EMA crossover + RSI confirmation with ADX market regime detection
- **Risk Management**: Strict position sizing, daily limits, circuit breakers
- **Adaptive**: Adjusts to trending/ranging market conditions
- **Real-time**: 1-second tick analysis with WebSocket feed
- **Monitoring**: Streamlit dashboard for performance tracking
- **Notifications**: Telegram/Discord alerts

## Project Structure

```
Derivtex/
├── src/
│   ├── bot.py              # Main bot orchestration
│   ├── strategy.py         # EMA+RSI strategy with ADX regime detection
│   ├── indicators.py       # Real-time technical indicators (EMA, RSI, ATR, ADX)
│   ├── risk_manager.py     # Position sizing, limits, circuit breakers
│   ├── deriv_client.py     # Deriv WebSocket + REST API client
│   ├── data_buffer.py      # Rolling tick data buffer
│   ├── trade_executor.py   # Order placement and management
│   ├── monitor.py          # Performance tracking and metrics
│   ├── notifier.py         # Telegram/Discord notifications
│   ├── dashboard.py        # Streamlit monitoring dashboard
│   └── data_fetcher.py     # Historical data fetching (synthetic/API)
├── config/
│   ├── config.yaml         # Main configuration (strategy, trading, risk)
│   └── risk_config.yaml    # Advanced risk parameters
├── tests/                  # Unit tests (to be implemented)
├── logs/                   # Runtime logs and trade history
├── data/                   # Historical data (gitignored)
├── backtest_results/       # Backtest reports (gitignored)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image
├── docker-compose.yml      # Multi-service orchestration
├── main.py                 # Direct entry point
├── run.py                  # CLI with subcommands
├── backtest.py             # Backtesting engine
└── README.md
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/rznaxone/Derivtex.git
cd Derivtex

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
cp .env.example .env
```

Edit `.env` and add your Deriv API credentials:

```bash
DERIV_APP_ID=your_app_id
DERIV_API_TOKEN=your_api_token
```

**How to get credentials:**
1. Log in to your Deriv account (demo or real)
2. Go to **Settings** → **API Tokens**
3. Create a new token with **Read** and **Trade** permissions
4. Copy the token to `.env`
5. Get your `app_id` from the [Deriv API portal](https://developers.deriv.com/)

### 3. Run the Bot

```bash
# Using CLI
python run.py bot

# Or directly
python main.py
```

### 4. Access Dashboard

Open browser to: **http://localhost:8501**

The dashboard shows real-time metrics, equity curve, and recent trades.

---

## Backtesting

Before live trading, backtest your strategy on historical data.

### 1. Get Historical Data

```bash
# Generate synthetic data for testing (quick start)
python run.py fetch --start 2024-01-01 --end 2024-01-31 --output data/r30_jan2024.csv

# For real historical data, you'll need to:
# - Use Deriv's API (if available)
# - Or use a data provider like Dukascopy, TrueData, etc.
# - Or export from your own tick history
```

### 2. Run Backtest

```bash
python run.py backtest \
  --data data/r30_jan2024.csv \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --balance 10000 \
  --output backtest_results
```

### 3. View Results

```bash
# View summary in terminal
cat backtest_results/report.json | python -m json.tool

# Open interactive HTML charts
open backtest_results/equity_curve.html
open backtest_results/profit_distribution.html
```

**Key metrics to evaluate:**
- **Win Rate**: Target 55-65%
- **Profit Factor**: Target >1.3
- **Sharpe Ratio**: Target >1.0
- **Max Drawdown**: Target <15%

---

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f bot

# Stop
docker-compose down
```

---

## Configuration

Edit `config/config.yaml` to customize:

- **Strategy parameters**: EMA periods, RSI thresholds, ATR multipliers
- **Risk limits**: Position size, daily loss limit, max trades
- **Trading hours**: When to trade/avoid
- **Notifications**: Telegram/Discord webhooks

Key parameters:

```yaml
strategy:
  ema_fast: 20
  ema_slow: 50
  rsi_period: 14
  atr_tp_multiplier: 3.0    # Take profit: ATR × 3
  atr_sl_multiplier: 1.5    # Stop loss: ATR × 1.5
  time_stop: 60             # Exit after 60 seconds

risk:
  risk_per_trade: 0.01      # 1% per trade
  daily_loss_limit: 0.05    # Stop after 5% daily loss
  daily_trade_limit: 40     # Max 40 trades/day
```

---

## CLI Commands

```bash
# Run trading bot
python run.py bot

# Run backtest
python run.py backtest --data ticks.csv --start 2024-01-01 --end 2024-12-31

# Fetch historical data
python run.py fetch --start 2024-01-01 --end 2024-01-31 --output data.csv

# View help
python run.py --help
```

## Strategy Overview

### Entry Conditions (LONG)
- EMA 20 > EMA 50 (uptrend)
- EMA 20 crossed above EMA 50 in last 3 ticks
- RSI > 50
- ADX > 20 OR (ADX < 20 AND RSI < 40)

### Entry Conditions (SHORT)
- EMA 20 < EMA 50 (downtrend)
- EMA 20 crossed below EMA 50 in last 3 ticks
- RSI < 50
- ADX > 20 OR (ADX < 20 AND RSI > 60)

### Exit Rules
- Take Profit: ATR × 3.0 (min 0.3%)
- Stop Loss: ATR × 1.5 (min 0.2%)
- Trailing Stop: Activates at ATR × 1.5 profit, trails by ATR × 0.5
- Time Stop: 60 seconds

### Risk Management
- Risk per trade: 1% (2% for high-probability setups)
- Daily loss limit: 5%
- Max trades per day: 40
- Circuit breaker: 15-minute pause after 3 consecutive losses
- Volatility filter: Pause if ATR > 2× historical average

## Backtesting

```bash
python -m src.backtest --start 2024-01-01 --end 2024-12-31
```

## Disclaimer

This bot is for educational purposes. Trading synthetic indices involves significant risk. Never trade with money you cannot afford to lose. Always test thoroughly in demo mode before live trading.

## License

MIT