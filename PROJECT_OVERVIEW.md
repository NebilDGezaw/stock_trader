# Stock Trader — Full Project Overview

## What This Project Is

An automated trading system built in Python that trades **stocks, forex, crypto, and commodities** using ICT (Inner Circle Trader) and Smart Money Concepts (SMC) strategies. The system is **halal-compliant** (no short selling, no interest-based transactions). It runs fully automated via GitHub Actions, sends notifications via Telegram, and has a Streamlit web UI for manual analysis.

**Code location:** `/Users/givenfamily/Documents/stock_trader/`  
**GitHub repo:** `https://github.com/NebilDGezaw/stock_trader`

---

## Two Separate Trading Systems

### 1. Alpaca — Stocks (US Market)
- **Platform:** Alpaca Markets (paper trading, transitioning to live)
- **Account type:** Paper account (~$107K equity as of Feb 13, 2026)
- **What it trades:** 28 US stocks across 6 categories + 7 leveraged ETFs
- **Halal compliance:** Long-only. No short selling. SELL signals are skipped entirely. The workaround for profiting from downtrends is using inverse ETFs (e.g., MSTZ is 2x short MSTR — buying MSTZ is halal because you're buying an asset, not short selling).

### 2. HFM (HF Markets) via MetaTrader 5 — Forex, Crypto, Commodities
- **Platform:** HFM demo account ($100K → currently ~$76K after initial losses)
- **Connection:** MT5 API running on Windows GitHub Actions runners
- **What it trades:** EUR/USD, GBP/USD, EUR/GBP, GBP/JPY, USD/CHF, USD/CAD, USD/JPY (forex); BTC, ETH, SOL, XRP (crypto); Gold (XAUUSD), Silver (XAGUSD) (commodities)

---

## Strategy Overview

### Regular Stocks (SMCStrategy)
- ICT/Smart Money Concepts: market structure breaks, order blocks, fair value gaps, liquidity sweeps, premium/discount zones
- ATR-based SL/TP (1.5x ATR for SL, 3.0x ATR for TP → natural 1:2 R:R)
- Trend momentum bonus: +2 score when aligned with 20-day SMA
- Fakeout filters with relaxed thresholds for stock candles
- Score thresholds: buy >= 3, strong_buy >= 5

### Leveraged ETFs (LeveragedMomentumStrategy)
- RSI with widened zones (oversold <= 35, overbought >= 65, extremes at 25/75)
- Dual Bollinger Bands (standard + extreme bands)
- Dual EMA crossovers (9/21 standard + 5/13 ultra-fast)
- Volume momentum (>= 1.2x average, extra signal at 2x)
- Mean-reversion on 5%+ single-day moves
- BB squeeze breakout detection
- Score thresholds: buy >= 3, strong_buy >= 4
- 3% risk per trade (more aggressive)
- Trailing stops enabled (1.0x ATR activation, 0.75x ATR trail distance)

### Forex (ForexICTStrategy)
- ICT-specific: order blocks, fair value gaps, liquidity sweeps, kill zones
- Session-based analysis (London, NY, Overlap)
- Score thresholds: buy >= 3, strong_buy >= 5
- 1.5% risk per trade

### Crypto & Commodities (CryptoMomentumStrategy)
- Momentum-based with RSI, Bollinger Bands, EMA crossovers
- Runs during Asian Crypto sessions (including weekends for crypto)
- 1.5% risk per trade

---

## Key Safety Features

### Halal Compliance (Alpaca)
- **No short selling** — only BUY signals are executed
- SELL/STRONG_SELL signals are logged but skipped
- Inverse ETFs (MSTZ) used as halal alternative to shorting MSTR
- MSTU/MSTZ mutual exclusion: never holds both simultaneously (auto-closes the opposite before buying the other)

### Market Regime Filter (Alpaca)
- Checks SPY vs 20-day SMA daily
- If SPY below SMA (bearish market): requires score >= 5 to enter any trade
- If SPY above SMA (bullish): normal thresholds apply
- Prevents buying weak signals in downtrends

### Position Rotation (Alpaca)
- When all 10 position slots are full and a strong signal comes in:
  - **Selling a profitable position** to make room: requires new signal score >= 5 (lower bar, profit is banked)
  - **Selling a losing position** to make room: requires new signal score >= 8 (very high bar, only exceptional opportunities)
  - Prefers selling the position with the highest profit % (most of the move is done)

### Risk Management (Both Platforms)
- **Alpaca:** 1.5% risk per trade (regular), 3% (leveraged), max 10 positions, 3% daily loss limit
- **HFM:** 1.5% risk per trade, max 2 positions per asset class (forex: 2, crypto: 2, commodity: 2), 4% daily loss limit
- ATR-based stop losses on all trades
- Bracket orders with SL/TP on every Alpaca trade
- Protective stop orders auto-placed on any position found without a SL

### Position Management (HFM)
- Max loss circuit-breaker: closes positions exceeding -2R
- 48-hour timeout: closes positions open longer than 2 days
- HOLD signal closure: closes losers when strategy signal goes neutral (no longer confirmed)
- Trailing stops when R >= 1.0
- Partial close (50%) at 1R profit

### Portfolio Allocation Caps (Alpaca)
| Category | Cap | Tickers |
|----------|-----|---------|
| Leveraged ETFs | 30% | MSTU, MSTR, MSTZ, TSLL, TQQQ, SOXL, FNGU |
| Mega Tech | 25% | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD |
| Semiconductors | 15% | AVGO, QCOM, ASML, MU |
| Healthcare | 15% | UNH, ABBV, LLY, ISRG |
| Clean Energy/EV | 10% | ENPH, FSLR, RIVN, NIO |
| Consumer | 5% | COST, TGT |

- NIO and RIVN capped at 1% each (speculative)
- Default single-ticker cap: 50% of category cap

---

## Automation (GitHub Actions)

### Alpaca Workflow (`.github/workflows/alpaca_trading.yml`)
- **Leveraged ETF entries:** Hourly during market hours (9:30 AM - 3:30 PM ET, Mon-Fri)
- **Regular stock entries:** Every 2 hours, staggered by category
- **Monitor:** Hourly (10 AM - 3 PM ET) — manages positions, places missing SLs, trails stops
- **EOD Summary:** 4:15 PM ET daily

### HFM Workflow (`.github/workflows/hfm_trading.yml`)
- **London entry:** 8:00 UTC weekdays (forex + metals)
- **Overlap entry:** 13:00 UTC weekdays
- **NY entry:** 14:30 UTC weekdays
- **Asian Crypto entry:** Multiple times including weekends
- **Monitor:** Hourly during active hours + 3x on weekends
- **EOD Summary:** 20:30 UTC weekdays
- **Cleanup mode:** Manual trigger to force-close all positions

### Required GitHub Secrets
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER` (true/false)
- `HFM_MT5_LOGIN`, `HFM_MT5_PASSWORD`, `HFM_MT5_SERVER`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (private/stocks), `TELEGRAM_GROUP_CHAT_ID` (group/forex+crypto)

---

## Telegram Notifications

### Two channels:
1. **Private bot (TELEGRAM_CHAT_ID):** Stock alerts only (Alpaca trades). User's personal channel.
2. **Group channel (TELEGRAM_GROUP_CHAT_ID):** Forex, crypto, and commodity alerts (HFM trades). Shared with other users.

### What gets sent:
- Entry notifications (executed trades with entry/SL/TP/shares)
- Skip reasons (halal compliance, bearish filter, position cap)
- Monitor updates (trailing stops, position closures)
- Daily summaries (balance, equity, open positions, PnL)

---

## Streamlit UI (`ui/dashboard.py`)

- Web-based dashboard at `https://stocktrader-qvglsnedggfyh72mvws788.streamlit.app/` (or run locally: `streamlit run ui/dashboard.py`)
- Features: interactive candlestick charts with SMC overlays, signal cards, score gauges, multi-ticker scanning
- Supports stocks, forex, crypto, commodities with appropriate strategies for each
- Backtesting engine for historical date ranges
- Mobile-responsive design
- Scanner results are clickable for detailed individual analysis

---

## File Structure

```
stock_trader/
├── config.py                    # All strategy parameters, thresholds, risk settings
├── data/fetcher.py              # yfinance data fetching
├── models/signals.py            # TradeSetup, TradeAction, Signal data models
├── strategies/
│   ├── smc_strategy.py          # ICT/SMC strategy (regular stocks)
│   ├── leveraged_momentum.py    # Momentum/mean-reversion (leveraged ETFs)
│   ├── forex_ict.py             # Forex ICT strategy
│   └── crypto_momentum.py       # Crypto/commodity momentum strategy
├── trading/
│   ├── alpaca_client.py         # Alpaca API wrapper
│   ├── alpaca_executor.py       # Stock trade execution (rotation, regime filter, halal)
│   ├── alpaca_position_manager.py # Stock position management (trail, SL protection)
│   ├── run_alpaca.py            # Alpaca entry point for GitHub Actions
│   ├── mt5_client.py            # MetaTrader 5 API wrapper
│   ├── executor.py              # HFM trade execution
│   ├── position_manager.py      # HFM position management (timeout, max-loss)
│   ├── run_trading.py           # HFM entry point for GitHub Actions
│   └── symbols.py               # Yahoo Finance ↔ MT5 symbol mapping
├── bt_engine/engine.py          # Backtesting engine + strategy router
├── ui/dashboard.py              # Streamlit web UI
├── alerts/telegram_bot.py       # Telegram alert system
├── .github/workflows/
│   ├── alpaca_trading.yml       # Stock automation schedule
│   └── hfm_trading.yml          # Forex/crypto automation schedule
└── requirements.txt
```

---

## Current State & Known Issues (as of Feb 13, 2026)

### Working:
- Alpaca stock trading is active (9 positions, ~$107K equity)
- Market regime filter correctly identifying bearish conditions
- Leveraged buy threshold raised from 2 to 3
- Position rotation logic implemented
- MSTU/MSTZ inverse pair handling
- HFM scheduled workflows running

### Recently Fixed:
- All 9 Alpaca positions had NO stop loss (SL=$0.00) — protective SL auto-placement added
- Bracket order child orders now properly cancelled during rotation
- Telegram 400 errors from message length
- Reversal detection interval mismatch for leveraged ETFs

### Monitoring:
- HFM demo stuck at ~$76K — stuck losing positions were blocking new entries. Cleanup mode added + position timeout/max-loss logic. Need to run cleanup to clear old positions.
- Alpaca paper account performance needs 1+ week of data after halal/regime filter changes to evaluate properly

### Future Plans:
- Transition Alpaca from paper to live trading once strategy is validated
- Transition HFM from demo to live (real money, with per-trade $ limits)
- Bi-weekly portfolio review and stock rotation
