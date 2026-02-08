# Stock Trader — ICT & Smart Money Concepts

A Python-based stock trading analysis tool that uses **ICT (Inner Circle Trader)** and **Smart Money Concepts (SMC)** to generate buy/sell signals.

## Features

| Module | What it does |
|---|---|
| **Market Structure** | Detects swing highs/lows, Break of Structure (BOS), and Change of Character (CHoCH) |
| **Order Blocks** | Identifies bullish and bearish institutional order blocks |
| **Fair Value Gaps** | Finds price imbalances (FVGs) that may act as magnets |
| **Liquidity Analysis** | Locates equal highs/lows and detects liquidity sweeps |
| **SMC Strategy Engine** | Combines all signals into a weighted score with buy/sell decisions |
| **Decision Engine** | Risk-managed position sizing and trade management |

## Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run analysis on a ticker
python main.py --ticker AAPL

# 4. Scan multiple tickers
python main.py --scan AAPL MSFT TSLA GOOGL AMZN

# 5. Change timeframe
python main.py --ticker SPY --interval 1h --period 1mo
```

## Project Structure

```
stock_trader/
├── main.py                  # CLI entry point
├── config.py                # Strategy parameters
├── requirements.txt         # Python dependencies
├── data/
│   └── fetcher.py           # Stock data API (yfinance)
├── strategies/
│   ├── market_structure.py  # BOS & CHoCH detection
│   ├── order_blocks.py      # Order block identification
│   ├── fair_value_gaps.py   # FVG detection
│   ├── liquidity.py         # Liquidity pools & sweeps
│   └── smc_strategy.py      # Combined ICT/SMC engine
├── models/
│   └── signals.py           # Signal & trade data models
├── trader/
│   └── decision_engine.py   # Trade decisions & risk mgmt
└── utils/
    └── helpers.py           # Shared utilities
```

## Strategy Overview

### ICT / Smart Money Concepts

1. **Market Structure Analysis** — Identify the prevailing trend via swing highs and lows. A *Break of Structure* confirms trend continuation; a *Change of Character* signals a potential reversal.

2. **Order Blocks** — Large institutional candles before a significant move. These zones often act as supply/demand areas where price reacts.

3. **Fair Value Gaps** — Three-candle patterns where the wicks of candle 1 and candle 3 don't overlap, leaving a "gap" that price tends to revisit.

4. **Liquidity Sweeps** — Equal highs/lows form liquidity pools. When price sweeps past them and reverses, it signals smart money accumulation or distribution.

5. **Premium / Discount Zones** — Trades in the discount zone (below equilibrium) favor longs; trades in the premium zone (above equilibrium) favor shorts.

## Disclaimer

This software is for **educational and research purposes only**. It does not constitute financial advice. Trading stocks involves risk of loss. Always do your own research and consult a licensed financial advisor before trading.
