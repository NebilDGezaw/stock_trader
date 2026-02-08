"""
Configuration for the Stock Trader application.
Contains default parameters for ICT/Smart Money Concept strategies.
"""

# ──────────────────────────────────────────────
#  Data Fetching
# ──────────────────────────────────────────────
DEFAULT_TICKER = "SPY"
DEFAULT_PERIOD = "6mo"          # yfinance period string
DEFAULT_INTERVAL = "1d"         # candle interval: 1m,5m,15m,1h,1d
FALLBACK_TICKERS = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]

# ──────────────────────────────────────────────
#  Asset Class Presets
# ──────────────────────────────────────────────
ASSET_CLASSES = {
    "Stocks": {
        "default_ticker": "SPY",
        "presets": {
            "US Large Cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B"],
            "US Tech":      ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "CRM", "ADBE"],
            "US Indices":   ["SPY", "QQQ", "DIA", "IWM"],
            "Sectors":      ["XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY"],
        },
        "currency_symbol": "$",
        "unit": "shares",
    },
    "Crypto": {
        "default_ticker": "BTC-USD",
        "presets": {
            "Major":       ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"],
            "DeFi":        ["UNI-USD", "AAVE-USD", "MKR-USD", "LINK-USD", "SNX-USD"],
            "Layer 2":     ["MATIC-USD", "ARB-USD", "OP-USD", "IMX-USD"],
            "Meme / Alt":  ["DOGE-USD", "SHIB-USD", "PEPE-USD", "AVAX-USD", "ADA-USD"],
        },
        "currency_symbol": "$",
        "unit": "units",
    },
    "Forex": {
        "default_ticker": "EURUSD=X",
        "presets": {
            "Majors":    ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X"],
            "Crosses":   ["EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDNZD=X", "EURCHF=X", "CADJPY=X"],
            "Exotics":   ["USDZAR=X", "USDMXN=X", "USDTRY=X", "USDSEK=X", "USDNOK=X"],
        },
        "currency_symbol": "",
        "unit": "lots",
    },
    "Commodities": {
        "default_ticker": "GC=F",
        "presets": {
            "Metals":      ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F"],
            "Energy":      ["CL=F", "NG=F", "BZ=F", "HO=F", "RB=F"],
            "Agriculture": ["ZC=F", "ZW=F", "ZS=F", "KC=F", "CC=F", "SB=F", "CT=F"],
        },
        "currency_symbol": "$",
        "unit": "contracts",
    },
}

# ──────────────────────────────────────────────
#  Market Structure
# ──────────────────────────────────────────────
SWING_LOOKBACK = 5              # bars to look left/right for swing detection
STRUCTURE_BREAK_CONFIRM = 1     # candles needed to confirm BOS/CHoCH

# ──────────────────────────────────────────────
#  Order Blocks
# ──────────────────────────────────────────────
OB_LOOKBACK = 20                # bars to search for order blocks
OB_BODY_RATIO_MIN = 0.5        # minimum body-to-range ratio for strong candle
OB_MAX_AGE = 50                 # max candles since OB formed to still be valid
OB_MITIGATION_TOUCH = True      # OB invalidated once price returns

# ──────────────────────────────────────────────
#  Fair Value Gaps
# ──────────────────────────────────────────────
FVG_MIN_GAP_PERCENT = 0.1      # minimum gap as % of price to qualify
FVG_MAX_AGE = 30                # max candles since FVG formed

# ──────────────────────────────────────────────
#  Liquidity
# ──────────────────────────────────────────────
LIQ_EQUAL_HIGHS_TOLERANCE = 0.001   # % tolerance for equal highs/lows
LIQ_SWEEP_WICK_MIN = 0.3            # minimum wick-to-range ratio for sweep
LIQ_LOOKBACK = 30                    # bars to find liquidity pools

# ──────────────────────────────────────────────
#  Premium / Discount Zones
# ──────────────────────────────────────────────
PREMIUM_THRESHOLD = 0.5         # above 50% of range = premium
DISCOUNT_THRESHOLD = 0.5        # below 50% of range = discount

# ──────────────────────────────────────────────
#  Kill Zones (UTC hours)
# ──────────────────────────────────────────────
KILL_ZONES = {
    "asian":   (0, 6),     # 00:00 – 06:00 UTC
    "london":  (7, 10),    # 07:00 – 10:00 UTC
    "new_york": (12, 15),  # 12:00 – 15:00 UTC
}

# ──────────────────────────────────────────────
#  Risk Management
# ──────────────────────────────────────────────
RISK_PER_TRADE = 0.02           # 2% of capital per trade
RISK_REWARD_MIN = 2.0           # minimum R:R to take a trade
INITIAL_CAPITAL = 100_000       # starting capital ($)
MAX_OPEN_POSITIONS = 3          # maximum concurrent positions

# ──────────────────────────────────────────────
#  Signal Scoring
# ──────────────────────────────────────────────
SIGNAL_SCORE_THRESHOLDS = {
    "strong_buy": 7,
    "buy": 5,
    "neutral": 3,
    "sell": 5,
    "strong_sell": 7,
}
