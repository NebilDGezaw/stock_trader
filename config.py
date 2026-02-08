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
#  Fakeout Detection
# ──────────────────────────────────────────────
FAKEOUT_DISPLACEMENT_MIN_BODY = 0.55   # min body-to-range ratio for displacement
FAKEOUT_VOLUME_LOOKBACK = 20           # bars to compute average volume
FAKEOUT_VOLUME_MULTIPLIER = 1.2        # break candle vol must be ≥ 1.2× avg
FAKEOUT_HOLD_CANDLES = 2               # how many candles must close beyond level
FAKEOUT_SWEEP_REVERSAL_BODY = 0.50     # min body ratio for reversal after sweep
FAKEOUT_PENALTY_NO_DISPLACEMENT = 1    # score deduction if no displacement
FAKEOUT_PENALTY_NO_VOLUME = 1          # score deduction if low volume
FAKEOUT_PENALTY_ISOLATED = 1           # deduction for signals without confluence

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

# ──────────────────────────────────────────────
#  Stock Mode (medium-risk overrides)
# ──────────────────────────────────────────────
# Daily stock candles have wider wicks from gaps/pre-market than forex.
# Leveraged ETFs (MSTU, MSTR, TSLL) are even more volatile.
# These overrides relax the fakeout filters and scoring for stocks.
STOCK_MODE = {
    # Relaxed fakeout thresholds — stock candles naturally have more wicks
    "fakeout_displacement_min_body": 0.38,   # was 0.55 — stocks gap more
    "fakeout_volume_multiplier": 1.0,        # was 1.2 — just above avg is fine
    "fakeout_sweep_reversal_body": 0.35,     # was 0.50 — sweeps are choppier

    # Lower score thresholds — so more signals trigger buy/sell
    "score_thresholds": {
        "strong_buy": 5,    # was 7
        "buy": 3,           # was 5
        "neutral": 2,       # was 3
        "sell": 3,          # was 5
        "strong_sell": 5,   # was 7
    },

    # ATR-based SL/TP — much better R:R than swing-point SL/TP for stocks
    "use_atr_sl_tp": True,
    "atr_period": 14,
    "atr_sl_multiplier": 1.5,    # SL = 1.5 × ATR below/above entry
    "atr_tp_multiplier": 3.0,    # TP = 3.0 × ATR — targets 1:2 R:R naturally

    # Trend momentum bonus — align with the macro trend
    "trend_sma_period": 20,      # 20-day SMA
    "trend_bonus_score": 2,      # +2 to aligned signals

    # Skip kill-zone penalty for stocks (daily candles aren't time-stamped intraday)
    "skip_killzone_penalty": True,

    # Risk
    "min_risk_reward": 1.5,      # was 2.0 — medium risk tolerance
}
