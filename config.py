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
RISK_PER_TRADE = 0.01           # 1% of capital per trade (was 2% — caused $24K loss)
RISK_REWARD_MIN = 2.0           # minimum R:R to take a trade
INITIAL_CAPITAL = 100_000       # starting capital ($)
MAX_OPEN_POSITIONS = 9          # absolute maximum concurrent positions (fallback)
MAX_DAILY_LOSS_PCT = 3.0        # halt trading after 3% daily drawdown (was 5%)

# ── Per-asset-class position limits ──────────
# Diversify: max 2 per category to limit correlated exposure
MAX_POSITIONS_PER_CLASS = {
    "forex": 2,          # was 3 — correlated pairs compound losses
    "crypto": 2,         # was 3
    "commodity": 2,      # was 3 — metals + energy
    "stock": 3,          # keep 3 for Alpaca stocks
}

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

# ──────────────────────────────────────────────
#  Leveraged ETF Mode (Momentum / Mean-Reversion)
# ──────────────────────────────────────────────
LEVERAGED_MODE = {
    # RSI — widened zones to catch more entries on volatile products
    "rsi_period": 14,
    "rsi_oversold": 35,             # was 30
    "rsi_overbought": 65,           # was 70
    "rsi_extreme_oversold": 25,     # was 20
    "rsi_extreme_overbought": 75,   # was 80

    # Bollinger Bands
    "bb_period": 20,
    "bb_std": 2.0,
    "bb_extreme_std": 1.0,          # NEW: extra signal when price > 1 std beyond band
    "bb_extreme_rsi_bull": 40,       # RSI < this + below lower BB = strong bounce
    "bb_extreme_rsi_bear": 60,       # RSI > this + above upper BB = strong reversal

    # EMA — dual pair for faster signals
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_ultra_fast": 5,            # NEW: 5/13 catches momentum earlier
    "ema_ultra_slow": 13,

    # ATR stops — tighter TP for more frequent wins, trail for runners
    "atr_period": 14,
    "atr_sl_multiplier": 1.0,
    "atr_tp_multiplier": 2.0,      # keep at 2.0 — 1.5 was too tight
    "trailing_stop": False,         # Disabled — hurts more than helps with tight TP
    "trail_activation_atr": 1.2,    # move SL to breakeven after 1.2x ATR
    "trail_distance_atr": 0.75,     # then trail at 0.75x ATR behind price

    # Volume
    "volume_lookback": 20,
    "volume_multiplier": 1.2,       # was 1.3 — slightly more permissive

    # Mean-reversion on big moves — NEW
    "mean_reversion": True,
    "big_move_threshold": 0.05,     # 5% single-day move triggers bounce signal
    "big_move_score": 2,

    # Scoring — much lower thresholds for more trades
    "score_thresholds": {
        "strong_buy": 4,            # was 5
        "buy": 2,                   # was 3
        "neutral": 1,               # was 2
        "sell": 2,                  # was 3
        "strong_sell": 4,           # was 5
    },

    # Risk — more aggressive for leveraged products
    "risk_per_trade": 0.03,         # 3% per trade (was 2%)
}

LEVERAGED_TICKERS = ["MSTU", "MSTR", "MSTZ", "TSLL", "TQQQ", "SOXL", "FNGU"]

# ──────────────────────────────────────────────
#  Crypto Momentum Mode
# ──────────────────────────────────────────────
CRYPTO_MODE = {
    "ema_fast": 9,
    "ema_mid": 21,
    "ema_slow": 50,
    "rsi_period": 14,
    "rsi_oversold": 40,
    "rsi_overbought": 60,
    "rsi_extreme_oversold": 25,
    "rsi_extreme_overbought": 75,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.0,
    "volume_lookback": 20,
    "volume_spike_multiplier": 2.0,
    "atr_period": 14,
    "atr_sl_multiplier": 1.5,
    "atr_tp_multiplier": 3.0,
    "trailing_stop": True,
    "trail_activation_atr": 1.5,
    "trail_distance_atr": 1.0,
    "exhaustion_consecutive_days": 5,
    "score_thresholds": {
        "strong_buy": 5,
        "buy": 3,
        "neutral": 2,
        "sell": 3,
        "strong_sell": 5,
    },
    "risk_per_trade": 0.01,         # was 0.02 — reduced to prevent over-leverage
}

CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "DOGE-USD", "ADA-USD"]

# ──────────────────────────────────────────────
#  Forex ICT Mode (1h candles, kill-zone driven)
# ──────────────────────────────────────────────
FOREX_MODE = {
    "swing_lookback": 3,
    "fvg_min_gap_pct": 0.02,
    "ob_proximity": 0.005,
    "atr_period": 14,
    "atr_sl_multiplier": 1.0,
    "atr_tp_multiplier": 2.0,
    "kill_zone_multiplier": 2.0,
    "off_session_multiplier": 0.5,
    "asian_penalty": 1,
    "overlap_bonus": 1,
    "score_thresholds": {
        "strong_buy": 6,
        "buy": 4,
        "neutral": 2,
        "sell": 4,
        "strong_sell": 6,
    },
    "risk_per_trade": 0.01,         # was 0.02 — reduced to prevent over-leverage
}
