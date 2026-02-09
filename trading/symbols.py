"""
Symbol Mapping — yfinance ticker → HFM MT5 symbol.
====================================================
HFM MT5 uses different naming conventions than yfinance.
This module provides bidirectional mapping and verification.

NOTE: Symbol names can vary by HFM account type (Cent, Zero, Pro).
      Run `verify_symbols()` on first connection to confirm availability.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  yfinance → HFM MT5 symbol mapping
# ──────────────────────────────────────────────────────────

YFINANCE_TO_MT5 = {
    # ── Forex Majors ──────────────────────────────
    "EURUSD=X":  "EURUSD",
    "GBPUSD=X":  "GBPUSD",
    "USDJPY=X":  "USDJPY",
    "USDCHF=X":  "USDCHF",
    "AUDUSD=X":  "AUDUSD",
    "USDCAD=X":  "USDCAD",
    "NZDUSD=X":  "NZDUSD",

    # ── Forex Crosses ─────────────────────────────
    "EURGBP=X":  "EURGBP",
    "EURJPY=X":  "EURJPY",
    "GBPJPY=X":  "GBPJPY",
    "AUDNZD=X":  "AUDNZD",
    "EURCHF=X":  "EURCHF",
    "CADJPY=X":  "CADJPY",

    # ── Metals ────────────────────────────────────
    "GC=F":      "XAUUSD",     # Gold
    "SI=F":      "XAGUSD",     # Silver
    "HG=F":      "COPPER",     # Copper (may vary)
    "PL=F":      "XPTUSD",     # Platinum
    "PA=F":      "XPDUSD",     # Palladium

    # ── Energy ────────────────────────────────────
    "CL=F":      "USOIL",      # WTI Crude Oil
    "BZ=F":      "UKOIL",      # Brent Crude Oil
    "NG=F":      "NGAS",       # Natural Gas

    # ── Crypto (HFM uses # prefix) ─────────────────
    "BTC-USD":   "#BTCUSD",
    "ETH-USD":   "#ETHUSD",
    "SOL-USD":   "#SOLUSD",
    "XRP-USD":   "#XRPUSD",
    "BNB-USD":   "#BNBUSD",
    "DOGE-USD":  "#DOGEUSD",
    "ADA-USD":   "#ADAUSD",
    "AVAX-USD":  "#AVAXUSD",
    "MATIC-USD": "#MATICUSD",
    "LINK-USD":  "#LINKUSD",
    "DOT-USD":   "#DOTUSD",
    "LTC-USD":   "#LTCUSD",
}

# Reverse mapping: MT5 → yfinance
MT5_TO_YFINANCE = {v: k for k, v in YFINANCE_TO_MT5.items()}


# ──────────────────────────────────────────────────────────
#  Suffix variants — some HFM account types add suffixes
# ──────────────────────────────────────────────────────────

# HFM may use suffixes like ".a" (cent), ".b" for different account types,
# or "#" prefix for crypto. We try the base symbol first, then variants.
_SUFFIXES = ["", ".a", ".b", "m", "_SB"]
_PREFIXES = ["", "#"]


# ──────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────

def to_mt5(yfinance_ticker: str) -> str:
    """
    Convert a yfinance ticker to the HFM MT5 symbol name.

    Returns the mapped symbol or a best-guess conversion if not in the table.
    """
    t = yfinance_ticker.strip().upper()

    # Direct lookup
    if t in YFINANCE_TO_MT5:
        return YFINANCE_TO_MT5[t]

    # Fallback: strip yfinance suffixes and guess
    # e.g., "EURUSD=X" → "EURUSD", "BTC-USD" → "BTCUSD"
    if t.endswith("=X"):
        return t.replace("=X", "")
    if t.endswith("=F"):
        return t.replace("=F", "")
    if "-USD" in t:
        return t.replace("-USD", "USD")

    # No conversion needed (might already be MT5 format)
    return t


def to_yfinance(mt5_symbol: str) -> str:
    """Convert an MT5 symbol back to yfinance ticker."""
    s = mt5_symbol.strip().upper()
    if s in MT5_TO_YFINANCE:
        return MT5_TO_YFINANCE[s]
    # Can't reverse — return as-is
    return s


def get_asset_type(yfinance_ticker: str) -> str:
    """Detect asset type from yfinance ticker for lot sizing."""
    t = yfinance_ticker.strip().upper()
    if "=X" in t:
        return "forex"
    if "=F" in t:
        # Distinguish metals from energy/agri
        if t in ("GC=F", "SI=F", "HG=F", "PL=F", "PA=F"):
            return "metal"
        return "commodity"
    if t.endswith("-USD"):
        return "crypto"
    return "stock"


def verify_symbols(mt5_client, yfinance_tickers: list[str]) -> dict:
    """
    Verify which yfinance tickers have valid MT5 symbols on the broker.

    Returns dict: {yfinance_ticker: {"mt5_symbol": str, "available": bool}}
    """
    results = {}

    # Get all available symbols from broker
    all_broker_symbols = set(mt5_client.list_symbols())
    logger.info(f"Broker has {len(all_broker_symbols)} symbols available")

    for ticker in yfinance_tickers:
        mt5_sym = to_mt5(ticker)
        found = False

        # Try base symbol first
        if mt5_sym in all_broker_symbols:
            found = True
        else:
            # Try with common prefixes and suffixes
            # Strip existing # prefix for clean base name
            base = mt5_sym.lstrip("#")
            for prefix in _PREFIXES:
                for suffix in _SUFFIXES:
                    candidate = prefix + base + suffix
                    if candidate in all_broker_symbols:
                        mt5_sym = candidate
                        found = True
                        break
                if found:
                    break

        results[ticker] = {
            "mt5_symbol": mt5_sym,
            "available": found,
        }

        if found:
            logger.info(f"  ✓ {ticker} → {mt5_sym}")
        else:
            logger.warning(f"  ✗ {ticker} → {mt5_sym} (NOT FOUND on broker)")

    return results
