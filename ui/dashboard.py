"""
Stock Trader Dashboard — Streamlit UI
======================================
Interactive web dashboard for ICT / Smart Money Concepts analysis.

Run with:  streamlit run ui/dashboard.py
"""

import sys
import os

# Ensure the project root is on the path so our modules resolve.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Force-reload config to avoid stale cached module
for _mod in list(sys.modules.keys()):
    if any(_mod.startswith(p) for p in ["config", "strategies", "utils", "models", "data", "trader", "bt_engine", "backtesting", "alerts"]):
        del sys.modules[_mod]

import config

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta

from data.fetcher import StockDataFetcher
from strategies.smc_strategy import SMCStrategy
from strategies.leveraged_momentum import LeveragedMomentumStrategy
from strategies.crypto_momentum import CryptoMomentumStrategy
from strategies.forex_ict import ForexICTStrategy
from models.signals import TradeAction, MarketBias, SignalType
from ui.charts import (
    build_main_chart,
    build_score_gauge,
    build_signal_breakdown,
)
from bt_engine.engine import BacktestEngine, run_multi

# ──────────────────────────────────────────────────────────
#  Page Configuration
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Trader — ICT & SMC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ═══════════ Base / Desktop ═══════════ */
.stApp { background: #0a0e17; }

section[data-testid="stSidebar"] {
    background: #0f1420;
    border-right: 1px solid #1e293b;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #131a2b 0%, #0f1420 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.action-badge {
    display: inline-block; padding: 8px 24px; border-radius: 8px;
    font-weight: 700; font-size: 1.3rem; text-align: center; margin: 4px 0;
}
.action-strong-buy  { background: linear-gradient(135deg, #059669, #10b981); color: #fff; }
.action-buy         { background: linear-gradient(135deg, #16a34a, #22c55e); color: #fff; }
.action-hold        { background: linear-gradient(135deg, #ca8a04, #eab308); color: #1a1a1a; }
.action-sell        { background: linear-gradient(135deg, #dc2626, #ef4444); color: #fff; }
.action-strong-sell { background: linear-gradient(135deg, #991b1b, #dc2626); color: #fff; }

.bias-badge {
    display: inline-block; padding: 4px 14px; border-radius: 6px;
    font-weight: 600; font-size: 0.85rem;
}
.bias-bullish { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.bias-bearish { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.bias-neutral { background: rgba(234,179,8,0.15); color: #fbbf24; border: 1px solid rgba(234,179,8,0.3); }

.signal-card {
    background: #131a2b; border: 1px solid #1e293b; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px; display: flex;
    align-items: center; gap: 12px;
}
.signal-icon {
    width: 36px; height: 36px; border-radius: 8px; display: flex;
    align-items: center; justify-content: center; font-size: 1.1rem; flex-shrink: 0;
}
.signal-icon.bullish { background: rgba(34,197,94,0.15); color: #4ade80; }
.signal-icon.bearish { background: rgba(239,68,68,0.15); color: #f87171; }
.signal-icon.neutral { background: rgba(234,179,8,0.15); color: #fbbf24; }
.signal-body { flex: 1; min-width: 0; }
.signal-type { font-size: 0.78rem; font-weight: 600; color: #cbd5e1; text-transform: uppercase; }
.signal-detail { font-size: 0.75rem; color: #64748b; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; }
.signal-score { font-size: 0.85rem; font-weight: 700; color: #e2e8f0; background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 6px; }

.info-box {
    background: linear-gradient(135deg, #131a2b 0%, #0f1420 100%);
    border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 16px;
}
.info-box h4 { margin: 0 0 8px 0; color: #e2e8f0; font-size: 0.9rem; }
.info-row {
    display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid #1e293b; font-size: 0.82rem;
}
.info-row:last-child { border-bottom: none; }
.info-label { color: #94a3b8; }
.info-value { color: #e2e8f0; font-weight: 500; }

.scanner-row {
    background: #131a2b; border: 1px solid #1e293b; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 8px; display: grid;
    grid-template-columns: 80px 120px 100px 80px 1fr; align-items: center; gap: 12px;
}
.scanner-ticker { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; }
.scanner-action { font-size: 0.8rem; font-weight: 600; padding: 4px 10px; border-radius: 6px; text-align: center; }
.scanner-score { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; text-align: center; }
.scanner-details { font-size: 0.78rem; color: #94a3b8; }

/* Page headers */
.page-header {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 8px 16px;
    margin-bottom: 20px;
}
.page-header-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #e2e8f0;
    line-height: 1.2;
    word-break: break-word;
}
.page-header-meta {
    color: #64748b;
    font-size: 0.85rem;
    white-space: nowrap;
}
.page-header-tag {
    font-size: 0.65rem;
    color: #a78bfa;
    background: rgba(167,139,250,0.15);
    padding: 2px 8px;
    border-radius: 4px;
    white-space: nowrap;
}

/* Ticker header (Search Ticker mode) */
.ticker-header {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 8px 12px;
    margin-bottom: 4px;
}
.ticker-symbol { font-size: 2rem; font-weight: 800; color: #e2e8f0; }
.ticker-price { font-size: 1.8rem; font-weight: 700; color: #e2e8f0; }
.ticker-change { font-size: 1rem; font-weight: 600; }
.ticker-badge {
    font-size: 0.75rem; color: #64748b; background: #1e293b;
    padding: 3px 10px; border-radius: 5px; white-space: nowrap;
}
.ticker-info-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}
.ticker-info-row span { white-space: nowrap; }

/* Landing page */
.landing-box {
    text-align: center;
    padding: 60px 20px;
    color: #64748b;
}
.landing-icon { font-size: 3rem; margin-bottom: 16px; }
.landing-title { font-size: 1.2rem; font-weight: 600; color: #94a3b8; }
.landing-sub { font-size: 0.85rem; margin-top: 8px; }

/* ═══════════ Mobile Responsive ═══════════ */

/* Tablets (≤ 1024px) */
@media (max-width: 1024px) {
    .scanner-row {
        grid-template-columns: 70px 100px 90px 60px 1fr;
        padding: 10px 12px;
        gap: 8px;
    }
    .scanner-ticker { font-size: 0.95rem; }
    .scanner-details { font-size: 0.72rem; }
    .page-header-title { font-size: 1.4rem !important; }
}

/* Phones (≤ 768px) — catches most phones */
@media (max-width: 768px) {
    /* Push content below Streamlit's sticky header bar (~3.5rem) */
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 3.5rem !important;
        max-width: 100% !important;
    }

    /* Hide the Fork / GitHub / menu bar on mobile — reclaim space */
    [data-testid="stHeader"] {
        background: #0a0e17 !important;
    }

    /* Header text: scale down */
    .stApp h1 { font-size: 1.2rem !important; }
    .stApp h2 { font-size: 1rem !important; }
    .stApp h3 { font-size: 0.9rem !important; }

    /* Page headers: wrap and shrink */
    .page-header {
        gap: 4px 8px;
        margin-bottom: 10px;
    }
    .page-header-title {
        font-size: 1.2rem !important;
        width: 100%;
        line-height: 1.3;
    }
    .page-header-meta {
        font-size: 0.72rem;
        white-space: normal;
        width: 100%;
    }
    .page-header-tag {
        font-size: 0.6rem;
    }

    /* Ticker header: wrap and shrink */
    .ticker-header { gap: 4px 8px; }
    .ticker-symbol { font-size: 1.35rem !important; }
    .ticker-price { font-size: 1.15rem !important; }
    .ticker-change { font-size: 0.8rem !important; }
    .ticker-badge { font-size: 0.65rem; padding: 2px 8px; }

    .ticker-info-row { gap: 6px; }
    .ticker-info-row span { font-size: 0.7rem !important; }

    /* Landing: compact */
    .landing-box { padding: 24px 12px; }
    .landing-icon { font-size: 2.2rem; margin-bottom: 10px; }
    .landing-title { font-size: 0.95rem; }
    .landing-sub { font-size: 0.75rem; line-height: 1.5; }

    /* Metric cards: compact */
    div[data-testid="stMetric"] {
        padding: 10px 12px;
        border-radius: 8px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.6rem !important;
        letter-spacing: 0;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 0.9rem !important;
    }

    /* Scanner rows: 2-col card layout on phones */
    .scanner-row {
        grid-template-columns: 1fr auto;
        grid-template-rows: auto auto auto;
        padding: 12px;
        gap: 4px 10px;
    }
    .scanner-ticker {
        font-size: 0.95rem;
        grid-column: 1;
        grid-row: 1;
    }
    .scanner-action {
        grid-column: 2;
        grid-row: 1;
        justify-self: end;
        font-size: 0.72rem;
    }
    .scanner-score {
        grid-column: 1;
        grid-row: 2;
        text-align: left;
        font-size: 0.8rem;
    }
    .scanner-details {
        grid-column: 1 / -1;
        grid-row: 3;
        font-size: 0.68rem;
        word-break: break-word;
        line-height: 1.4;
    }

    /* Action badge: smaller */
    .action-badge {
        padding: 5px 14px;
        font-size: 0.95rem;
    }

    /* Signal cards: tighter */
    .signal-card {
        padding: 10px 12px;
        gap: 8px;
    }
    .signal-icon {
        width: 28px; height: 28px;
        font-size: 0.85rem;
    }
    .signal-type { font-size: 0.68rem; }
    .signal-detail { font-size: 0.65rem; }
    .signal-score { font-size: 0.72rem; padding: 3px 8px; }

    /* Info boxes: tighter */
    .info-box { padding: 12px; border-radius: 8px; }
    .info-row { font-size: 0.72rem; padding: 4px 0; }
    .info-box h4 { font-size: 0.78rem; }

    /* Bias badge */
    .bias-badge { padding: 3px 10px; font-size: 0.7rem; }

    /* Tabs: smaller */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.75rem !important;
        padding: 6px 10px !important;
    }

    /* Tables: horizontal scroll */
    .stDataFrame { overflow-x: auto !important; }
    .stDataFrame table { font-size: 0.72rem !important; }

    /* Charts */
    .js-plotly-plot { min-height: 220px; }

    /* Expanders */
    details[data-testid="stExpander"] summary {
        font-size: 0.8rem !important;
    }

    /* Keep Streamlit columns side-by-side on mobile for buttons */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        gap: 0.5rem !important;
    }
}

/* Very small phones (≤ 400px) */
@media (max-width: 400px) {
    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        padding-top: 3rem !important;
    }
    .page-header-title { font-size: 1.05rem !important; }
    .page-header-meta { font-size: 0.65rem; }
    .ticker-symbol { font-size: 1.15rem !important; }
    .ticker-price { font-size: 0.95rem !important; }
    .ticker-change { font-size: 0.7rem !important; }

    .scanner-row {
        grid-template-columns: 1fr;
        grid-template-rows: auto;
        padding: 10px;
    }
    .scanner-ticker { grid-column: 1; }
    .scanner-action { grid-column: 1; justify-self: start; }
    .scanner-score { grid-column: 1; }
    .scanner-details { grid-column: 1; }

    div[data-testid="stMetric"] { padding: 8px 10px; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 0.8rem !important;
    }
    .landing-box { padding: 16px 8px; }
    .landing-icon { font-size: 1.8rem; }
    .landing-title { font-size: 0.85rem; }
    .landing-sub { font-size: 0.7rem; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  Constants & Default Watchlists
# ──────────────────────────────────────────────────────────

PERIODS_FOR_INTERVAL = {
    "1m":  ["1d", "5d"],
    "5m":  ["1d", "5d", "1mo", "60d"],
    "15m": ["1d", "5d", "1mo", "60d"],
    "30m": ["1d", "5d", "1mo", "60d"],
    "1h":  ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d":  ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    "1wk": ["3mo", "6mo", "1y", "2y", "5y"],
}
PERIOD_LABELS = {
    "1d": "1 Day", "5d": "5 Days", "7d": "7 Days", "1mo": "1 Month",
    "3mo": "3 Months", "6mo": "6 Months", "60d": "60 Days",
    "1y": "1 Year", "2y": "2 Years", "5y": "5 Years",
}
INTERVAL_LABELS = {
    "1m": "1 min", "5m": "5 min", "15m": "15 min", "30m": "30 min",
    "1h": "1 hour", "1d": "1 day", "1wk": "1 week",
}
AC_ICONS = {"Stocks": "🏛️", "Crypto": "₿", "Forex": "💱", "Commodities": "🛢️"}

# Default watchlists — same tickers the Telegram bot scans daily
# Organized by session to match the workflow
DAILY_SESSIONS = {
    "🏛️ Stocks — NY Open": {
        "tickers": ["MSTU", "MSTR", "MSTZ", "TSLL", "SPY", "AAPL", "MSFT", "TSLA", "GOOGL", "NVDA", "META"],
        "interval": "1d",
        "period": "6mo",
        "stock_mode": True,
        "icon": "🏛️",
    },
    "🇬🇧 London Session — Forex & Metals": {
        "tickers": ["EURUSD=X", "GBPUSD=X", "EURGBP=X", "GBPJPY=X", "USDCHF=X", "GC=F", "SI=F"],
        "interval": "1h",
        "period": "1mo",
        "stock_mode": False,
        "icon": "🇬🇧",
    },
    "🇺🇸 NY Session — Crypto, Metals & Forex": {
        "tickers": ["GC=F", "SI=F", "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "USDJPY=X", "USDCAD=X", "USDCHF=X", "GBPUSD=X"],
        "interval": "1h",
        "period": "1mo",
        "stock_mode": False,
        "icon": "🇺🇸",
    },
}


# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────

def action_class(a):
    return {"STRONG BUY": "action-strong-buy", "BUY": "action-buy",
            "HOLD": "action-hold", "SELL": "action-sell",
            "STRONG SELL": "action-strong-sell"}.get(a.value, "action-hold")

def bias_class(b):
    return {"bullish": "bias-bullish", "bearish": "bias-bearish",
            "neutral": "bias-neutral"}.get(b.value, "bias-neutral")

def sig_icon(sig):
    if sig.bias == MarketBias.BULLISH:
        return '<div class="signal-icon bullish">▲</div>'
    elif sig.bias == MarketBias.BEARISH:
        return '<div class="signal-icon bearish">▼</div>'
    return '<div class="signal-icon neutral">●</div>'

def fmt_price(price, sym="$"):
    if price >= 1000:    return f"{sym}{price:,.2f}"
    elif price >= 1:     return f"{sym}{price:.2f}"
    elif price >= 0.01:  return f"{sym}{price:.4f}"
    else:                return f"{sym}{price:.6f}"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker, period, interval):
    return StockDataFetcher(ticker).fetch(period=period, interval=interval)

def detect_asset_type(ticker: str) -> str:
    """Auto-detect asset type from ticker symbol."""
    t = ticker.upper()
    if t in config.LEVERAGED_TICKERS:
        return "leveraged"
    # Crypto: ends with -USD (but not forex pairs)
    crypto_tickers = [tk for presets in config.ASSET_CLASSES.get("Crypto", {}).get("presets", {}).values() for tk in presets]
    if t in crypto_tickers or (t.endswith("-USD") and "=" not in t):
        return "crypto"
    # Forex: contains =X
    if "=X" in t:
        return "forex"
    # Commodities: contains =F
    commodity_tickers = [tk for presets in config.ASSET_CLASSES.get("Commodities", {}).get("presets", {}).values() for tk in presets]
    if t in commodity_tickers or "=F" in t:
        return "commodity"
    return "stocks"

def run_analysis(df, ticker, stock_mode=False):
    asset_type = detect_asset_type(ticker)
    if asset_type == "leveraged":
        return LeveragedMomentumStrategy(df, ticker=ticker, stock_mode=True).run()
    elif asset_type == "crypto":
        return CryptoMomentumStrategy(df, ticker=ticker).run()
    elif asset_type == "forex":
        return ForexICTStrategy(df, ticker=ticker).run()
    elif asset_type == "commodity":
        # Crypto momentum works great on gold/silver
        return CryptoMomentumStrategy(df, ticker=ticker).run()
    else:
        return SMCStrategy(df, ticker=ticker, stock_mode=stock_mode).run()

def confidence_badge(signals):
    warned = sum(1 for s in signals if "⚠" in s.details)
    if warned == 0:
        return '<span style="color:#4ade80;font-weight:600;">🟢 HIGH</span>'
    elif warned <= 2:
        return '<span style="color:#fbbf24;font-weight:600;">🟡 MODERATE</span>'
    else:
        return '<span style="color:#f87171;font-weight:600;">🔴 LOW</span>'

def trend_info(signals):
    for s in signals:
        if "Trend momentum" in s.details or "No clear trend" in s.details:
            return s.details
    return None

def action_bg(action_val):
    return {"action-strong-buy": "background:rgba(16,185,129,0.2);color:#34d399;",
            "action-buy": "background:rgba(34,197,94,0.2);color:#4ade80;",
            "action-hold": "background:rgba(234,179,8,0.2);color:#fbbf24;",
            "action-sell": "background:rgba(239,68,68,0.2);color:#f87171;",
            "action-strong-sell": "background:rgba(220,38,38,0.2);color:#fca5a5;"
            }.get(action_val, "")


# ──────────────────────────────────────────────────────────
#  Scanner result renderer (shared by Daily Analysis & Custom Scanner)
# ──────────────────────────────────────────────────────────

def render_scanner_results(results, currency_sym, show_obs, show_fvgs,
                           show_liq, show_structure, show_trade, show_pd):
    """Render scanner results: metrics, rows, and expandable charts."""
    if not results:
        st.error("No results. Check your ticker symbols.")
        return

    results.sort(key=lambda r: abs(r["setup"].composite_score), reverse=True)

    actionable = [r for r in results if r["setup"].action.value in ("STRONG BUY", "BUY", "SELL", "STRONG SELL")]
    buys = [r for r in actionable if "BUY" in r["setup"].action.value]
    sells = [r for r in actionable if "SELL" in r["setup"].action.value]
    holds = [r for r in results if r["setup"].action.value == "HOLD"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scanned", len(results))
    m2.metric("Actionable", len(actionable))
    m3.metric("Buy Signals", len(buys))
    m4.metric("Sell Signals", len(sells))

    st.markdown("")

    for r in results:
        s = r["setup"]
        abg = action_bg(action_class(s.action))
        cur = r["df"].iloc[-1]["Close"]
        prv = r["df"].iloc[-2]["Close"] if len(r["df"]) > 1 else cur
        c = ((cur - prv) / prv) * 100
        cc = "#4ade80" if c >= 0 else "#f87171"
        cs = "+" if c >= 0 else ""
        conf = confidence_badge(s.signals)

        st.markdown(
            f'<div class="scanner-row">'
            f'<div class="scanner-ticker">{s.ticker}</div>'
            f'<div class="scanner-action" style="{abg}">{s.action.value}</div>'
            f'<div><span class="bias-badge {bias_class(s.bias)}">{s.bias.value.upper()}</span></div>'
            f'<div class="scanner-score">{s.composite_score}</div>'
            f'<div class="scanner-details">{fmt_price(cur, currency_sym)} '
            f'<span style="color:{cc};">{cs}{c:.2f}%</span>'
            f' · SL {fmt_price(s.stop_loss, currency_sym)}'
            f' · TP {fmt_price(s.take_profit, currency_sym)}'
            f' · R:R 1:{s.risk_reward:.1f}'
            f' · {conf}'
            f' · {len(s.signals)} signals</div></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Detailed Charts")
    for r in results:
        with st.expander(f"📈 {r['ticker']} — {r['setup'].action.value} (Score: {r['setup'].composite_score})"):
            ch = build_main_chart(
                r["df"], r["strategy"],
                show_order_blocks=show_obs, show_fvgs=show_fvgs,
                show_liquidity=show_liq, show_structure=show_structure,
                show_trade_levels=show_trade, show_premium_discount=show_pd,
                height=480,
            )
            st.plotly_chart(ch, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})


def run_scan(tickers, period, interval, stock_mode):
    """Run analysis on a list of tickers and return results."""
    results = []
    progress = st.progress(0, text="Scanning tickers...")
    for i, t in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"Analyzing {t}... ({i+1}/{len(tickers)})")
        try:
            data = fetch_data(t, period, interval)
            strat = run_analysis(data, t, stock_mode=stock_mode)
            results.append({"ticker": t, "setup": strat.trade_setup, "strategy": strat, "df": data})
        except Exception as e:
            st.warning(f"Skipped **{t}**: {e}")
    progress.empty()
    return results


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📊 Stock Trader")
    st.caption("ICT & Smart Money Concepts")

    # ── Mode (top-level choice) ───────────────────
    st.subheader("Mode", divider="gray")
    mode = st.radio(
        "Analysis Mode",
        ["📊 Daily Analysis", "🔍 Search Ticker", "📋 Custom Scanner", "🧪 Backtest"],
        label_visibility="collapsed",
        key="mode_selector",
    )

    # ── Asset Class (always visible) ─────────────
    st.subheader("Asset Class", divider="gray")
    asset_class_names = list(config.ASSET_CLASSES.keys())
    asset_class = st.radio(
        "Asset Class",
        asset_class_names,
        horizontal=True,
        label_visibility="collapsed",
        key="asset_class_selector",
    )
    ac = config.ASSET_CLASSES[asset_class]
    currency_sym = ac["currency_symbol"]
    position_unit = ac["unit"]
    ac_icon = AC_ICONS.get(asset_class, "📊")
    use_stock_mode = (asset_class == "Stocks")

    # ── Mode-specific controls ─────────────────────
    placeholders = {
        "Stocks": "e.g. AAPL, SPY, TSLA",
        "Crypto": "e.g. BTC-USD, ETH-USD",
        "Forex":  "e.g. EURUSD=X, GBPUSD=X",
        "Commodities": "e.g. GC=F, CL=F, SI=F",
    }

    # Map asset class → best-matching session for Daily Analysis
    ASSET_TO_SESSION = {
        "Stocks": "🏛️ Stocks — NY Open",
        "Crypto": "🇺🇸 NY Session — Crypto, Metals & Forex",
        "Forex": "🇬🇧 London Session — Forex & Metals",
        "Commodities": "🇬🇧 London Session — Forex & Metals",
    }

    if mode == "📊 Daily Analysis":
        st.subheader("Trading Session", divider="gray")
        session_names = list(DAILY_SESSIONS.keys())
        # Auto-select the session that matches the asset class
        default_session = ASSET_TO_SESSION.get(asset_class, session_names[0])
        default_idx = session_names.index(default_session) if default_session in session_names else 0
        selected_session = st.selectbox(
            "Select trading session",
            session_names,
            index=default_idx,
            key=f"session_selector_{asset_class}",
        )
        dw = DAILY_SESSIONS[selected_session]
        st.caption(", ".join(dw["tickers"]))
        if dw["stock_mode"]:
            st.info("**Stock Mode** active — medium risk, ATR stops, trend momentum", icon="🏛️")

        # Optional timeframe override
        st.subheader("Timeframe", divider="gray")
        da_override = st.toggle(
            "Override default interval",
            value=False,
            key=f"da_override_{asset_class}",
        )
        if da_override:
            da_intervals = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"]
            da_default_idx = da_intervals.index(dw["interval"]) if dw["interval"] in da_intervals else 4
            da_interval = st.selectbox(
                "Interval",
                da_intervals,
                index=da_default_idx,
                format_func=lambda x: INTERVAL_LABELS.get(x, x),
                key=f"da_interval_{asset_class}",
            )
            da_valid_periods = PERIODS_FOR_INTERVAL.get(da_interval, ["1mo", "3mo", "6mo", "1y"])
            da_period = st.selectbox(
                "Period",
                da_valid_periods,
                index=min(len(da_valid_periods) - 1, 2),
                format_func=lambda x: PERIOD_LABELS.get(x, x),
                key=f"da_period_{asset_class}_{da_interval}",
            )
        else:
            da_interval = dw["interval"]
            da_period = dw["period"]

        st.info(f"Interval: **{INTERVAL_LABELS.get(da_interval, da_interval)}** · Period: **{PERIOD_LABELS.get(da_period, da_period)}**", icon="⏱️")

    elif mode == "🔍 Search Ticker":
        st.subheader("Ticker", divider="gray")
        ticker = st.text_input(
            "Enter ticker symbol",
            value=ac["default_ticker"],
            placeholder=placeholders.get(asset_class, ""),
            key=f"ticker_{asset_class}",
        ).upper().strip()

        # Preset group picker
        preset_names = list(ac["presets"].keys())
        selected_group = st.selectbox(
            "Or pick from presets",
            ["— Type above or pick —"] + preset_names,
            key=f"group_{asset_class}",
        )
        if selected_group != "— Type above or pick —":
            ticker = st.selectbox(
                "Select ticker",
                ac["presets"][selected_group],
                key=f"pick_{asset_class}_{selected_group}",
            )

    elif mode == "📋 Custom Scanner":
        st.subheader("Tickers", divider="gray")
        preset_names = list(ac["presets"].keys())
        scanner_source = st.selectbox(
            "Watchlist",
            ["Custom"] + preset_names,
            key=f"scan_src_{asset_class}",
        )
        if scanner_source == "Custom":
            default_val = ", ".join(ac["presets"][preset_names[0]]) if preset_names else ""
            tickers_input = st.text_input(
                "Tickers (comma-separated)",
                value=default_val,
                key=f"scan_input_{asset_class}",
            )
            tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        else:
            tickers_list = ac["presets"][scanner_source]
            st.caption(", ".join(tickers_list))

    # ── Backtest controls ──────────────────────
    if mode == "🧪 Backtest":
        st.subheader("Date Range", divider="gray")

        # Preset date ranges
        BT_PRESETS = {
            "Week 1: Jan 26–30": ("2026-01-26", "2026-01-30"),
            "Week 2: Feb 2–6": ("2026-02-02", "2026-02-06"),
            "Both weeks: Jan 26 – Feb 6": ("2026-01-26", "2026-02-06"),
            "Custom": None,
        }
        bt_preset = st.selectbox(
            "Quick select",
            list(BT_PRESETS.keys()),
            key="bt_preset",
        )
        if BT_PRESETS[bt_preset] is not None:
            bt_start_date = datetime.strptime(BT_PRESETS[bt_preset][0], "%Y-%m-%d").date()
            bt_end_date = datetime.strptime(BT_PRESETS[bt_preset][1], "%Y-%m-%d").date()
        else:
            bt_start_date = st.date_input("Start date", value=date(2026, 1, 26), key="bt_start")
            bt_end_date = st.date_input("End date", value=date(2026, 1, 30), key="bt_end")

        st.caption(f"📅 {bt_start_date} → {bt_end_date}")

        st.subheader("Trading Session", divider="gray")
        session_names = list(DAILY_SESSIONS.keys())
        default_session_bt = ASSET_TO_SESSION.get(asset_class, session_names[0])
        default_idx_bt = session_names.index(default_session_bt) if default_session_bt in session_names else 0
        bt_session = st.selectbox(
            "Session preset",
            session_names,
            index=default_idx_bt,
            key=f"bt_session_{asset_class}",
        )
        bt_dw = DAILY_SESSIONS[bt_session]
        st.caption(", ".join(bt_dw["tickers"]))
        if bt_dw["stock_mode"]:
            st.info("**Stock Mode** active", icon="🏛️")

        st.subheader("Settings", divider="gray")
        bt_interval = st.selectbox(
            "Interval",
            ["1h", "1d"],
            index=0 if not bt_dw["stock_mode"] else 1,
            format_func=lambda x: INTERVAL_LABELS.get(x, x),
            key="bt_interval",
        )
        bt_max_hold = st.slider("Max bars to hold", 10, 100, 50, 5, key="bt_maxhold")

    # ── Timeframe (Search & Custom only) ──────
    if mode in ("🔍 Search Ticker", "📋 Custom Scanner"):
        st.subheader("Timeframe", divider="gray")
        all_intervals = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"]
        interval = st.selectbox(
            "Interval",
            all_intervals,
            index=5,
            format_func=lambda x: INTERVAL_LABELS.get(x, x),
            key="interval_sel",
        )
        valid_periods = PERIODS_FOR_INTERVAL.get(interval, ["1mo", "3mo", "6mo", "1y"])
        period = st.selectbox(
            "Period",
            valid_periods,
            index=min(len(valid_periods) - 1, 2),
            format_func=lambda x: PERIOD_LABELS.get(x, x),
            key=f"period_sel_{interval}",
        )
        if interval in ("1m", "5m", "15m", "30m"):
            st.info(f"⏱️ Max: **{PERIOD_LABELS.get(valid_periods[-1], valid_periods[-1])}** for {INTERVAL_LABELS[interval]} candles.", icon="ℹ️")

    # ── Chart Overlays ────────────────────────────
    st.subheader("Overlays", divider="gray")
    show_structure = st.toggle("Market Structure", value=True, key="t_ms")
    show_obs = st.toggle("Order Blocks", value=True, key="t_ob")
    show_fvgs = st.toggle("Fair Value Gaps", value=True, key="t_fvg")
    show_liq = st.toggle("Liquidity Levels", value=True, key="t_liq")
    show_pd = st.toggle("Premium / Discount", value=True, key="t_pd")
    show_trade = st.toggle("Trade Levels (SL/TP)", value=True, key="t_sl")

    # ── Risk ──────────────────────────────────────
    st.subheader("Risk", divider="gray")
    capital = st.number_input("Capital ($)", value=100_000, step=10_000, key="capital")
    risk_pct = st.slider("Risk per trade (%)", 0.5, 5.0, 2.0, 0.5, key="risk")

    st.divider()
    st.caption("For educational purposes only. Not financial advice.")


# ══════════════════════════════════════════════════════════
#  Session state for results persistence
# ══════════════════════════════════════════════════════════

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
    st.session_state.scan_session = None


# ══════════════════════════════════════════════════════════
#  MAIN CONTENT — Daily Analysis
# ══════════════════════════════════════════════════════════

if mode == "📊 Daily Analysis":
    dw = DAILY_SESSIONS[selected_session]
    session_icon = dw["icon"]
    session_stock_mode = dw["stock_mode"]

    # Clear cached results when session or asset class changes
    session_key = f"{asset_class}_{selected_session}"
    if st.session_state.scan_session != session_key:
        st.session_state.scan_results = None
        st.session_state.scan_session = session_key

    # Use the (possibly overridden) interval and period
    eff_interval = da_interval
    eff_period = da_period

    st.markdown(
        f'<div class="page-header">'
        f'<span class="page-header-title">{selected_session}</span>'
        f'<span class="page-header-meta">'
        f'{len(dw["tickers"])} tickers · {PERIOD_LABELS.get(eff_period, eff_period)} · '
        f'{INTERVAL_LABELS.get(eff_interval, eff_interval)}</span>'
        + (' <span class="page-header-tag">STOCK MODE</span>' if session_stock_mode else '')
        + '</div>',
        unsafe_allow_html=True,
    )

    col_run, col_home = st.columns([4, 1], gap="small")
    with col_run:
        run_clicked = st.button("🚀  Run Analysis", type="primary", use_container_width=True)
    with col_home:
        home_clicked = st.button("🏠", use_container_width=True, help="Back to home")

    if home_clicked:
        st.session_state.scan_results = None
        st.rerun()

    if run_clicked:
        results = run_scan(dw["tickers"], eff_period, eff_interval, stock_mode=session_stock_mode)
        st.session_state.scan_results = results
        st.session_state.scan_session = session_key

    # Show results if we have them
    if st.session_state.scan_results:
        scan_sym = "$"
        render_scanner_results(st.session_state.scan_results, scan_sym, show_obs,
                               show_fvgs, show_liq, show_structure, show_trade, show_pd)
    else:
        st.markdown(
            f'<div class="landing-box">'
            f'<div class="landing-icon">{session_icon}</div>'
            f'<div class="landing-title">Select a session and click Run Analysis</div>'
            f'<div class="landing-sub">{", ".join(dw["tickers"])}</div></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════
#  MAIN CONTENT — Search Ticker
# ══════════════════════════════════════════════════════════

elif mode == "🔍 Search Ticker":
    try:
        with st.spinner(f"Fetching {ticker} data..."):
            df = fetch_data(ticker, period, interval)
            strategy = run_analysis(df, ticker, stock_mode=use_stock_mode)
            setup = strategy.trade_setup
    except Exception as e:
        st.error(f"Could not fetch data for **{ticker}**: {e}")
        st.stop()

    # ── Header row ────────────────────────────────
    top_left, top_right = st.columns([3, 1])

    with top_left:
        price = df.iloc[-1]["Close"]
        prev = df.iloc[-2]["Close"] if len(df) > 1 else price
        chg = price - prev
        chg_pct = (chg / prev) * 100 if prev else 0
        color = "#22c55e" if chg >= 0 else "#ef4444"
        sign = "+" if chg >= 0 else ""

        mode_tag = '<span class="page-header-tag">STOCK MODE</span>' if use_stock_mode else ""
        st.markdown(
            f'<div class="ticker-header">'
            f'<span class="ticker-symbol">{ticker}</span>'
            f'<span class="ticker-price">{fmt_price(price, currency_sym)}</span>'
            f'<span class="ticker-change" style="color:{color};">{sign}{chg:.2f} ({sign}{chg_pct:.2f}%)</span>'
            f'<span class="ticker-badge">{ac_icon} {asset_class}</span>{mode_tag}</div>',
            unsafe_allow_html=True,
        )

        conf = confidence_badge(setup.signals)
        trend = trend_info(setup.signals)
        trend_html = f'<span style="color:#94a3b8;font-size:0.78rem;margin-left:12px;">{trend}</span>' if trend else ""

        st.markdown(
            f'<div class="ticker-info-row">'
            f'<span class="bias-badge {bias_class(setup.bias)}">{setup.bias.value.upper()} BIAS</span>'
            f'<span style="font-size:0.78rem;">Confidence: {conf}</span>'
            f'{trend_html}'
            f'<span style="color:#64748b;font-size:0.78rem;">'
            f'{len(df)} candles · {PERIOD_LABELS.get(period, period)} · {INTERVAL_LABELS.get(interval, interval)}</span></div>',
            unsafe_allow_html=True,
        )

    with top_right:
        st.markdown(
            f'<div style="text-align:right;padding-top:8px;">'
            f'<div class="action-badge {action_class(setup.action)}">{setup.action.value}</div>'
            f'<div style="color:#64748b;font-size:0.75rem;margin-top:4px;">Score: {setup.composite_score}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Tabs ──────────────────────────────────────
    tab_chart, tab_signals, tab_details = st.tabs(["📈 Chart", "🔔 Signals", "📋 Details"])

    with tab_chart:
        chart = build_main_chart(
            df, strategy,
            show_order_blocks=show_obs, show_fvgs=show_fvgs,
            show_liquidity=show_liq, show_structure=show_structure,
            show_trade_levels=show_trade, show_premium_discount=show_pd,
            height=620,
        )
        st.plotly_chart(chart, use_container_width=True, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
        })

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Entry", fmt_price(setup.entry_price, currency_sym))
        m2.metric("Stop Loss", fmt_price(setup.stop_loss, currency_sym))
        m3.metric("Take Profit", fmt_price(setup.take_profit, currency_sym))
        m4.metric("R : R", f"1 : {setup.risk_reward:.1f}")
        m5.metric("Position", f"{setup.position_size} {position_unit}")
        m6.metric("Signals", f"{len(setup.signals)}")

    with tab_signals:
        st.subheader("Score Breakdown")
        gauge = build_score_gauge(strategy.bullish_score, strategy.bearish_score)
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

        c1, c2, c3 = st.columns(3)
        c1.metric("Bullish Score", strategy.bullish_score)
        c2.metric("Bearish Score", strategy.bearish_score)
        c3.metric("Net Score", strategy.net_score)

        col_pie, col_list = st.columns([1, 2])
        with col_pie:
            st.subheader("Distribution")
            if setup.signals:
                st.plotly_chart(build_signal_breakdown(setup.signals),
                                use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No signals detected.")

        with col_list:
            st.subheader("All Signals")
            for sig in setup.signals:
                nice = sig.signal_type.value.replace("_", " ").title()
                st.markdown(
                    f'<div class="signal-card">{sig_icon(sig)}'
                    f'<div class="signal-body"><div class="signal-type">{nice}</div>'
                    f'<div class="signal-detail">{sig.details}</div></div>'
                    f'<div class="signal-score">+{sig.score}</div></div>',
                    unsafe_allow_html=True,
                )
            if not setup.signals:
                st.info("No signals detected for this ticker/timeframe.")

    with tab_details:
        col_a, col_b = st.columns(2)
        _u = position_unit.rstrip("s") if position_unit != "lots" else "lot"
        asset_type = detect_asset_type(ticker)
        strategy_name = {"leveraged": "Leveraged Momentum", "crypto": "Crypto Momentum",
                         "forex": "Forex ICT", "commodity": "Crypto Momentum",
                         "stocks": "ICT / Smart Money"}.get(asset_type, "ICT / Smart Money")

        with col_a:
            st.markdown(
                '<div class="info-box"><h4>Trade Setup</h4>'
                + "".join([
                    f'<div class="info-row"><span class="info-label">{k}</span><span class="info-value">{v}</span></div>'
                    for k, v in [
                        ("Strategy", strategy_name),
                        ("Action", setup.action.value),
                        ("Entry Price", fmt_price(setup.entry_price, currency_sym)),
                        ("Stop Loss", fmt_price(setup.stop_loss, currency_sym)),
                        ("Take Profit", fmt_price(setup.take_profit, currency_sym)),
                        ("Risk : Reward", f"1 : {setup.risk_reward:.1f}"),
                        ("Position Size", f"{setup.position_size} {position_unit}"),
                    ]
                ]) + '</div>',
                unsafe_allow_html=True,
            )

            # ICT-specific details (only for SMC and Forex ICT strategies)
            if hasattr(strategy, 'structure') and strategy.structure is not None:
                ms = strategy.structure
                st.markdown(
                    '<div class="info-box"><h4>Market Structure</h4>'
                    + "".join([
                        f'<div class="info-row"><span class="info-label">{k}</span><span class="info-value">{v}</span></div>'
                        for k, v in [
                            ("Bias", ms.bias.value.upper()),
                            ("Swing Highs", len(ms.swing_highs)),
                            ("Swing Lows", len(ms.swing_lows)),
                            ("Structure Signals", len(ms.signals)),
                        ]
                    ]) + '</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Momentum strategy summary
                bull_sigs = [s for s in setup.signals if s.bias == MarketBias.BULLISH]
                bear_sigs = [s for s in setup.signals if s.bias == MarketBias.BEARISH]
                st.markdown(
                    '<div class="info-box"><h4>Signal Summary</h4>'
                    + "".join([
                        f'<div class="info-row"><span class="info-label">{k}</span><span class="info-value">{v}</span></div>'
                        for k, v in [
                            ("Strategy", strategy_name),
                            ("Bullish Signals", f"{len(bull_sigs)} (score: {sum(s.score for s in bull_sigs)})"),
                            ("Bearish Signals", f"{len(bear_sigs)} (score: {sum(s.score for s in bear_sigs)})"),
                            ("Net Score", setup.composite_score),
                        ]
                    ]) + '</div>',
                    unsafe_allow_html=True,
                )

        with col_b:
            if hasattr(strategy, 'ob_detector') and strategy.ob_detector is not None:
                ob = strategy.ob_detector
                st.markdown(
                    '<div class="info-box"><h4>Order Blocks</h4>'
                    + "".join([
                        f'<div class="info-row"><span class="info-label">{k}</span><span class="info-value">{v}</span></div>'
                        for k, v in [
                            ("Total Found", len(ob.order_blocks)),
                            ("Active", len(ob.active_blocks())),
                            ("Bullish", sum(1 for o in ob.order_blocks if o.ob_type == "bullish")),
                            ("Bearish", sum(1 for o in ob.order_blocks if o.ob_type == "bearish")),
                        ]
                    ]) + '</div>',
                    unsafe_allow_html=True,
                )
            if hasattr(strategy, 'fvg_detector') and strategy.fvg_detector is not None:
                fvg = strategy.fvg_detector
                st.markdown(
                    '<div class="info-box"><h4>Fair Value Gaps</h4>'
                    + "".join([
                        f'<div class="info-row"><span class="info-label">{k}</span><span class="info-value">{v}</span></div>'
                        for k, v in [
                            ("Total Found", len(fvg.fvgs)),
                            ("Active", len(fvg.active_fvgs())),
                            ("Bullish", sum(1 for f in fvg.fvgs if f.fvg_type == "bullish")),
                            ("Bearish", sum(1 for f in fvg.fvgs if f.fvg_type == "bearish")),
                        ]
                    ]) + '</div>',
                    unsafe_allow_html=True,
                )
            if hasattr(strategy, 'liq_analyzer') and strategy.liq_analyzer is not None:
                liq = strategy.liq_analyzer
                st.markdown(
                    '<div class="info-box"><h4>Liquidity</h4>'
                    + "".join([
                        f'<div class="info-row"><span class="info-label">{k}</span><span class="info-value">{v}</span></div>'
                        for k, v in [
                            ("Levels", len(liq.levels)),
                            ("Equal Highs", sum(1 for l in liq.levels if l.level_type == "equal_highs")),
                            ("Equal Lows", sum(1 for l in liq.levels if l.level_type == "equal_lows")),
                            ("Sweeps", sum(1 for l in liq.levels if l.swept)),
                        ]
                    ]) + '</div>',
                    unsafe_allow_html=True,
                )
            # For non-ICT strategies, show indicator details
            if not hasattr(strategy, 'ob_detector'):
                # Group signals by type
                from collections import Counter
                type_counts = Counter(s.signal_type.value for s in setup.signals)
                st.markdown(
                    '<div class="info-box"><h4>Indicator Breakdown</h4>'
                    + "".join([
                        f'<div class="info-row"><span class="info-label">{k.replace("_", " ").title()}</span><span class="info-value">{v} signal{"s" if v > 1 else ""}</span></div>'
                        for k, v in type_counts.most_common()
                    ]) + '</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("View Raw OHLCV Data"):
            _s = df.iloc[-1]["Close"]
            _d = 2 if _s >= 1 else (4 if _s >= 0.01 else 6)
            _f = f"{currency_sym}{{:.{_d}f}}"
            st.dataframe(
                df.tail(50).style.format({"Open": _f, "High": _f, "Low": _f, "Close": _f, "Volume": "{:,.0f}"}),
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════
#  MAIN CONTENT — Custom Scanner
# ══════════════════════════════════════════════════════════

elif mode == "📋 Custom Scanner":
    st.markdown(
        f'<div class="page-header">'
        f'<span class="page-header-title">{ac_icon} {asset_class} Scanner</span>'
        f'<span class="page-header-meta">'
        f'{len(tickers_list)} tickers · {PERIOD_LABELS.get(period, period)} · '
        f'{INTERVAL_LABELS.get(interval, interval)}</span></div>',
        unsafe_allow_html=True,
    )

    if st.button("🔍  Run Scanner", type="primary", use_container_width=True):
        results = run_scan(tickers_list, period, interval, stock_mode=use_stock_mode)
        render_scanner_results(results, currency_sym, show_obs, show_fvgs,
                               show_liq, show_structure, show_trade, show_pd)


# ══════════════════════════════════════════════════════════
#  MAIN CONTENT — Backtest
# ══════════════════════════════════════════════════════════

elif mode == "🧪 Backtest":
    st.markdown(
        f'<div class="page-header">'
        f'<span class="page-header-title">🧪 Strategy Backtest</span>'
        f'<span class="page-header-meta">'
        f'{bt_start_date} → {bt_end_date} · {bt_session}</span></div>',
        unsafe_allow_html=True,
    )

    if st.button("🚀  Run Backtest", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting backtest...")
        tickers = bt_dw["tickers"]

        # Map each ticker to the right strategy class
        def _strategy_for(t):
            at = detect_asset_type(t)
            if at == "leveraged":
                return LeveragedMomentumStrategy
            elif at == "crypto":
                return CryptoMomentumStrategy
            elif at == "forex":
                return ForexICTStrategy
            elif at == "commodity":
                return CryptoMomentumStrategy  # works well on gold/silver
            return SMCStrategy

        # Run backtest for each ticker
        engines = {}
        for i, t in enumerate(tickers):
            progress.progress((i + 1) / len(tickers), text=f"Backtesting {t}... ({i+1}/{len(tickers)})")
            try:
                eng = BacktestEngine(
                    ticker=t,
                    period="2y",
                    interval=bt_interval,
                    stock_mode=bt_dw["stock_mode"],
                    strategy_class=_strategy_for(t),
                )
                eng.run()
                engines[t] = eng
            except Exception as e:
                st.warning(f"Skipped **{t}**: {e}")
        progress.empty()

        # ── Aggregate metrics across all tickers ──
        all_trades = []
        for eng in engines.values():
            all_trades.extend(eng._trades)

        total_trades = len(all_trades)
        total_wins = sum(1 for t in all_trades if t.pnl > 0)
        total_losses = sum(1 for t in all_trades if t.pnl <= 0 and t.exit_price > 0)
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        total_return = 0
        for eng in engines.values():
            ret = (eng._equity - eng._initial_capital) / eng._initial_capital * 100
            total_return += ret
        avg_return = total_return / len(engines) if engines else 0

        # ── Summary metrics ──
        st.markdown(
            '<div class="page-header" style="margin-top:20px;">'
            '<span class="page-header-title" style="font-size:1.3rem;">📊 Overall Performance</span></div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Trades", total_trades)
        m2.metric("Wins", total_wins)
        m3.metric("Losses", total_losses)
        m4.metric("Tickers", len(engines))
        m5.metric("Win Rate", f"{win_rate:.1f}%")
        m6.metric("Avg Return", f"{avg_return:+.2f}%")

        if total_trades > 0:
            # ── Equity curve (use the first engine's full curve as example) ──
            st.markdown(
                '<div class="page-header" style="margin-top:30px;">'
                '<span class="page-header-title" style="font-size:1.3rem;">📈 Equity Curves</span></div>',
                unsafe_allow_html=True,
            )
            import plotly.graph_objects as go
            eq_fig = go.Figure()
            colors = ["#4ade80", "#60a5fa", "#fbbf24", "#f87171", "#a78bfa",
                      "#34d399", "#38bdf8", "#fb923c", "#e879f9", "#94a3b8"]
            for idx, (ticker, eng) in enumerate(engines.items()):
                ec = eng.equity_curve
                if not ec.empty:
                    eq_fig.add_trace(go.Scatter(
                        x=ec["date"].astype(str) if "date" in ec.columns else list(range(len(ec))),
                        y=ec["equity"].tolist(),
                        mode="lines",
                        name=ticker,
                        line=dict(color=colors[idx % len(colors)], width=2),
                    ))
            eq_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e17",
                plot_bgcolor="#0a0e17",
                height=350,
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(showgrid=False, title="Bar"),
                yaxis=dict(title="Equity ($)", gridcolor="#1e293b"),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(eq_fig, use_container_width=True, config={"displayModeBar": False})

            # ── Per-ticker breakdown ──
            st.markdown(
                '<div class="page-header" style="margin-top:30px;">'
                '<span class="page-header-title" style="font-size:1.3rem;">🔍 Per-Ticker Breakdown</span></div>',
                unsafe_allow_html=True,
            )

            for ticker, eng in engines.items():
                closed = eng._trades
                total = len(closed)
                if total == 0:
                    continue
                wins = sum(1 for t in closed if t.pnl > 0)
                wr = (wins / total * 100) if total else 0
                ret = (eng._equity - eng._initial_capital) / eng._initial_capital * 100
                avg_r = sum(t.r_multiple for t in closed) / total if total else 0

                with st.expander(
                    f"{'✅' if wr >= 50 else '⚠️'} {ticker} — "
                    f"{total} trades · {wr:.0f}% win · {ret:+.2f}% return"
                ):
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("Trades", total)
                    rc2.metric("Win Rate", f"{wr:.0f}%")
                    rc3.metric("Return", f"{ret:+.2f}%")
                    rc4.metric("Avg R", f"{avg_r:+.2f}R")

                    if closed:
                        trade_data = []
                        for t in closed:
                            emoji = "✅" if t.pnl > 0 else "❌"
                            trade_data.append({
                                "Entry Date": str(t.entry_date)[:10] if t.entry_date else "",
                                "Direction": t.direction.upper(),
                                "Entry": f"${t.entry_price:.4f}",
                                "SL": f"${t.stop_loss:.4f}",
                                "TP": f"${t.take_profit:.4f}",
                                "Exit": f"${t.exit_price:.4f}",
                                "Exit Reason": t.exit_reason,
                                f"{emoji} P&L": f"${t.pnl:+.2f}",
                                "P&L %": f"{t.pnl_pct:+.2f}%",
                                "R": f"{t.r_multiple:+.2f}",
                            })
                        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)

            # ── All trades summary ──
            st.markdown(
                '<div class="page-header" style="margin-top:30px;">'
                '<span class="page-header-title" style="font-size:1.3rem;">📋 All Trades</span></div>',
                unsafe_allow_html=True,
            )
            all_data = []
            for t in sorted(all_trades, key=lambda x: str(x.entry_date)):
                emoji = "✅" if t.pnl > 0 else "❌"
                all_data.append({
                    "Ticker": t.ticker,
                    "Date": str(t.entry_date)[:10] if t.entry_date else "",
                    "Dir": t.direction.upper(),
                    "Entry": f"${t.entry_price:.4f}",
                    "Exit": f"${t.exit_price:.4f}",
                    "Reason": t.exit_reason,
                    f"{emoji} P&L": f"${t.pnl:+.2f}",
                    "R": f"{t.r_multiple:+.2f}",
                })
            st.dataframe(pd.DataFrame(all_data), use_container_width=True, hide_index=True)

        elif total_trades == 0:
            st.warning(
                "No trades were generated in this backtest period. "
                "Try a longer period or different session/tickers."
            )

    else:
        st.markdown(
            '<div class="landing-box">'
            '<div class="landing-icon">🧪</div>'
            '<div class="landing-title">Configure your backtest and click Run</div>'
            '<div class="landing-sub">'
            'The strategy will run day-by-day on historical data and simulate trade outcomes</div>'
            '<div class="landing-sub" style="margin-top:8px;">'
            'For each signal: walks forward bar-by-bar to check if TP or SL is hit first</div></div>',
            unsafe_allow_html=True,
        )
