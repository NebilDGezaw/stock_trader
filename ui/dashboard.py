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
for _mod in ["config", "strategies.smc_strategy", "utils.helpers"]:
    if _mod in sys.modules:
        del sys.modules[_mod]

import config

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta

from data.fetcher import StockDataFetcher
from strategies.smc_strategy import SMCStrategy
from models.signals import TradeAction, MarketBias, SignalType
from ui.charts import (
    build_main_chart,
    build_score_gauge,
    build_signal_breakdown,
)
from backtesting.engine import Backtester, backtest_session, BacktestReport

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

def run_analysis(df, ticker, stock_mode=False):
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
        st.info(f"Interval: **{INTERVAL_LABELS.get(dw['interval'], dw['interval'])}** · Period: **{PERIOD_LABELS.get(dw['period'], dw['period'])}**", icon="⏱️")
        if dw["stock_mode"]:
            st.info("**Stock Mode** active — medium risk, ATR stops, trend momentum", icon="🏛️")

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

    st.markdown(
        f'<div style="margin-bottom:20px;">'
        f'<span style="font-size:1.8rem;font-weight:800;color:#e2e8f0;">'
        f'{selected_session}</span>'
        f'<span style="color:#64748b;font-size:0.85rem;margin-left:16px;">'
        f'{len(dw["tickers"])} tickers · {PERIOD_LABELS.get(dw["period"], dw["period"])} · '
        f'{INTERVAL_LABELS.get(dw["interval"], dw["interval"])}</span>'
        + (' <span style="font-size:0.65rem;color:#a78bfa;background:rgba(167,139,250,0.15);padding:2px 8px;border-radius:4px;margin-left:8px;">STOCK MODE</span>' if session_stock_mode else '')
        + '</div>',
        unsafe_allow_html=True,
    )

    btn_col, home_col = st.columns([3, 1])
    with btn_col:
        run_clicked = st.button("🚀  Run Analysis", type="primary", use_container_width=True)
    with home_col:
        home_clicked = st.button("🏠 Home", use_container_width=True)

    if home_clicked:
        st.session_state.scan_results = None
        st.rerun()

    if run_clicked:
        results = run_scan(dw["tickers"], dw["period"], dw["interval"], stock_mode=session_stock_mode)
        st.session_state.scan_results = results
        st.session_state.scan_session = session_key

    # Show results if we have them
    if st.session_state.scan_results:
        scan_sym = "$"
        render_scanner_results(st.session_state.scan_results, scan_sym, show_obs,
                               show_fvgs, show_liq, show_structure, show_trade, show_pd)
    else:
        st.markdown(
            f'<div style="text-align:center;padding:60px 20px;color:#64748b;">'
            f'<div style="font-size:3rem;margin-bottom:16px;">{session_icon}</div>'
            f'<div style="font-size:1.2rem;font-weight:600;color:#94a3b8;">Select a session and click Run Analysis</div>'
            f'<div style="font-size:0.85rem;margin-top:8px;">'
            f'{", ".join(dw["tickers"])}</div></div>',
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

        mode_tag = '<span style="font-size:0.65rem;color:#a78bfa;background:rgba(167,139,250,0.15);padding:2px 8px;border-radius:4px;margin-left:8px;">STOCK MODE</span>' if use_stock_mode else ""
        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px;">'
            f'<span style="font-size:2rem;font-weight:800;color:#e2e8f0;">{ticker}</span>'
            f'<span style="font-size:1.8rem;font-weight:700;color:#e2e8f0;">{fmt_price(price, currency_sym)}</span>'
            f'<span style="font-size:1rem;font-weight:600;color:{color};">{sign}{chg:.2f} ({sign}{chg_pct:.2f}%)</span>'
            f'<span style="font-size:0.75rem;color:#64748b;background:#1e293b;padding:3px 10px;border-radius:5px;">'
            f'{ac_icon} {asset_class}</span>{mode_tag}</div>',
            unsafe_allow_html=True,
        )

        conf = confidence_badge(setup.signals)
        trend = trend_info(setup.signals)
        trend_html = f'<span style="color:#94a3b8;font-size:0.78rem;margin-left:12px;">{trend}</span>' if trend else ""

        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">'
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

        with col_a:
            st.markdown(
                '<div class="info-box"><h4>Trade Setup</h4>'
                + "".join([
                    f'<div class="info-row"><span class="info-label">{k}</span><span class="info-value">{v}</span></div>'
                    for k, v in [
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

        with col_b:
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
        f'<div style="margin-bottom:20px;">'
        f'<span style="font-size:1.8rem;font-weight:800;color:#e2e8f0;">'
        f'{ac_icon} {asset_class} Scanner</span>'
        f'<span style="color:#64748b;font-size:0.85rem;margin-left:16px;">'
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
        f'<div style="margin-bottom:20px;">'
        f'<span style="font-size:1.8rem;font-weight:800;color:#e2e8f0;">'
        f'🧪 Strategy Backtest</span>'
        f'<span style="color:#64748b;font-size:0.85rem;margin-left:16px;">'
        f'{bt_start_date} → {bt_end_date} · {bt_session}</span></div>',
        unsafe_allow_html=True,
    )

    if st.button("🚀  Run Backtest", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting backtest...")
        tickers = bt_dw["tickers"]

        def update_progress(i, total, ticker):
            progress.progress((i + 1) / total, text=f"Backtesting {ticker}... ({i+1}/{total})")

        reports = backtest_session(
            tickers=tickers,
            start_date=str(bt_start_date),
            end_date=str(bt_end_date),
            interval=bt_interval,
            lookback="3mo",
            stock_mode=bt_dw["stock_mode"],
            max_hold=bt_max_hold,
            progress_callback=update_progress,
        )
        progress.empty()

        # ── Aggregate metrics across all tickers ──
        all_trades = []
        for r in reports:
            all_trades.extend(r.trades)

        total_trades = len(all_trades)
        total_wins = sum(1 for t in all_trades if t.outcome == "WIN")
        total_losses = sum(1 for t in all_trades if t.outcome == "LOSS")
        total_timeouts = sum(1 for t in all_trades if t.outcome == "TIMEOUT")
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t.pnl_pct for t in all_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # ── Summary metrics ──
        st.markdown(
            '<div style="margin:20px 0 10px;font-size:1.3rem;font-weight:700;color:#e2e8f0;">'
            '📊 Overall Performance</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Trades", total_trades)
        m2.metric("Wins", total_wins)
        m3.metric("Losses", total_losses)
        m4.metric("Timeouts", total_timeouts)
        m5.metric("Win Rate", f"{win_rate:.1f}%")
        m6.metric("Total P&L", f"{total_pnl:+.2f}%")

        if total_trades > 0:
            # ── Equity curve ──
            st.markdown(
                '<div style="margin:30px 0 10px;font-size:1.3rem;font-weight:700;color:#e2e8f0;">'
                '📈 Equity Curve (cumulative P&L)</div>',
                unsafe_allow_html=True,
            )
            sorted_trades = sorted(all_trades, key=lambda t: t.date)
            equity = [0.0]
            labels = ["Start"]
            for t in sorted_trades:
                equity.append(equity[-1] + t.pnl_pct)
                labels.append(f"{t.ticker} ({t.date})")

            import plotly.graph_objects as go
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                x=labels, y=equity,
                mode="lines+markers",
                line=dict(color="#4ade80" if equity[-1] >= 0 else "#f87171", width=2),
                marker=dict(size=6),
                fill="tozeroy",
                fillcolor="rgba(74,222,128,0.1)" if equity[-1] >= 0 else "rgba(248,113,113,0.1)",
            ))
            eq_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0e17",
                plot_bgcolor="#0a0e17",
                height=350,
                margin=dict(l=40, r=20, t=20, b=60),
                xaxis=dict(showgrid=False, tickangle=-45),
                yaxis=dict(title="Cumulative P&L (%)", gridcolor="#1e293b"),
            )
            st.plotly_chart(eq_fig, use_container_width=True, config={"displayModeBar": False})

            # ── Per-ticker breakdown ──
            st.markdown(
                '<div style="margin:30px 0 10px;font-size:1.3rem;font-weight:700;color:#e2e8f0;">'
                '🔍 Per-Ticker Breakdown</div>',
                unsafe_allow_html=True,
            )

            for r in reports:
                if r.total_trades == 0:
                    continue
                with st.expander(
                    f"{'✅' if r.win_rate >= 50 else '⚠️'} {r.ticker} — "
                    f"{r.total_trades} trades · {r.win_rate:.0f}% win · "
                    f"{r.total_pnl_pct:+.2f}% P&L"
                ):
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("Trades", r.total_trades)
                    rc2.metric("Win Rate", f"{r.win_rate:.0f}%")
                    rc3.metric("Avg P&L", f"{r.avg_pnl_pct:+.2f}%")
                    rc4.metric("Best / Worst", f"{r.best_trade_pnl:+.1f}% / {r.worst_trade_pnl:+.1f}%")

                    # Trade table
                    if r.trades:
                        trade_data = []
                        for t in r.trades:
                            outcome_emoji = {"WIN": "✅", "LOSS": "❌", "TIMEOUT": "⏰", "OPEN": "🔵"}.get(t.outcome, "")
                            trade_data.append({
                                "Date": t.date,
                                "Action": t.action,
                                "Entry": f"${t.entry_price:.4f}",
                                "SL": f"${t.stop_loss:.4f}",
                                "TP": f"${t.take_profit:.4f}",
                                "Exit": f"${t.exit_price:.4f}" if t.exit_price > 0 else "—",
                                "Outcome": f"{outcome_emoji} {t.outcome}",
                                "P&L": f"{t.pnl_pct:+.2f}%",
                                "R:R": f"{t.actual_rr:+.1f}",
                                "Bars": t.bars_held,
                                "Confidence": t.confidence,
                                "Score": t.score,
                            })
                        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)

            # ── All trades table ──
            st.markdown(
                '<div style="margin:30px 0 10px;font-size:1.3rem;font-weight:700;color:#e2e8f0;">'
                '📋 All Trades</div>',
                unsafe_allow_html=True,
            )
            if all_trades:
                all_data = []
                for t in sorted_trades:
                    outcome_emoji = {"WIN": "✅", "LOSS": "❌", "TIMEOUT": "⏰", "OPEN": "🔵"}.get(t.outcome, "")
                    all_data.append({
                        "Ticker": t.ticker,
                        "Date": t.date,
                        "Action": t.action,
                        "Entry": f"${t.entry_price:.4f}",
                        "SL": f"${t.stop_loss:.4f}",
                        "TP": f"${t.take_profit:.4f}",
                        "Exit": f"${t.exit_price:.4f}" if t.exit_price > 0 else "—",
                        "Outcome": f"{outcome_emoji} {t.outcome}",
                        "P&L": f"{t.pnl_pct:+.2f}%",
                        "Bars": t.bars_held,
                        "Confidence": t.confidence,
                    })
                st.dataframe(pd.DataFrame(all_data), use_container_width=True, hide_index=True)

        elif total_trades == 0:
            st.warning(
                "No actionable trades found in this date range. "
                "The strategy may not have generated BUY/SELL signals for these tickers "
                "during this period. Try a wider date range or different session."
            )

    else:
        st.markdown(
            '<div style="text-align:center;padding:60px 20px;color:#64748b;">'
            '<div style="font-size:3rem;margin-bottom:16px;">🧪</div>'
            '<div style="font-size:1.2rem;font-weight:600;color:#94a3b8;">Configure your backtest and click Run</div>'
            '<div style="font-size:0.85rem;margin-top:8px;">'
            'The strategy will run day-by-day on historical data and simulate trade outcomes</div>'
            '<div style="font-size:0.78rem;margin-top:16px;color:#64748b;">'
            'For each signal: walks forward bar-by-bar to check if TP or SL is hit first</div></div>',
            unsafe_allow_html=True,
        )
