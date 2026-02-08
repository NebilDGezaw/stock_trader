"""
Stock Trader Dashboard — Streamlit UI
======================================
Interactive web dashboard for ICT / Smart Money Concepts analysis.

Run with:  streamlit run ui/dashboard.py
"""

import sys
import os

# Ensure the project root is on the path so our modules resolve.
# Must happen before ANY project imports.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Force-reload our config in case a stale/different 'config' was cached
if "config" in sys.modules:
    del sys.modules["config"]

import config   # noqa: E402  — must import after path setup

import streamlit as st
import pandas as pd
from datetime import datetime

from data.fetcher import StockDataFetcher
from strategies.smc_strategy import SMCStrategy
from models.signals import TradeAction, MarketBias, SignalType
from trader.decision_engine import DecisionEngine
from ui.charts import (
    build_main_chart,
    build_score_gauge,
    build_signal_breakdown,
)

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
#  Custom CSS
# ──────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.stApp {
    background: #0a0e17;
}

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0f1420;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0;
}

/* ── Metric cards ───────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #131a2b 0%, #0f1420 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
}

div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}

/* ── Action badge ───────────────────────────────────── */
.action-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1.3rem;
    letter-spacing: 0.03em;
    text-align: center;
    margin: 4px 0;
}
.action-strong-buy  { background: linear-gradient(135deg, #059669, #10b981); color: #fff; }
.action-buy         { background: linear-gradient(135deg, #16a34a, #22c55e); color: #fff; }
.action-hold        { background: linear-gradient(135deg, #ca8a04, #eab308); color: #1a1a1a; }
.action-sell        { background: linear-gradient(135deg, #dc2626, #ef4444); color: #fff; }
.action-strong-sell { background: linear-gradient(135deg, #991b1b, #dc2626); color: #fff; }

/* ── Bias badge ─────────────────────────────────────── */
.bias-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
}
.bias-bullish { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.bias-bearish { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.bias-neutral { background: rgba(234,179,8,0.15); color: #fbbf24; border: 1px solid rgba(234,179,8,0.3); }

/* ── Signal card ────────────────────────────────────── */
.signal-card {
    background: #131a2b;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: border-color 0.2s;
}
.signal-card:hover {
    border-color: #334155;
}
.signal-icon {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.signal-icon.bullish { background: rgba(34,197,94,0.15); color: #4ade80; }
.signal-icon.bearish { background: rgba(239,68,68,0.15); color: #f87171; }
.signal-icon.neutral { background: rgba(234,179,8,0.15); color: #fbbf24; }
.signal-body { flex: 1; min-width: 0; }
.signal-type { font-size: 0.78rem; font-weight: 600; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.04em; }
.signal-detail { font-size: 0.75rem; color: #64748b; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.signal-score { font-size: 0.85rem; font-weight: 700; color: #e2e8f0; background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 6px; }

/* ── Info box ───────────────────────────────────────── */
.info-box {
    background: linear-gradient(135deg, #131a2b 0%, #0f1420 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.info-box h4 {
    margin: 0 0 8px 0;
    color: #e2e8f0;
    font-size: 0.9rem;
    font-weight: 600;
}
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1e293b;
    font-size: 0.82rem;
}
.info-row:last-child { border-bottom: none; }
.info-label { color: #94a3b8; }
.info-value { color: #e2e8f0; font-weight: 500; }

/* ── Section header ─────────────────────────────────── */
.section-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}

/* ── Scanner table ──────────────────────────────────── */
.scanner-row {
    background: #131a2b;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: grid;
    grid-template-columns: 80px 120px 100px 80px 1fr;
    align-items: center;
    gap: 12px;
    transition: border-color 0.2s, transform 0.15s;
}
.scanner-row:hover {
    border-color: #334155;
    transform: translateY(-1px);
}
.scanner-ticker {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
}
.scanner-action {
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    text-align: center;
}
.scanner-score {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    text-align: center;
}
.scanner-details {
    font-size: 0.78rem;
    color: #94a3b8;
}

/* ── Hide Streamlit branding (keep header for sidebar toggle) ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] {
    background: rgba(10, 14, 23, 0.85);
    backdrop-filter: blur(8px);
}
[data-testid="stDecoration"] { display: none; }

/* ── Tab styling ────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #0f1420;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  Helper functions
# ──────────────────────────────────────────────────────────

# Valid period options per interval (yfinance limits)
_PERIODS_FOR_INTERVAL = {
    "1m":  ["1d", "5d", "7d"],
    "2m":  ["1d", "5d", "7d", "1mo", "60d"],
    "5m":  ["1d", "5d", "7d", "1mo", "60d"],
    "15m": ["1d", "5d", "7d", "1mo", "60d"],
    "30m": ["1d", "5d", "7d", "1mo", "60d"],
    "1h":  ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d":  ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    "1wk": ["3mo", "6mo", "1y", "2y", "5y"],
}

_PERIOD_LABELS = {
    "1d": "1 Day", "5d": "5 Days", "7d": "7 Days",
    "1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months", "60d": "60 Days",
    "1y": "1 Year", "2y": "2 Years", "5y": "5 Years",
}

_INTERVAL_LABELS = {
    "1m": "1 min", "5m": "5 min", "15m": "15 min", "30m": "30 min",
    "1h": "1 hour", "1d": "1 day", "1wk": "1 week",
}


def get_action_class(action: TradeAction) -> str:
    return {
        TradeAction.STRONG_BUY:  "action-strong-buy",
        TradeAction.BUY:         "action-buy",
        TradeAction.HOLD:        "action-hold",
        TradeAction.SELL:        "action-sell",
        TradeAction.STRONG_SELL: "action-strong-sell",
    }.get(action, "action-hold")


def get_bias_class(bias: MarketBias) -> str:
    return {
        MarketBias.BULLISH: "bias-bullish",
        MarketBias.BEARISH: "bias-bearish",
        MarketBias.NEUTRAL: "bias-neutral",
    }.get(bias, "bias-neutral")


def signal_icon_html(sig) -> str:
    if sig.bias == MarketBias.BULLISH:
        return '<div class="signal-icon bullish">▲</div>'
    elif sig.bias == MarketBias.BEARISH:
        return '<div class="signal-icon bearish">▼</div>'
    return '<div class="signal-icon neutral">●</div>'


def format_price(price: float, currency_symbol: str = "$") -> str:
    """Format price with appropriate decimal places."""
    if price >= 1000:
        return f"{currency_symbol}{price:,.2f}"
    elif price >= 1:
        return f"{currency_symbol}{price:.2f}"
    elif price >= 0.01:
        return f"{currency_symbol}{price:.4f}"
    else:
        return f"{currency_symbol}{price:.6f}"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str, interval: str):
    """Cached data fetcher — period is auto-clamped inside StockDataFetcher."""
    fetcher = StockDataFetcher(ticker)
    return fetcher.fetch(period=period, interval=interval)


def run_analysis(df, ticker):
    """Run the SMC strategy on the data."""
    strategy = SMCStrategy(df, ticker=ticker).run()
    return strategy


# ──────────────────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 📊 Stock Trader")
    st.markdown(
        '<p style="color:#64748b; font-size:0.8rem; margin-top:-10px;">'
        'ICT & Smart Money Concepts</p>',
        unsafe_allow_html=True,
    )

    # ── Asset Class ───────────────────────────────────
    st.markdown('<div class="section-header">Asset Class</div>', unsafe_allow_html=True)

    asset_class_names = list(config.ASSET_CLASSES.keys())
    asset_class = st.radio(
        "Asset Class",
        asset_class_names,
        horizontal=True,
        label_visibility="collapsed",
    )
    ac = config.ASSET_CLASSES[asset_class]
    currency_sym = ac["currency_symbol"]
    position_unit = ac["unit"]

    # ── Analysis Mode ─────────────────────────────────
    st.markdown('<div class="section-header">Analysis Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Mode",
        ["Single Ticker", "Multi-Ticker Scanner"],
        label_visibility="collapsed",
    )

    # ── Ticker Input ──────────────────────────────────
    st.markdown('<div class="section-header">Ticker</div>', unsafe_allow_html=True)

    # Placeholder text per asset class
    _placeholders = {
        "Stocks": "e.g. AAPL, SPY, TSLA",
        "Crypto": "e.g. BTC-USD, ETH-USD",
        "Forex":  "e.g. EURUSD=X, GBPUSD=X",
    }

    if mode == "Single Ticker":
        ticker = st.text_input(
            "Ticker Symbol",
            value=ac["default_ticker"],
            label_visibility="collapsed",
            placeholder=_placeholders.get(asset_class, ""),
            key=f"ticker_input_{asset_class}",
        ).upper().strip()

        # Quick-pick preset buttons
        preset_names = list(ac["presets"].keys())
        if preset_names:
            selected_preset = st.selectbox(
                "Quick pick",
                ["— Custom —"] + preset_names,
                label_visibility="collapsed",
                key=f"preset_group_{asset_class}",
            )
            if selected_preset != "— Custom —":
                preset_tickers = ac["presets"][selected_preset]
                quick_pick = st.selectbox(
                    "Select ticker",
                    preset_tickers,
                    label_visibility="collapsed",
                    key=f"preset_ticker_{asset_class}_{selected_preset}",
                )
                ticker = quick_pick
    else:
        # Scanner: let user pick a preset group or type custom
        preset_names = list(ac["presets"].keys())
        scanner_source = st.selectbox(
            "Watchlist",
            ["Custom"] + preset_names,
            label_visibility="collapsed",
            key=f"scanner_source_{asset_class}",
        )
        if scanner_source == "Custom":
            default_scan = ", ".join(ac["presets"][preset_names[0]]) if preset_names else ""
            tickers_input = st.text_input(
                "Tickers (comma-separated)",
                value=default_scan,
                label_visibility="collapsed",
                key=f"scanner_input_{asset_class}",
            )
            tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        else:
            tickers_list = ac["presets"][scanner_source]
            st.markdown(
                f'<p style="color:#94a3b8; font-size:0.78rem;">'
                f'{", ".join(tickers_list)}</p>',
                unsafe_allow_html=True,
            )

    # ── Timeframe ─────────────────────────────────────
    st.markdown('<div class="section-header">Timeframe</div>', unsafe_allow_html=True)

    all_intervals = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"]
    col_i, col_p = st.columns(2)

    with col_i:
        interval = st.selectbox(
            "Interval",
            all_intervals,
            index=all_intervals.index("1d"),
            format_func=lambda x: _INTERVAL_LABELS.get(x, x),
            label_visibility="collapsed",
        )

    # Only show valid periods for the chosen interval
    valid_periods = _PERIODS_FOR_INTERVAL.get(interval, ["1mo", "3mo", "6mo", "1y"])
    with col_p:
        period = st.selectbox(
            "Period",
            valid_periods,
            index=min(len(valid_periods) - 1, 2),  # pick a middle default
            format_func=lambda x: _PERIOD_LABELS.get(x, x),
            label_visibility="collapsed",
        )

    # Show a tip for intraday
    if interval in ("1m", "5m", "15m", "30m"):
        st.caption(f"Intraday intervals limited to max {valid_periods[-1]} by data provider.")

    # ── Chart Overlays ────────────────────────────────
    st.markdown('<div class="section-header">Chart Overlays</div>', unsafe_allow_html=True)

    show_structure = st.toggle("Market Structure", value=True)
    show_obs = st.toggle("Order Blocks", value=True)
    show_fvgs = st.toggle("Fair Value Gaps", value=True)
    show_liq = st.toggle("Liquidity Levels", value=True)
    show_pd = st.toggle("Premium / Discount", value=True)
    show_trade = st.toggle("Trade Levels (SL/TP)", value=True)

    # ── Risk Settings ─────────────────────────────────
    st.markdown('<div class="section-header">Risk Settings</div>', unsafe_allow_html=True)

    capital = st.number_input(
        "Capital ($)", value=config.INITIAL_CAPITAL, step=10000,
        label_visibility="collapsed",
    )
    risk_pct = st.slider("Risk per trade (%)", 0.5, 5.0, 2.0, 0.5)

    st.markdown("---")
    st.markdown(
        '<p style="color:#475569; font-size:0.7rem; text-align:center;">'
        'For educational purposes only.<br>Not financial advice.</p>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────
#  Main Content — Single Ticker
# ──────────────────────────────────────────────────────────

if mode == "Single Ticker":
    # ── Header ────────────────────────────────────────
    try:
        with st.spinner(f"Fetching {ticker} data..."):
            df = fetch_data(ticker, period, interval)
            strategy = run_analysis(df, ticker)
            setup = strategy.trade_setup
    except Exception as e:
        st.error(f"Could not fetch data for **{ticker}**: {e}")
        st.stop()

    # Top row: ticker name + action badge + bias
    top_left, top_right = st.columns([3, 1])

    with top_left:
        # Price info
        current_price = df.iloc[-1]["Close"]
        prev_price = df.iloc[-2]["Close"] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price else 0
        change_color = "#22c55e" if price_change >= 0 else "#ef4444"
        change_sign = "+" if price_change >= 0 else ""

        # Asset class badge
        _ac_badge = {
            "Stocks": ("🏛️", "Stocks"),
            "Crypto": ("₿", "Crypto"),
            "Forex":  ("💱", "Forex"),
        }
        ac_icon, ac_label = _ac_badge.get(asset_class, ("📊", asset_class))

        st.markdown(
            f'<div style="display:flex; align-items:baseline; gap:16px; margin-bottom:4px;">'
            f'<span style="font-size:2rem; font-weight:800; color:#e2e8f0;">{ticker}</span>'
            f'<span style="font-size:1.8rem; font-weight:700; color:#e2e8f0;">'
            f'{format_price(current_price, currency_sym)}</span>'
            f'<span style="font-size:1rem; font-weight:600; color:{change_color};">'
            f'{change_sign}{price_change:.2f} ({change_sign}{price_change_pct:.2f}%)</span>'
            f'<span style="font-size:0.75rem; color:#64748b; background:#1e293b; '
            f'padding:3px 10px; border-radius:5px; margin-left:4px;">'
            f'{ac_icon} {ac_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="display:flex; gap:10px; align-items:center;">'
            f'<span class="bias-badge {get_bias_class(setup.bias)}">'
            f'{setup.bias.value.upper()} BIAS</span>'
            f'<span style="color:#64748b; font-size:0.78rem;">'
            f'{len(df)} candles &middot; {_PERIOD_LABELS.get(period, period)} &middot; '
            f'{_INTERVAL_LABELS.get(interval, interval)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with top_right:
        st.markdown(
            f'<div style="text-align:right; padding-top:8px;">'
            f'<div class="action-badge {get_action_class(setup.action)}">'
            f'{setup.action.value}</div>'
            f'<div style="color:#64748b; font-size:0.75rem; margin-top:4px;">'
            f'Score: {setup.composite_score}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Tabs ──────────────────────────────────────────
    tab_chart, tab_signals, tab_details = st.tabs([
        "📈  Chart",
        "🔔  Signals",
        "📋  Details",
    ])

    with tab_chart:
        # Main chart
        chart = build_main_chart(
            df, strategy,
            show_order_blocks=show_obs,
            show_fvgs=show_fvgs,
            show_liquidity=show_liq,
            show_structure=show_structure,
            show_trade_levels=show_trade,
            show_premium_discount=show_pd,
            height=620,
        )
        st.plotly_chart(chart, use_container_width=True, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
        })

        # Metrics row beneath chart
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Entry", format_price(setup.entry_price, currency_sym))
        m2.metric("Stop Loss", format_price(setup.stop_loss, currency_sym))
        m3.metric("Take Profit", format_price(setup.take_profit, currency_sym))
        m4.metric("R : R", f"1 : {setup.risk_reward:.1f}")
        m5.metric("Position", f"{setup.position_size} {position_unit}")
        m6.metric("Signals", f"{len(setup.signals)}")

    with tab_signals:
        # Score gauge
        st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
        gauge = build_score_gauge(strategy.bullish_score, strategy.bearish_score)
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

        cols_breakdown = st.columns(3)
        cols_breakdown[0].metric("Bullish Score", strategy.bullish_score)
        cols_breakdown[1].metric("Bearish Score", strategy.bearish_score)
        cols_breakdown[2].metric("Net Score", strategy.net_score)

        # Signal distribution chart + signal list
        col_chart, col_list = st.columns([1, 2])

        with col_chart:
            st.markdown('<div class="section-header">Signal Distribution</div>', unsafe_allow_html=True)
            if setup.signals:
                pie = build_signal_breakdown(setup.signals)
                st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No signals detected.")

        with col_list:
            st.markdown('<div class="section-header">All Signals</div>', unsafe_allow_html=True)
            for sig in setup.signals:
                nice_type = sig.signal_type.value.replace("_", " ").title()
                st.markdown(
                    f'<div class="signal-card">'
                    f'{signal_icon_html(sig)}'
                    f'<div class="signal-body">'
                    f'<div class="signal-type">{nice_type}</div>'
                    f'<div class="signal-detail">{sig.details}</div>'
                    f'</div>'
                    f'<div class="signal-score">+{sig.score}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if not setup.signals:
                st.info("No signals detected for this ticker/timeframe.")

    with tab_details:
        col_a, col_b = st.columns(2)

        with col_a:
            # Trade setup details
            _unit_singular = position_unit.rstrip("s") if position_unit != "lots" else "lot"
            st.markdown(
                '<div class="info-box">'
                '<h4>Trade Setup</h4>'
                f'<div class="info-row"><span class="info-label">Action</span>'
                f'<span class="info-value">{setup.action.value}</span></div>'
                f'<div class="info-row"><span class="info-label">Entry Price</span>'
                f'<span class="info-value">{format_price(setup.entry_price, currency_sym)}</span></div>'
                f'<div class="info-row"><span class="info-label">Stop Loss</span>'
                f'<span class="info-value">{format_price(setup.stop_loss, currency_sym)}</span></div>'
                f'<div class="info-row"><span class="info-label">Take Profit</span>'
                f'<span class="info-value">{format_price(setup.take_profit, currency_sym)}</span></div>'
                f'<div class="info-row"><span class="info-label">Risk per {_unit_singular.title()}</span>'
                f'<span class="info-value">{format_price(setup.risk_per_share, currency_sym)}</span></div>'
                f'<div class="info-row"><span class="info-label">Reward per {_unit_singular.title()}</span>'
                f'<span class="info-value">{format_price(setup.reward_per_share, currency_sym)}</span></div>'
                f'<div class="info-row"><span class="info-label">Risk : Reward</span>'
                f'<span class="info-value">1 : {setup.risk_reward:.1f}</span></div>'
                f'<div class="info-row"><span class="info-label">Position Size</span>'
                f'<span class="info-value">{setup.position_size} {position_unit}</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Market structure summary
            ms = strategy.structure
            st.markdown(
                '<div class="info-box">'
                '<h4>Market Structure</h4>'
                f'<div class="info-row"><span class="info-label">Bias</span>'
                f'<span class="info-value">{ms.bias.value.upper()}</span></div>'
                f'<div class="info-row"><span class="info-label">Swing Highs</span>'
                f'<span class="info-value">{len(ms.swing_highs)}</span></div>'
                f'<div class="info-row"><span class="info-label">Swing Lows</span>'
                f'<span class="info-value">{len(ms.swing_lows)}</span></div>'
                f'<div class="info-row"><span class="info-label">Structure Signals</span>'
                f'<span class="info-value">{len(ms.signals)}</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with col_b:
            # Order blocks
            ob = strategy.ob_detector
            active_obs = ob.active_blocks()
            st.markdown(
                '<div class="info-box">'
                '<h4>Order Blocks</h4>'
                f'<div class="info-row"><span class="info-label">Total Found</span>'
                f'<span class="info-value">{len(ob.order_blocks)}</span></div>'
                f'<div class="info-row"><span class="info-label">Active (Unmitigated)</span>'
                f'<span class="info-value">{len(active_obs)}</span></div>'
                f'<div class="info-row"><span class="info-label">Bullish OBs</span>'
                f'<span class="info-value">'
                f'{sum(1 for o in ob.order_blocks if o.ob_type == "bullish")}</span></div>'
                f'<div class="info-row"><span class="info-label">Bearish OBs</span>'
                f'<span class="info-value">'
                f'{sum(1 for o in ob.order_blocks if o.ob_type == "bearish")}</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # FVGs
            fvg = strategy.fvg_detector
            active_fvgs = fvg.active_fvgs()
            st.markdown(
                '<div class="info-box">'
                '<h4>Fair Value Gaps</h4>'
                f'<div class="info-row"><span class="info-label">Total Found</span>'
                f'<span class="info-value">{len(fvg.fvgs)}</span></div>'
                f'<div class="info-row"><span class="info-label">Active (Unfilled)</span>'
                f'<span class="info-value">{len(active_fvgs)}</span></div>'
                f'<div class="info-row"><span class="info-label">Bullish FVGs</span>'
                f'<span class="info-value">'
                f'{sum(1 for f in fvg.fvgs if f.fvg_type == "bullish")}</span></div>'
                f'<div class="info-row"><span class="info-label">Bearish FVGs</span>'
                f'<span class="info-value">'
                f'{sum(1 for f in fvg.fvgs if f.fvg_type == "bearish")}</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Liquidity
            liq = strategy.liq_analyzer
            st.markdown(
                '<div class="info-box">'
                '<h4>Liquidity Analysis</h4>'
                f'<div class="info-row"><span class="info-label">Liquidity Levels</span>'
                f'<span class="info-value">{len(liq.levels)}</span></div>'
                f'<div class="info-row"><span class="info-label">Equal Highs</span>'
                f'<span class="info-value">'
                f'{sum(1 for l in liq.levels if l.level_type == "equal_highs")}</span></div>'
                f'<div class="info-row"><span class="info-label">Equal Lows</span>'
                f'<span class="info-value">'
                f'{sum(1 for l in liq.levels if l.level_type == "equal_lows")}</span></div>'
                f'<div class="info-row"><span class="info-label">Sweeps Detected</span>'
                f'<span class="info-value">'
                f'{sum(1 for l in liq.levels if l.swept)}</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Raw data expander
        with st.expander("View Raw OHLCV Data"):
            # Determine decimal places based on price magnitude
            _sample = df.iloc[-1]["Close"]
            _decimals = 2 if _sample >= 1 else (4 if _sample >= 0.01 else 6)
            _pfmt = f"{currency_sym}{{:.{_decimals}f}}"
            st.dataframe(
                df.tail(50).style.format({
                    "Open": _pfmt,
                    "High": _pfmt,
                    "Low": _pfmt,
                    "Close": _pfmt,
                    "Volume": "{:,.0f}",
                }),
                use_container_width=True,
            )


# ──────────────────────────────────────────────────────────
#  Main Content — Multi-Ticker Scanner
# ──────────────────────────────────────────────────────────

elif mode == "Multi-Ticker Scanner":
    _ac_icon_map = {"Stocks": "🏛️", "Crypto": "₿", "Forex": "💱"}
    st.markdown(
        '<div style="margin-bottom:20px;">'
        f'<span style="font-size:1.8rem; font-weight:800; color:#e2e8f0;">'
        f'{_ac_icon_map.get(asset_class, "📊")} {asset_class} Scanner</span>'
        '<span style="color:#64748b; font-size:0.85rem; margin-left:16px;">'
        f'{len(tickers_list)} tickers &middot; '
        f'{_PERIOD_LABELS.get(period, period)} &middot; '
        f'{_INTERVAL_LABELS.get(interval, interval)}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    run_scan = st.button("🔍  Run Scanner", type="primary", use_container_width=True)

    if run_scan:
        results = []
        progress = st.progress(0, text="Scanning tickers...")

        for i, t in enumerate(tickers_list):
            progress.progress(
                (i + 1) / len(tickers_list),
                text=f"Analyzing {t}... ({i+1}/{len(tickers_list)})",
            )
            try:
                data = fetch_data(t, period, interval)
                strat = run_analysis(data, t)
                s = strat.trade_setup
                results.append({
                    "ticker": t,
                    "setup": s,
                    "strategy": strat,
                    "df": data,
                })
            except Exception as e:
                st.warning(f"Skipped **{t}**: {e}")

        progress.empty()

        if not results:
            st.error("No results. Check your ticker symbols.")
            st.stop()

        # Sort by absolute score
        results.sort(key=lambda r: abs(r["setup"].composite_score), reverse=True)

        # Summary metrics
        actionable = [r for r in results if r["setup"].action in (
            TradeAction.STRONG_BUY, TradeAction.BUY,
            TradeAction.SELL, TradeAction.STRONG_SELL,
        )]
        buys = [r for r in actionable if r["setup"].action in (TradeAction.BUY, TradeAction.STRONG_BUY)]
        sells = [r for r in actionable if r["setup"].action in (TradeAction.SELL, TradeAction.STRONG_SELL)]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tickers Scanned", len(results))
        m2.metric("Actionable", len(actionable))
        m3.metric("Buy Signals", len(buys))
        m4.metric("Sell Signals", len(sells))

        st.markdown("")

        # Results table
        for r in results:
            s = r["setup"]
            action_cls = get_action_class(s.action)
            bias_cls = get_bias_class(s.bias)

            # Determine action badge color
            action_bg = {
                "action-strong-buy":  "background:rgba(16,185,129,0.2); color:#34d399;",
                "action-buy":         "background:rgba(34,197,94,0.2); color:#4ade80;",
                "action-hold":        "background:rgba(234,179,8,0.2); color:#fbbf24;",
                "action-sell":        "background:rgba(239,68,68,0.2); color:#f87171;",
                "action-strong-sell": "background:rgba(220,38,38,0.2); color:#fca5a5;",
            }.get(action_cls, "")

            current = r["df"].iloc[-1]["Close"]
            prev = r["df"].iloc[-2]["Close"] if len(r["df"]) > 1 else current
            chg = ((current - prev) / prev) * 100
            chg_color = "#4ade80" if chg >= 0 else "#f87171"
            chg_sign = "+" if chg >= 0 else ""

            st.markdown(
                f'<div class="scanner-row">'
                f'<div class="scanner-ticker">{s.ticker}</div>'
                f'<div class="scanner-action" style="{action_bg}">{s.action.value}</div>'
                f'<div><span class="bias-badge {bias_cls}">{s.bias.value.upper()}</span></div>'
                f'<div class="scanner-score">{s.composite_score}</div>'
                f'<div class="scanner-details">'
                f'{format_price(current, currency_sym)} '
                f'<span style="color:{chg_color};">{chg_sign}{chg:.2f}%</span>'
                f' &middot; SL {format_price(s.stop_loss, currency_sym)}'
                f' &middot; TP {format_price(s.take_profit, currency_sym)}'
                f' &middot; R:R 1:{s.risk_reward:.1f}'
                f' &middot; {len(s.signals)} signals'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Expandable detail for each
        st.markdown('<div class="section-header">Detailed Charts</div>', unsafe_allow_html=True)
        for r in results:
            with st.expander(f"📈 {r['ticker']} — {r['setup'].action.value} (Score: {r['setup'].composite_score})"):
                chart = build_main_chart(
                    r["df"], r["strategy"],
                    show_order_blocks=show_obs,
                    show_fvgs=show_fvgs,
                    show_liquidity=show_liq,
                    show_structure=show_structure,
                    show_trade_levels=show_trade,
                    show_premium_discount=show_pd,
                    height=480,
                )
                st.plotly_chart(chart, use_container_width=True, config={
                    "displayModeBar": True,
                    "displaylogo": False,
                })
