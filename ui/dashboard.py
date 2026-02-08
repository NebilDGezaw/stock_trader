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
if "config" in sys.modules:
    del sys.modules["config"]

import config

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
#  Minimal CSS (no hiding of header/sidebar elements)
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
#  Constants
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

def run_analysis(df, ticker):
    return SMCStrategy(df, ticker=ticker).run()


# ══════════════════════════════════════════════════════════
#  SIDEBAR — all native Streamlit widgets, no custom HTML
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📊 Stock Trader")
    st.caption("ICT & Smart Money Concepts")

    # ── Asset Class ───────────────────────────────
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

    # ── Mode ──────────────────────────────────────
    st.subheader("Mode", divider="gray")
    mode = st.radio(
        "Analysis Mode",
        ["Single Ticker", "Multi-Ticker Scanner"],
        label_visibility="collapsed",
        key="mode_selector",
    )

    # ── Ticker ────────────────────────────────────
    st.subheader("Ticker", divider="gray")

    placeholders = {
        "Stocks": "e.g. AAPL, SPY, TSLA",
        "Crypto": "e.g. BTC-USD, ETH-USD",
        "Forex":  "e.g. EURUSD=X, GBPUSD=X",
        "Commodities": "e.g. GC=F, CL=F, SI=F",
    }

    if mode == "Single Ticker":
        # Text input for custom ticker
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
    else:
        # Scanner mode
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

    # ── Timeframe ─────────────────────────────────
    st.subheader("Timeframe", divider="gray")

    all_intervals = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"]
    interval = st.selectbox(
        "Interval",
        all_intervals,
        index=5,  # default "1d"
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
        st.info(f"⏱️ Max lookback: **{PERIOD_LABELS.get(valid_periods[-1], valid_periods[-1])}** for {INTERVAL_LABELS[interval]} candles.", icon="ℹ️")

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
#  MAIN CONTENT — Single Ticker
# ══════════════════════════════════════════════════════════

if mode == "Single Ticker":
    try:
        with st.spinner(f"Fetching {ticker} data..."):
            df = fetch_data(ticker, period, interval)
            strategy = run_analysis(df, ticker)
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

        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px;">'
            f'<span style="font-size:2rem;font-weight:800;color:#e2e8f0;">{ticker}</span>'
            f'<span style="font-size:1.8rem;font-weight:700;color:#e2e8f0;">{fmt_price(price, currency_sym)}</span>'
            f'<span style="font-size:1rem;font-weight:600;color:{color};">{sign}{chg:.2f} ({sign}{chg_pct:.2f}%)</span>'
            f'<span style="font-size:0.75rem;color:#64748b;background:#1e293b;padding:3px 10px;border-radius:5px;">'
            f'{ac_icon} {asset_class}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:center;">'
            f'<span class="bias-badge {bias_class(setup.bias)}">{setup.bias.value.upper()} BIAS</span>'
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
                        (f"Risk per {_u.title()}", fmt_price(setup.risk_per_share, currency_sym)),
                        (f"Reward per {_u.title()}", fmt_price(setup.reward_per_share, currency_sym)),
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
#  MAIN CONTENT — Multi-Ticker Scanner
# ══════════════════════════════════════════════════════════

elif mode == "Multi-Ticker Scanner":
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
        results = []
        progress = st.progress(0, text="Scanning tickers...")

        for i, t in enumerate(tickers_list):
            progress.progress((i + 1) / len(tickers_list), text=f"Analyzing {t}... ({i+1}/{len(tickers_list)})")
            try:
                data = fetch_data(t, period, interval)
                strat = run_analysis(data, t)
                results.append({"ticker": t, "setup": strat.trade_setup, "strategy": strat, "df": data})
            except Exception as e:
                st.warning(f"Skipped **{t}**: {e}")

        progress.empty()

        if not results:
            st.error("No results. Check your ticker symbols.")
            st.stop()

        results.sort(key=lambda r: abs(r["setup"].composite_score), reverse=True)

        actionable = [r for r in results if r["setup"].action.value in ("STRONG BUY", "BUY", "SELL", "STRONG SELL")]
        buys = [r for r in actionable if "BUY" in r["setup"].action.value]
        sells = [r for r in actionable if "SELL" in r["setup"].action.value]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scanned", len(results))
        m2.metric("Actionable", len(actionable))
        m3.metric("Buy Signals", len(buys))
        m4.metric("Sell Signals", len(sells))

        st.markdown("")

        for r in results:
            s = r["setup"]
            abg = {"action-strong-buy": "background:rgba(16,185,129,0.2);color:#34d399;",
                   "action-buy": "background:rgba(34,197,94,0.2);color:#4ade80;",
                   "action-hold": "background:rgba(234,179,8,0.2);color:#fbbf24;",
                   "action-sell": "background:rgba(239,68,68,0.2);color:#f87171;",
                   "action-strong-sell": "background:rgba(220,38,38,0.2);color:#fca5a5;"}.get(action_class(s.action), "")

            cur = r["df"].iloc[-1]["Close"]
            prv = r["df"].iloc[-2]["Close"] if len(r["df"]) > 1 else cur
            c = ((cur - prv) / prv) * 100
            cc = "#4ade80" if c >= 0 else "#f87171"
            cs = "+" if c >= 0 else ""

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
