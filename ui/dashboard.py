"""
Stock Trader Dashboard — Streamlit UI
======================================
Interactive web dashboard for ICT / Smart Money Concepts analysis.

Run with:  streamlit run ui/dashboard.py
"""

import sys
import os

# Ensure the project root is on the path so our modules resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
import config

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

/* ── Hide Streamlit branding ────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str, interval: str):
    """Cached data fetcher."""
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

    st.markdown('<div class="section-header">Analysis Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Mode",
        ["Single Ticker", "Multi-Ticker Scanner"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-header">Ticker</div>', unsafe_allow_html=True)

    if mode == "Single Ticker":
        ticker = st.text_input(
            "Ticker Symbol",
            value=config.DEFAULT_TICKER,
            label_visibility="collapsed",
            placeholder="e.g. AAPL, SPY, TSLA",
        ).upper().strip()
    else:
        tickers_input = st.text_input(
            "Tickers (comma-separated)",
            value="AAPL, MSFT, TSLA, GOOGL, AMZN, META, NVDA, SPY",
            label_visibility="collapsed",
        )
        tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.markdown('<div class="section-header">Timeframe</div>', unsafe_allow_html=True)

    col_p, col_i = st.columns(2)
    with col_p:
        period = st.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=2,
            label_visibility="collapsed",
        )
    with col_i:
        interval = st.selectbox(
            "Interval",
            ["5m", "15m", "1h", "1d", "1wk"],
            index=3,
            label_visibility="collapsed",
        )

    st.markdown('<div class="section-header">Chart Overlays</div>', unsafe_allow_html=True)

    show_structure = st.toggle("Market Structure", value=True)
    show_obs = st.toggle("Order Blocks", value=True)
    show_fvgs = st.toggle("Fair Value Gaps", value=True)
    show_liq = st.toggle("Liquidity Levels", value=True)
    show_pd = st.toggle("Premium / Discount", value=True)
    show_trade = st.toggle("Trade Levels (SL/TP)", value=True)

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

        st.markdown(
            f'<div style="display:flex; align-items:baseline; gap:16px; margin-bottom:4px;">'
            f'<span style="font-size:2rem; font-weight:800; color:#e2e8f0;">{ticker}</span>'
            f'<span style="font-size:1.8rem; font-weight:700; color:#e2e8f0;">'
            f'${current_price:.2f}</span>'
            f'<span style="font-size:1rem; font-weight:600; color:{change_color};">'
            f'{change_sign}{price_change:.2f} ({change_sign}{price_change_pct:.2f}%)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="display:flex; gap:10px; align-items:center;">'
            f'<span class="bias-badge {get_bias_class(setup.bias)}">'
            f'{setup.bias.value.upper()} BIAS</span>'
            f'<span style="color:#64748b; font-size:0.78rem;">'
            f'{len(df)} candles &middot; {period} &middot; {interval}</span>'
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
        m1.metric("Entry", f"${setup.entry_price:.2f}")
        m2.metric("Stop Loss", f"${setup.stop_loss:.2f}")
        m3.metric("Take Profit", f"${setup.take_profit:.2f}")
        m4.metric("R : R", f"1 : {setup.risk_reward:.1f}")
        m5.metric("Position", f"{setup.position_size} shares")
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
            st.markdown(
                '<div class="info-box">'
                '<h4>Trade Setup</h4>'
                f'<div class="info-row"><span class="info-label">Action</span>'
                f'<span class="info-value">{setup.action.value}</span></div>'
                f'<div class="info-row"><span class="info-label">Entry Price</span>'
                f'<span class="info-value">${setup.entry_price:.2f}</span></div>'
                f'<div class="info-row"><span class="info-label">Stop Loss</span>'
                f'<span class="info-value">${setup.stop_loss:.2f}</span></div>'
                f'<div class="info-row"><span class="info-label">Take Profit</span>'
                f'<span class="info-value">${setup.take_profit:.2f}</span></div>'
                f'<div class="info-row"><span class="info-label">Risk per Share</span>'
                f'<span class="info-value">${setup.risk_per_share:.2f}</span></div>'
                f'<div class="info-row"><span class="info-label">Reward per Share</span>'
                f'<span class="info-value">${setup.reward_per_share:.2f}</span></div>'
                f'<div class="info-row"><span class="info-label">Risk : Reward</span>'
                f'<span class="info-value">1 : {setup.risk_reward:.1f}</span></div>'
                f'<div class="info-row"><span class="info-label">Position Size</span>'
                f'<span class="info-value">{setup.position_size} shares</span></div>'
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
            st.dataframe(
                df.tail(50).style.format({
                    "Open": "${:.2f}",
                    "High": "${:.2f}",
                    "Low": "${:.2f}",
                    "Close": "${:.2f}",
                    "Volume": "{:,.0f}",
                }),
                use_container_width=True,
            )


# ──────────────────────────────────────────────────────────
#  Main Content — Multi-Ticker Scanner
# ──────────────────────────────────────────────────────────

elif mode == "Multi-Ticker Scanner":
    st.markdown(
        '<div style="margin-bottom:20px;">'
        '<span style="font-size:1.8rem; font-weight:800; color:#e2e8f0;">Multi-Ticker Scanner</span>'
        '<span style="color:#64748b; font-size:0.85rem; margin-left:16px;">'
        f'{len(tickers_list)} tickers &middot; {period} &middot; {interval}</span>'
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
                f'${current:.2f} '
                f'<span style="color:{chg_color};">{chg_sign}{chg:.2f}%</span>'
                f' &middot; SL ${s.stop_loss:.2f} &middot; TP ${s.take_profit:.2f}'
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
