"""
Chart rendering module — builds interactive Plotly candlestick charts
with ICT/SMC overlays (order blocks, FVGs, liquidity levels, structure).
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional

from strategies.smc_strategy import SMCStrategy
from models.signals import MarketBias


# ── Color palette ─────────────────────────────────────────

COLORS = {
    "bg":              "#0e1117",
    "paper":           "#0e1117",
    "grid":            "#1e2533",
    "text":            "#e0e0e0",
    "text_dim":        "#6b7280",
    "candle_up":       "#22c55e",
    "candle_down":     "#ef4444",
    "candle_up_fill":  "#22c55e",
    "candle_down_fill":"#ef4444",
    "volume_up":       "rgba(34,197,94,0.25)",
    "volume_down":     "rgba(239,68,68,0.25)",
    "ob_bull":         "rgba(34,197,94,0.12)",
    "ob_bull_line":    "rgba(34,197,94,0.5)",
    "ob_bear":         "rgba(239,68,68,0.12)",
    "ob_bear_line":    "rgba(239,68,68,0.5)",
    "fvg_bull":        "rgba(59,130,246,0.10)",
    "fvg_bull_line":   "rgba(59,130,246,0.4)",
    "fvg_bear":        "rgba(249,115,22,0.10)",
    "fvg_bear_line":   "rgba(249,115,22,0.4)",
    "liq_high":        "rgba(168,85,247,0.6)",
    "liq_low":         "rgba(168,85,247,0.6)",
    "swing_high":      "#facc15",
    "swing_low":       "#38bdf8",
    "entry":           "#ffffff",
    "sl":              "#ef4444",
    "tp":              "#22c55e",
    "equilibrium":     "rgba(156,163,175,0.4)",
}


def build_main_chart(
    df: pd.DataFrame,
    strategy: SMCStrategy,
    show_order_blocks: bool = True,
    show_fvgs: bool = True,
    show_liquidity: bool = True,
    show_structure: bool = True,
    show_trade_levels: bool = True,
    show_premium_discount: bool = True,
    height: int = 700,
) -> go.Figure:
    """
    Build the full interactive candlestick chart with all SMC overlays.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
    )

    # ── Candlestick ───────────────────────────────────────
    colors_up = [COLORS["candle_up"] if c >= o else COLORS["candle_down"]
                 for o, c in zip(df["Open"], df["Close"])]

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=COLORS["candle_up"],
            decreasing_line_color=COLORS["candle_down"],
            increasing_fillcolor=COLORS["candle_up_fill"],
            decreasing_fillcolor=COLORS["candle_down_fill"],
            name="Price",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── Volume bars ───────────────────────────────────────
    vol_colors = [COLORS["volume_up"] if c >= o else COLORS["volume_down"]
                  for o, c in zip(df["Open"], df["Close"])]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=vol_colors,
            name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ── Overlays (safe for non-ICT strategies) ─────────────
    # Wrap each overlay in try/except so non-ICT strategies
    # (LeveragedMomentum, CryptoMomentum) don't crash the chart.
    try:
        if show_structure and getattr(strategy, 'structure', None):
            _add_structure_overlay(fig, df, strategy)
    except Exception:
        pass

    try:
        if show_order_blocks and getattr(strategy, 'ob_detector', None):
            _add_order_block_overlay(fig, df, strategy)
    except Exception:
        pass

    try:
        if show_fvgs and getattr(strategy, 'fvg_detector', None):
            _add_fvg_overlay(fig, df, strategy)
    except Exception:
        pass

    try:
        if show_liquidity and getattr(strategy, 'liq_analyzer', None):
            _add_liquidity_overlay(fig, df, strategy)
    except Exception:
        pass

    try:
        if show_premium_discount and getattr(strategy, 'structure', None):
            _add_premium_discount_overlay(fig, df, strategy)
    except Exception:
        pass

    try:
        if show_trade_levels and getattr(strategy, 'trade_setup', None):
            _add_trade_levels(fig, df, strategy)
    except Exception:
        pass

    # ── Layout ────────────────────────────────────────────
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["paper"],
        font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=COLORS["text_dim"]),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=12,
            font_color=COLORS["text"],
            bordercolor="#334155",
        ),
    )

    for axis in ["xaxis", "xaxis2"]:
        fig.update_layout(**{
            axis: dict(
                gridcolor=COLORS["grid"],
                showgrid=True,
                zeroline=False,
                showspikes=True,
                spikecolor="#475569",
                spikethickness=1,
                spikedash="dot",
            )
        })

    for axis in ["yaxis", "yaxis2"]:
        fig.update_layout(**{
            axis: dict(
                gridcolor=COLORS["grid"],
                showgrid=True,
                zeroline=False,
                side="right",
            )
        })

    return fig


# ── Overlay builders ──────────────────────────────────────

def _add_structure_overlay(fig, df, strategy):
    """Add swing high/low markers and BOS/CHoCH annotations."""
    ms = getattr(strategy, "structure", None)

    # Swing highs
    sh_indices = [i for i, _ in ms.swing_highs]
    sh_prices = [p for _, p in ms.swing_highs]
    if sh_indices:
        fig.add_trace(
            go.Scatter(
                x=[df.index[i] for i in sh_indices if i < len(df)],
                y=[p for i, p in zip(sh_indices, sh_prices) if i < len(df)],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=9,
                    color=COLORS["swing_high"],
                    line=dict(width=1, color=COLORS["swing_high"]),
                ),
                name="Swing High",
                hovertemplate="Swing High: $%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Swing lows
    sl_indices = [i for i, _ in ms.swing_lows]
    sl_prices = [p for _, p in ms.swing_lows]
    if sl_indices:
        fig.add_trace(
            go.Scatter(
                x=[df.index[i] for i in sl_indices if i < len(df)],
                y=[p for i, p in zip(sl_indices, sl_prices) if i < len(df)],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=9,
                    color=COLORS["swing_low"],
                    line=dict(width=1, color=COLORS["swing_low"]),
                ),
                name="Swing Low",
                hovertemplate="Swing Low: $%{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # BOS / CHoCH annotations on the last few signals
    recent_signals = ms.signals[-8:] if len(ms.signals) > 8 else ms.signals
    for sig in recent_signals:
        label = "BOS" if "break_of_structure" in sig.signal_type.value else "CHoCH"
        color = COLORS["candle_up"] if sig.bias == MarketBias.BULLISH else COLORS["candle_down"]
        fig.add_annotation(
            x=sig.timestamp,
            y=sig.price,
            text=f"<b>{label}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowwidth=1.5,
            arrowcolor=color,
            ax=0,
            ay=-30 if sig.bias == MarketBias.BULLISH else 30,
            font=dict(size=10, color=color, family="Inter, system-ui, sans-serif"),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        )


def _add_order_block_overlay(fig, df, strategy):
    """Draw rectangles for active order blocks."""
    obs = getattr(strategy, "ob_detector", None).order_blocks
    last_idx = len(df) - 1

    for ob in obs:
        if ob.formation_index >= len(df):
            continue

        is_active = not ob.mitigated
        opacity_mult = 1.0 if is_active else 0.3

        x0 = df.index[ob.formation_index]
        # Extend the block to the right (up to the end or when mitigated)
        x1_idx = min(ob.formation_index + 30, last_idx)
        x1 = df.index[x1_idx]

        if ob.ob_type == "bullish":
            fill = COLORS["ob_bull"]
            line = COLORS["ob_bull_line"]
        else:
            fill = COLORS["ob_bear"]
            line = COLORS["ob_bear_line"]

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=ob.bottom, y1=ob.top,
            fillcolor=fill,
            line=dict(color=line, width=1, dash="dot" if ob.mitigated else "solid"),
            opacity=opacity_mult,
            row=1, col=1,
        )

        # Label
        if is_active:
            fig.add_annotation(
                x=x0, y=ob.top if ob.ob_type == "bearish" else ob.bottom,
                text=f"{'Bull' if ob.ob_type == 'bullish' else 'Bear'} OB",
                showarrow=False,
                font=dict(
                    size=9,
                    color=COLORS["candle_up"] if ob.ob_type == "bullish" else COLORS["candle_down"],
                ),
                xanchor="left",
                yanchor="bottom" if ob.ob_type == "bearish" else "top",
                bgcolor="rgba(0,0,0,0.5)",
                borderpad=2,
            )


def _add_fvg_overlay(fig, df, strategy):
    """Draw rectangles for fair value gaps."""
    fvgs = getattr(strategy, "fvg_detector", None).fvgs
    last_idx = len(df) - 1

    for fvg in fvgs:
        if fvg.formation_index >= len(df):
            continue

        is_active = not fvg.filled

        x0 = df.index[fvg.formation_index]
        x1_idx = min(fvg.formation_index + 20, last_idx)
        x1 = df.index[x1_idx]

        if fvg.fvg_type == "bullish":
            fill = COLORS["fvg_bull"]
            line = COLORS["fvg_bull_line"]
        else:
            fill = COLORS["fvg_bear"]
            line = COLORS["fvg_bear_line"]

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=fvg.bottom, y1=fvg.top,
            fillcolor=fill if is_active else "rgba(0,0,0,0)",
            line=dict(
                color=line,
                width=1,
                dash="solid" if is_active else "dot",
            ),
            opacity=1.0 if is_active else 0.2,
            row=1, col=1,
        )


def _add_liquidity_overlay(fig, df, strategy):
    """Draw horizontal lines for liquidity levels."""
    levels = getattr(strategy, "liq_analyzer", None).levels

    # Only show the top N most-touched levels to avoid clutter
    sorted_levels = sorted(levels, key=lambda l: l.touch_count, reverse=True)[:10]

    for level in sorted_levels:
        dash_style = "dot"
        color = COLORS["liq_high"] if level.level_type == "equal_highs" else COLORS["liq_low"]

        if level.swept:
            dash_style = "dashdot"
            color = "rgba(168,85,247,0.25)"

        fig.add_hline(
            y=level.price,
            line=dict(color=color, width=1, dash=dash_style),
            annotation_text=(
                f"{'EQH' if level.level_type == 'equal_highs' else 'EQL'} "
                f"({level.touch_count}x)"
                f"{' SWEPT' if level.swept else ''}"
            ),
            annotation_position="left",
            annotation_font=dict(size=9, color=color),
            row=1, col=1,
        )


def _add_premium_discount_overlay(fig, df, strategy):
    """Add equilibrium line and shade premium/discount zones."""
    sh = getattr(strategy, "structure", None).get_last_swing_high()
    sl = getattr(strategy, "structure", None).get_last_swing_low()

    if not sh or not sl:
        return

    range_high = sh[1]
    range_low = sl[1]
    eq = (range_high + range_low) / 2

    # Equilibrium line
    fig.add_hline(
        y=eq,
        line=dict(color=COLORS["equilibrium"], width=1.5, dash="dash"),
        annotation_text="Equilibrium",
        annotation_position="right",
        annotation_font=dict(size=9, color=COLORS["text_dim"]),
        row=1, col=1,
    )


def _add_trade_levels(fig, df, strategy):
    """Draw entry, SL, and TP horizontal lines."""
    setup = strategy.trade_setup
    if not setup:
        return

    # Entry
    fig.add_hline(
        y=setup.entry_price,
        line=dict(color=COLORS["entry"], width=1.5, dash="dash"),
        annotation_text=f"Entry ${setup.entry_price:.2f}",
        annotation_position="right",
        annotation_font=dict(size=10, color=COLORS["entry"]),
        row=1, col=1,
    )

    # Stop Loss
    fig.add_hline(
        y=setup.stop_loss,
        line=dict(color=COLORS["sl"], width=1.5, dash="dash"),
        annotation_text=f"SL ${setup.stop_loss:.2f}",
        annotation_position="right",
        annotation_font=dict(size=10, color=COLORS["sl"]),
        row=1, col=1,
    )

    # Take Profit
    fig.add_hline(
        y=setup.take_profit,
        line=dict(color=COLORS["tp"], width=1.5, dash="dash"),
        annotation_text=f"TP ${setup.take_profit:.2f}",
        annotation_position="right",
        annotation_font=dict(size=10, color=COLORS["tp"]),
        row=1, col=1,
    )


# ── Mini score gauge chart ────────────────────────────────

def build_score_gauge(bullish: int, bearish: int) -> go.Figure:
    """Build a horizontal diverging bar to visualize bull vs bear score."""
    net = bullish - bearish
    max_score = max(bullish + bearish, 1)

    fig = go.Figure()

    # Bearish bar (left)
    fig.add_trace(go.Bar(
        y=["Score"],
        x=[-bearish],
        orientation="h",
        marker_color=COLORS["candle_down"],
        name="Bearish",
        text=[f"Bear: {bearish}"],
        textposition="inside",
        textfont=dict(color="white", size=13, family="Inter, system-ui, sans-serif"),
        hoverinfo="skip",
    ))

    # Bullish bar (right)
    fig.add_trace(go.Bar(
        y=["Score"],
        x=[bullish],
        orientation="h",
        marker_color=COLORS["candle_up"],
        name="Bullish",
        text=[f"Bull: {bullish}"],
        textposition="inside",
        textfont=dict(color="white", size=13, family="Inter, system-ui, sans-serif"),
        hoverinfo="skip",
    ))

    fig.update_layout(
        barmode="relative",
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(
            showgrid=False, zeroline=True,
            zerolinecolor="#475569", zerolinewidth=2,
            showticklabels=False, range=[-max_score * 1.2, max_score * 1.2],
        ),
        yaxis=dict(showticklabels=False),
        showlegend=False,
    )

    return fig


# ── Signal distribution pie ──────────────────────────────

def build_signal_breakdown(signals) -> go.Figure:
    """Donut chart showing signal type distribution."""
    from collections import Counter
    type_counts = Counter()
    for sig in signals:
        nice_name = sig.signal_type.value.replace("_", " ").title()
        type_counts[nice_name] += 1

    labels = list(type_counts.keys())
    values = list(type_counts.values())

    color_map = {
        "Break Of Structure":       "#3b82f6",
        "Change Of Character":      "#f59e0b",
        "Bullish Order Block":      "#22c55e",
        "Bearish Order Block":      "#ef4444",
        "Bullish Fair Value Gap":   "#6366f1",
        "Bearish Fair Value Gap":   "#f97316",
        "Liquidity Sweep High":     "#a855f7",
        "Liquidity Sweep Low":      "#a855f7",
        "Premium Zone":             "#ef4444",
        "Discount Zone":            "#22c55e",
        "Kill Zone":                "#64748b",
    }
    colors = [color_map.get(l, "#64748b") for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
        textinfo="label+value",
        textfont=dict(size=10, color="white"),
        hovertemplate="%{label}: %{value} signals<extra></extra>",
    )])

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        showlegend=False,
    )

    return fig
