"""
Feature Engineering Pipeline
=============================
Transforms raw trade records and market context into ML-ready features.

Feature categories:
  1. Trade-level features (per trade)
  2. Portfolio-level features (aggregated over time windows)
  3. Market regime features (from SPY, VIX, BTC, DXY)
  4. Temporal features (time-of-day, day-of-week, seasonality)
  5. Streak/momentum features (win/loss streaks, equity momentum)
  6. Risk features (drawdown depth, concentration, exposure)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  Trade-Level Features
# ══════════════════════════════════════════════════════════

def compute_trade_features(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich trade DataFrame with computed features.

    Input columns: timestamp, ticker, asset_type, action, pnl, pnl_pct,
                   r_multiple, hold_duration_hours, composite_score, volume
    """
    if trades_df.empty:
        return trades_df

    df = trades_df.copy()

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # ── Temporal features ──
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon, 6=Sun
    df["day_name"] = df["timestamp"].dt.day_name()
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["month"] = df["timestamp"].dt.month
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)

    # ── Outcome features ──
    df["is_winner"] = (df["pnl"] > 0).astype(int)
    df["is_loser"] = (df["pnl"] < 0).astype(int)
    df["abs_pnl"] = df["pnl"].abs()

    # ── Streak features ──
    df["win_streak"] = _compute_streaks(df["is_winner"].values)
    df["loss_streak"] = _compute_streaks(df["is_loser"].values)

    # ── Rolling performance (last N trades) ──
    for window in [5, 10, 20]:
        df[f"win_rate_{window}"] = df["is_winner"].rolling(window, min_periods=1).mean()
        df[f"avg_pnl_{window}"] = df["pnl"].rolling(window, min_periods=1).mean()
        df[f"avg_r_{window}"] = df["r_multiple"].rolling(window, min_periods=1).mean()
        df[f"pnl_std_{window}"] = df["pnl"].rolling(window, min_periods=1).std().fillna(0)

    # ── Cumulative equity features ──
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["equity_peak"] = df["cumulative_pnl"].cummax()
    df["drawdown"] = df["cumulative_pnl"] - df["equity_peak"]
    df["drawdown_pct"] = np.where(
        df["equity_peak"] > 0,
        df["drawdown"] / df["equity_peak"] * 100,
        0,
    )

    # ── Hold duration features ──
    df["hold_hours_log"] = np.log1p(df["hold_duration_hours"].clip(lower=0))
    df["is_quick_trade"] = (df["hold_duration_hours"] < 4).astype(int)
    df["is_long_hold"] = (df["hold_duration_hours"] > 48).astype(int)

    # ── Score features ──
    if "composite_score" in df.columns:
        df["high_conviction"] = (df["composite_score"].abs() >= 6).astype(int)
        df["low_conviction"] = (df["composite_score"].abs() <= 3).astype(int)

    return df


def _compute_streaks(binary_array: np.ndarray) -> np.ndarray:
    """Compute running streak length for a binary series."""
    streaks = np.zeros(len(binary_array), dtype=int)
    current_streak = 0
    for i, val in enumerate(binary_array):
        if val == 1:
            current_streak += 1
        else:
            current_streak = 0
        streaks[i] = current_streak
    return streaks


# ══════════════════════════════════════════════════════════
#  Portfolio-Level Features (Time Series)
# ══════════════════════════════════════════════════════════

def compute_portfolio_timeseries(
    trades_df: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Build a daily portfolio timeseries from trades and snapshots.

    Returns DataFrame indexed by date with columns:
      equity, daily_pnl, cumulative_pnl, drawdown, drawdown_pct,
      trade_count, win_count, loss_count, win_rate,
      sharpe_rolling, sortino_rolling, profit_factor_rolling
    """
    # Start with snapshots if available
    if not snapshots_df.empty:
        snapshots_df = snapshots_df.copy()
        snapshots_df["date"] = pd.to_datetime(snapshots_df["timestamp"]).dt.date
        daily = snapshots_df.groupby("date").agg(
            equity=("equity", "last"),
            daily_pnl=("daily_pnl", "sum"),
        ).sort_index()
    else:
        daily = pd.DataFrame()

    # Add trade-based metrics
    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["date"] = pd.to_datetime(trades_df["timestamp"]).dt.date
        trade_daily = trades_df.groupby("date").agg(
            trade_count=("pnl", "count"),
            win_count=("is_winner", "sum") if "is_winner" in trades_df.columns else ("pnl", lambda x: (x > 0).sum()),
            loss_count=("is_loser", "sum") if "is_loser" in trades_df.columns else ("pnl", lambda x: (x < 0).sum()),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            max_win=("pnl", "max"),
            max_loss=("pnl", "min"),
            avg_hold_hours=("hold_duration_hours", "mean"),
        ).sort_index()

        trade_daily["win_rate"] = np.where(
            trade_daily["trade_count"] > 0,
            trade_daily["win_count"] / trade_daily["trade_count"],
            0,
        )

        if daily.empty:
            daily = trade_daily
        else:
            daily = daily.join(trade_daily, how="outer")

    if daily.empty:
        return pd.DataFrame()

    daily = daily.fillna(0)

    # Cumulative metrics
    pnl_col = "daily_pnl" if "daily_pnl" in daily.columns else "total_pnl"
    daily["cumulative_pnl"] = daily[pnl_col].cumsum()
    daily["equity_peak"] = daily["cumulative_pnl"].cummax()
    daily["drawdown"] = daily["cumulative_pnl"] - daily["equity_peak"]

    # Rolling risk metrics (21-day ~ 1 month)
    returns = daily[pnl_col]
    daily["sharpe_21d"] = (
        returns.rolling(21, min_periods=5).mean()
        / returns.rolling(21, min_periods=5).std().replace(0, np.nan)
    ) * np.sqrt(252)

    # Sortino: only downside deviation
    downside = returns.clip(upper=0)
    daily["sortino_21d"] = (
        returns.rolling(21, min_periods=5).mean()
        / downside.rolling(21, min_periods=5).std().replace(0, np.nan)
    ) * np.sqrt(252)

    # Profit factor (rolling 21 days)
    gains = returns.clip(lower=0).rolling(21, min_periods=5).sum()
    losses = returns.clip(upper=0).abs().rolling(21, min_periods=5).sum()
    daily["profit_factor_21d"] = np.where(losses > 0, gains / losses, np.nan)

    # Calmar ratio
    if daily["equity_peak"].max() > 0:
        max_dd = daily["drawdown"].min()
        annual_return = daily[pnl_col].mean() * 252
        daily["calmar_ratio"] = annual_return / abs(max_dd) if max_dd != 0 else np.nan

    return daily


# ══════════════════════════════════════════════════════════
#  Market Regime Features
# ══════════════════════════════════════════════════════════

def compute_regime_features(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market regime indicators from benchmark data.

    Returns DataFrame with: regime (bull/bear/neutral), volatility_regime,
    trend_strength, correlation features.
    """
    if market_df.empty:
        return pd.DataFrame()

    df = market_df.copy()
    features = pd.DataFrame(index=df.index)

    # SPY regime
    if "SPY_close" in df.columns:
        spy = df["SPY_close"]
        sma20 = spy.rolling(20).mean()
        sma50 = spy.rolling(50).mean()

        features["spy_above_sma20"] = (spy > sma20).astype(int)
        features["spy_above_sma50"] = (spy > sma50).astype(int)
        features["spy_sma20_slope"] = sma20.pct_change(5)  # 5-day slope
        features["spy_regime"] = np.where(
            (spy > sma20) & (sma20 > sma50), "bull",
            np.where((spy < sma20) & (sma20 < sma50), "bear", "neutral"),
        )

    # VIX regime
    if "VIX_close" in df.columns:
        vix = df["VIX_close"]
        features["vix_level"] = vix
        features["vix_regime"] = np.where(
            vix < 15, "low_vol",
            np.where(vix < 25, "normal_vol",
            np.where(vix < 35, "high_vol", "crisis")),
        )
        features["vix_expanding"] = (vix > vix.rolling(10).mean()).astype(int)

    # BTC regime
    if "BTC_close" in df.columns:
        btc = df["BTC_close"]
        btc_sma20 = btc.rolling(20).mean()
        features["btc_above_sma20"] = (btc > btc_sma20).astype(int)
        features["btc_momentum_20d"] = btc.pct_change(20)

    # Dollar strength (affects forex)
    if "DXY_close" in df.columns:
        dxy = df["DXY_close"]
        features["dxy_trend"] = dxy.pct_change(20)
        features["dxy_above_sma20"] = (dxy > dxy.rolling(20).mean()).astype(int)

    # Cross-asset correlations (rolling 20-day)
    return_cols = [c for c in df.columns if c.endswith("_return")]
    if len(return_cols) >= 2:
        for i, col1 in enumerate(return_cols):
            for col2 in return_cols[i+1:]:
                name = f"corr_{col1.split('_')[0]}_{col2.split('_')[0]}"
                features[name] = df[col1].rolling(20).corr(df[col2])

    return features


# ══════════════════════════════════════════════════════════
#  Weakness Detection Features
# ══════════════════════════════════════════════════════════

def compute_weakness_features(trades_df: pd.DataFrame) -> dict:
    """
    Compute per-dimension breakdown for weakness detection.

    Returns dict of DataFrames keyed by dimension:
      by_asset_type, by_ticker, by_day_of_week, by_hour,
      by_session, by_hold_duration_bucket, by_score_bucket
    """
    if trades_df.empty:
        return {}

    df = trades_df.copy()
    result = {}

    # Helper to compute stats per group
    def _group_stats(group_col: str) -> pd.DataFrame:
        if group_col not in df.columns:
            return pd.DataFrame()
        stats = df.groupby(group_col).agg(
            trade_count=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            win_rate=("is_winner", "mean") if "is_winner" in df.columns else ("pnl", lambda x: (x > 0).mean()),
            avg_r=("r_multiple", "mean"),
            max_drawdown=("pnl", "min"),
            best_trade=("pnl", "max"),
            pnl_std=("pnl", "std"),
        ).fillna(0)

        # Sharpe per group (annualized approx)
        if "pnl_std" in stats.columns:
            stats["sharpe_approx"] = np.where(
                stats["pnl_std"] > 0,
                stats["avg_pnl"] / stats["pnl_std"] * np.sqrt(52),  # weekly-ish
                0,
            )

        return stats.sort_values("total_pnl")

    # By asset type
    result["by_asset_type"] = _group_stats("asset_type")

    # By ticker
    result["by_ticker"] = _group_stats("ticker")

    # By day of week
    if "day_name" in df.columns:
        result["by_day_of_week"] = _group_stats("day_name")

    # By hour
    if "hour" in df.columns:
        result["by_hour"] = _group_stats("hour")

    # By hold duration bucket
    if "hold_duration_hours" in df.columns:
        df["hold_bucket"] = pd.cut(
            df["hold_duration_hours"],
            bins=[0, 1, 4, 12, 24, 48, 72, float("inf")],
            labels=["<1h", "1-4h", "4-12h", "12-24h", "1-2d", "2-3d", "3d+"],
        )
        result["by_hold_duration"] = _group_stats("hold_bucket")

    # By composite score bucket
    if "composite_score" in df.columns:
        df["score_bucket"] = pd.cut(
            df["composite_score"].abs(),
            bins=[0, 3, 5, 7, 10],
            labels=["low(1-3)", "med(4-5)", "high(6-7)", "very_high(8+)"],
        )
        result["by_score"] = _group_stats("score_bucket")

    return result


# ══════════════════════════════════════════════════════════
#  Git Change Impact Features
# ══════════════════════════════════════════════════════════

def compute_change_impact(
    trades_df: pd.DataFrame,
    commits: list[dict],
    window_days: int = 5,
) -> list[dict]:
    """
    For each strategy-affecting commit, compute performance before vs after.

    Returns list of dicts with: commit info, before/after win_rate,
    avg_pnl, avg_r, and estimated impact.
    """
    if trades_df.empty or not commits:
        return []

    df = trades_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    impacts = []
    for commit in commits:
        # Only analyze commits that touch strategy/config/risk files
        if not any([
            commit.get("touches_strategy"),
            commit.get("touches_executor"),
            commit.get("touches_config"),
            commit.get("touches_risk"),
        ]):
            continue

        commit_time = pd.to_datetime(commit.get("timestamp", ""), utc=True)
        if pd.isna(commit_time):
            continue

        before = df[
            (df["timestamp"] >= commit_time - timedelta(days=window_days))
            & (df["timestamp"] < commit_time)
        ]
        after = df[
            (df["timestamp"] >= commit_time)
            & (df["timestamp"] < commit_time + timedelta(days=window_days))
        ]

        if len(before) < 3 or len(after) < 3:
            continue  # Not enough data

        impact = {
            "sha": commit["sha"][:8],
            "timestamp": commit["timestamp"],
            "message": commit["message"][:80],
            "files_changed": len(commit.get("files_changed", [])),
            "touches": ", ".join(
                [k.replace("touches_", "") for k in
                 ["touches_strategy", "touches_executor", "touches_config", "touches_risk"]
                 if commit.get(k)]
            ),
            "before_trades": len(before),
            "after_trades": len(after),
            "before_win_rate": float((before["pnl"] > 0).mean()),
            "after_win_rate": float((after["pnl"] > 0).mean()),
            "before_avg_pnl": float(before["pnl"].mean()),
            "after_avg_pnl": float(after["pnl"].mean()),
            "before_avg_r": float(before["r_multiple"].mean()) if "r_multiple" in before.columns else 0,
            "after_avg_r": float(after["r_multiple"].mean()) if "r_multiple" in after.columns else 0,
        }

        # Compute impact verdict
        pnl_change = impact["after_avg_pnl"] - impact["before_avg_pnl"]
        wr_change = impact["after_win_rate"] - impact["before_win_rate"]
        impact["pnl_impact"] = pnl_change
        impact["win_rate_impact"] = wr_change
        impact["verdict"] = (
            "POSITIVE" if pnl_change > 0 and wr_change >= -0.05
            else "NEGATIVE" if pnl_change < 0 and wr_change <= 0.05
            else "MIXED"
        )

        impacts.append(impact)

    return impacts
