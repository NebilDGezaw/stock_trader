"""
Portfolio Analyzer — Orchestrates the full ML analysis pipeline.
================================================================
Produces a comprehensive PortfolioReport for each system (Alpaca / HFM)
by coordinating data collection, feature engineering, model fitting,
forecasting, weakness detection, and suggestion generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from ml.data_collector import collect_all, _load_json
from ml.features import (
    compute_trade_features,
    compute_portfolio_timeseries,
    compute_regime_features,
    compute_weakness_features,
    compute_change_impact,
)
from ml.models import (
    EquityForecaster,
    MonteCarloSimulator,
    TradeClassifier,
    PortfolioAnomalyDetector,
    BayesianEstimator,
    compute_health_score,
    detect_changepoints,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  Report Data Structure
# ══════════════════════════════════════════════════════════

@dataclass
class PortfolioReport:
    """Complete analysis output for one trading system."""
    system: str                           # "alpaca" or "hfm"
    generated_at: str = ""
    total_trades: int = 0
    data_days: int = 0

    # ── Performance Summary ──
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    week_pnl: float = 0.0
    month_pnl: float = 0.0
    win_rate: float = 0.0
    avg_r_multiple: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    current_equity: float = 0.0
    avg_hold_hours: float = 0.0

    # ── Health Score ──
    health_score: float = 0.0
    health_grade: str = ""
    health_components: dict = field(default_factory=dict)

    # ── Forecasts ──
    forecast_1w: dict = field(default_factory=dict)   # next week
    forecast_1m: dict = field(default_factory=dict)   # next month
    forecast_3m: dict = field(default_factory=dict)   # next quarter
    forecast_eoy: dict = field(default_factory=dict)  # end of year
    monte_carlo_1w: dict = field(default_factory=dict)
    monte_carlo_1m: dict = field(default_factory=dict)
    monte_carlo_eoy: dict = field(default_factory=dict)

    # ── Weaknesses ──
    weaknesses: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)

    # ── Weakness Breakdowns ──
    by_asset_type: dict = field(default_factory=dict)
    by_ticker: dict = field(default_factory=dict)
    by_day: dict = field(default_factory=dict)
    by_hour: dict = field(default_factory=dict)

    # ── Git Impact ──
    code_changes: list = field(default_factory=list)

    # ── Model Info ──
    classifier_accuracy: float = 0.0
    top_features: list = field(default_factory=list)
    anomalous_periods: int = 0
    changepoints: list = field(default_factory=list)
    bayesian_summary: dict = field(default_factory=dict)

    # ── Market Context ──
    market_regime: str = ""
    vix_level: float = 0.0


# ══════════════════════════════════════════════════════════
#  Main Analyzer
# ══════════════════════════════════════════════════════════

class PortfolioAnalyzer:
    """
    Runs the complete ML analysis pipeline for a trading system.
    """

    def __init__(self, system: str, lookback_days: int = 90):
        assert system in ("alpaca", "hfm"), f"Unknown system: {system}"
        self.system = system
        self.lookback_days = lookback_days

    def analyze(self, skip_collection: bool = False) -> PortfolioReport:
        """
        Run the full analysis pipeline.

        Args:
            skip_collection: If True, only use persisted data (no API calls).

        Returns:
            PortfolioReport with all analysis results.
        """
        report = PortfolioReport(
            system=self.system,
            generated_at=datetime.utcnow().isoformat(),
        )

        # ── Step 1: Collect data ──
        if not skip_collection:
            try:
                collect_all(system=self.system, days=self.lookback_days)
            except Exception as e:
                logger.warning(f"Data collection failed: {e} — using persisted data")

        trades_raw = _load_json(f"trades_{self.system}.json")
        snapshots_raw = _load_json(f"snapshots_{self.system}.json")
        git_commits = _load_json("git_commits.json")

        if not trades_raw and not snapshots_raw:
            logger.warning(f"No data available for {self.system}")
            report.health_grade = "No Data"
            return report

        trades_df = pd.DataFrame(trades_raw)
        snapshots_df = pd.DataFrame(snapshots_raw)

        report.total_trades = len(trades_df)
        logger.info(
            f"Analyzing {self.system}: {len(trades_df)} trades, "
            f"{len(snapshots_df)} snapshots, {len(git_commits)} commits"
        )

        # ── Step 2: Feature engineering ──
        if not trades_df.empty:
            trades_df = compute_trade_features(trades_df)
            report.data_days = (
                trades_df["timestamp"].max() - trades_df["timestamp"].min()
            ).days if "timestamp" in trades_df.columns else 0

        portfolio_ts = compute_portfolio_timeseries(trades_df, snapshots_df)

        # ── Step 3: Basic metrics ──
        self._compute_basic_metrics(report, trades_df, snapshots_df, portfolio_ts)

        # ── Step 4: Health score ──
        health = compute_health_score({
            "win_rate": report.win_rate,
            "profit_factor": report.profit_factor,
            "sharpe_ratio": report.sharpe_ratio,
            "max_drawdown_pct": report.max_drawdown_pct,
            "avg_r_multiple": report.avg_r_multiple,
            "avg_risk_reward": max(report.avg_r_multiple * 1.5, 1.0),
            "daily_pnl_std": float(portfolio_ts["daily_pnl"].std()) if "daily_pnl" in portfolio_ts.columns and len(portfolio_ts) > 0 else 0,
            "avg_daily_pnl": float(portfolio_ts["daily_pnl"].mean()) if "daily_pnl" in portfolio_ts.columns and len(portfolio_ts) > 0 else 0,
        })
        report.health_score = health["overall_score"]
        report.health_grade = health["grade"]
        report.health_components = health["component_scores"]

        # ── Step 5: Forecasting ──
        self._run_forecasting(report, portfolio_ts, snapshots_df)

        # ── Step 6: Weakness detection ──
        self._detect_weaknesses(report, trades_df)

        # ── Step 7: Trade classifier ──
        self._run_classifier(report, trades_df)

        # ── Step 8: Anomaly detection ──
        self._run_anomaly_detection(report, portfolio_ts)

        # ── Step 9: Changepoint detection ──
        self._run_changepoint_detection(report, portfolio_ts)

        # ── Step 10: Bayesian estimation ──
        self._run_bayesian(report, trades_df)

        # ── Step 11: Git change impact ──
        self._analyze_git_impact(report, trades_df, git_commits)

        # ── Step 12: Market context ──
        self._add_market_context(report)

        # ── Step 13: Generate suggestions ──
        self._generate_suggestions(report)

        return report

    # ──────────────────────────────────────────────────────
    #  Step 3: Basic Metrics
    # ──────────────────────────────────────────────────────

    def _compute_basic_metrics(self, report, trades_df, snapshots_df, portfolio_ts):
        if trades_df.empty:
            return

        report.total_pnl = float(trades_df["pnl"].sum())
        report.win_rate = float((trades_df["pnl"] > 0).mean())

        if "r_multiple" in trades_df.columns:
            report.avg_r_multiple = float(trades_df["r_multiple"].mean())

        if "hold_duration_hours" in trades_df.columns:
            report.avg_hold_hours = float(trades_df["hold_duration_hours"].mean())

        # Profit factor
        gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        report.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Recent P&L
        if "timestamp" in trades_df.columns:
            now = datetime.utcnow()
            week_trades = trades_df[trades_df["timestamp"] >= now - timedelta(days=7)]
            month_trades = trades_df[trades_df["timestamp"] >= now - timedelta(days=30)]
            report.week_pnl = float(week_trades["pnl"].sum())
            report.month_pnl = float(month_trades["pnl"].sum())

        # From portfolio timeseries
        if not portfolio_ts.empty:
            if "sharpe_21d" in portfolio_ts.columns:
                report.sharpe_ratio = float(portfolio_ts["sharpe_21d"].iloc[-1])
                if np.isnan(report.sharpe_ratio):
                    report.sharpe_ratio = 0
            if "sortino_21d" in portfolio_ts.columns:
                report.sortino_ratio = float(portfolio_ts["sortino_21d"].iloc[-1])
                if np.isnan(report.sortino_ratio):
                    report.sortino_ratio = 0
            if "drawdown" in portfolio_ts.columns:
                report.max_drawdown = float(portfolio_ts["drawdown"].min())
            if "cumulative_pnl" in portfolio_ts.columns and portfolio_ts["cumulative_pnl"].max() > 0:
                report.max_drawdown_pct = float(
                    portfolio_ts["drawdown"].min() / portfolio_ts["cumulative_pnl"].max() * 100
                )

        # Current equity
        if not snapshots_df.empty and "equity" in snapshots_df.columns:
            report.current_equity = float(snapshots_df["equity"].iloc[-1])
            if report.current_equity > 0:
                report.total_pnl_pct = report.total_pnl / report.current_equity * 100

    # ──────────────────────────────────────────────────────
    #  Step 5: Forecasting
    # ──────────────────────────────────────────────────────

    def _run_forecasting(self, report, portfolio_ts, snapshots_df):
        # Equity curve forecasting (ARIMA/ExpSmooth)
        equity_col = None
        if not portfolio_ts.empty:
            for col in ["cumulative_pnl", "equity"]:
                if col in portfolio_ts.columns:
                    equity_col = portfolio_ts[col].dropna()
                    break

        if equity_col is not None and len(equity_col) >= 10:
            forecaster = EquityForecaster(equity_col)
            forecaster.fit()

            # 1 week (5 trading days)
            f1w = forecaster.forecast(5)
            report.forecast_1w = self._format_forecast(f1w, equity_col.iloc[-1], "1 week")

            # 1 month (21 trading days)
            f1m = forecaster.forecast(21)
            report.forecast_1m = self._format_forecast(f1m, equity_col.iloc[-1], "1 month")

            # 3 months (63 trading days)
            f3m = forecaster.forecast(63)
            report.forecast_3m = self._format_forecast(f3m, equity_col.iloc[-1], "3 months")

            # End of year
            today = datetime.utcnow()
            eoy = datetime(today.year, 12, 31)
            days_to_eoy = max(5, (eoy - today).days * 5 // 7)  # approximate trading days
            feoy = forecaster.forecast(days_to_eoy)
            report.forecast_eoy = self._format_forecast(feoy, equity_col.iloc[-1], "end of year")

        # Monte Carlo simulation
        daily_returns = None
        if not portfolio_ts.empty:
            for col in ["daily_pnl", "total_pnl"]:
                if col in portfolio_ts.columns:
                    # Normalize to percentage returns
                    equity_base = report.current_equity if report.current_equity > 0 else 100000
                    daily_returns = portfolio_ts[col] / equity_base
                    break

        if daily_returns is not None and len(daily_returns) >= 10:
            mc = MonteCarloSimulator(daily_returns, n_simulations=10000)

            report.monte_carlo_1w = mc.simulate(5)
            report.monte_carlo_1m = mc.simulate(21)

            today = datetime.utcnow()
            eoy = datetime(today.year, 12, 31)
            days_to_eoy = max(5, (eoy - today).days * 5 // 7)
            report.monte_carlo_eoy = mc.simulate(days_to_eoy)

    def _format_forecast(self, forecast: dict, current_value: float, label: str) -> dict:
        if not forecast.get("forecast", np.array([])).size:
            return {"label": label, "available": False}

        fcast = forecast["forecast"]
        lower = forecast["lower_ci"]
        upper = forecast["upper_ci"]

        return {
            "label": label,
            "available": True,
            "method": forecast["method"],
            "predicted_value": float(fcast[-1]),
            "predicted_change": float(fcast[-1] - current_value),
            "predicted_change_pct": float((fcast[-1] - current_value) / abs(current_value) * 100) if current_value != 0 else 0,
            "lower_80": float(lower[-1]),
            "upper_80": float(upper[-1]),
            "lower_change_pct": float((lower[-1] - current_value) / abs(current_value) * 100) if current_value != 0 else 0,
            "upper_change_pct": float((upper[-1] - current_value) / abs(current_value) * 100) if current_value != 0 else 0,
        }

    # ──────────────────────────────────────────────────────
    #  Step 6: Weakness Detection
    # ──────────────────────────────────────────────────────

    def _detect_weaknesses(self, report, trades_df):
        if trades_df.empty:
            return

        breakdowns = compute_weakness_features(trades_df)
        weaknesses = []

        # Check each dimension for underperformers
        overall_avg_pnl = trades_df["pnl"].mean()
        overall_wr = (trades_df["pnl"] > 0).mean()

        for dim_name, dim_df in breakdowns.items():
            if dim_df.empty:
                continue

            # Store breakdown in report
            if dim_name == "by_asset_type":
                report.by_asset_type = dim_df.to_dict("index")
            elif dim_name == "by_ticker":
                report.by_ticker = dim_df.to_dict("index")
            elif dim_name == "by_day_of_week":
                report.by_day = dim_df.to_dict("index")
            elif dim_name == "by_hour":
                report.by_hour = dim_df.to_dict("index")

            # Detect weaknesses
            for idx, row in dim_df.iterrows():
                if row.get("trade_count", 0) < 5:
                    continue  # Not enough data

                avg_pnl = row.get("avg_pnl", 0)
                wr = row.get("win_rate", 0.5)

                # Significantly worse than portfolio average
                if avg_pnl < overall_avg_pnl * 0.5 and avg_pnl < 0:
                    weaknesses.append({
                        "dimension": dim_name.replace("by_", ""),
                        "value": str(idx),
                        "metric": "avg_pnl",
                        "portfolio_avg": round(overall_avg_pnl, 2),
                        "this_value": round(avg_pnl, 2),
                        "trade_count": int(row["trade_count"]),
                        "total_pnl": round(row.get("total_pnl", 0), 2),
                        "severity": "high" if avg_pnl < -abs(overall_avg_pnl) else "medium",
                    })

                if wr < overall_wr - 0.15 and wr < 0.40:
                    weaknesses.append({
                        "dimension": dim_name.replace("by_", ""),
                        "value": str(idx),
                        "metric": "win_rate",
                        "portfolio_avg": round(overall_wr * 100, 1),
                        "this_value": round(wr * 100, 1),
                        "trade_count": int(row["trade_count"]),
                        "severity": "high" if wr < 0.30 else "medium",
                    })

        # Sort by severity
        weaknesses.sort(key=lambda w: (0 if w["severity"] == "high" else 1))
        report.weaknesses = weaknesses[:15]  # Top 15

    # ──────────────────────────────────────────────────────
    #  Step 7: Classifier
    # ──────────────────────────────────────────────────────

    def _run_classifier(self, report, trades_df):
        if trades_df.empty:
            return

        classifier = TradeClassifier()
        classifier.fit(trades_df)
        report.classifier_accuracy = classifier.accuracy
        report.top_features = classifier.get_top_features(5)

    # ──────────────────────────────────────────────────────
    #  Step 8: Anomaly Detection
    # ──────────────────────────────────────────────────────

    def _run_anomaly_detection(self, report, portfolio_ts):
        if portfolio_ts.empty:
            return

        detector = PortfolioAnomalyDetector()
        anomaly_scores = detector.fit_predict(portfolio_ts)
        if len(anomaly_scores) > 0:
            report.anomalous_periods = int((anomaly_scores == -1).sum())

    # ──────────────────────────────────────────────────────
    #  Step 9: Changepoints
    # ──────────────────────────────────────────────────────

    def _run_changepoint_detection(self, report, portfolio_ts):
        if portfolio_ts.empty:
            return

        for col in ["cumulative_pnl", "daily_pnl"]:
            if col in portfolio_ts.columns:
                cps = detect_changepoints(portfolio_ts[col])
                report.changepoints = cps[:5]  # Top 5
                break

    # ──────────────────────────────────────────────────────
    #  Step 10: Bayesian Estimation
    # ──────────────────────────────────────────────────────

    def _run_bayesian(self, report, trades_df):
        if trades_df.empty:
            return

        estimator = BayesianEstimator()
        for _, trade in trades_df.iterrows():
            won = trade.get("pnl", 0) > 0
            r = trade.get("r_multiple", 0)
            estimator.update(won, r)

        report.bayesian_summary = estimator.summary()

    # ──────────────────────────────────────────────────────
    #  Step 11: Git Change Impact
    # ──────────────────────────────────────────────────────

    def _analyze_git_impact(self, report, trades_df, git_commits):
        if trades_df.empty or not git_commits:
            return
        report.code_changes = compute_change_impact(trades_df, git_commits)

    # ──────────────────────────────────────────────────────
    #  Step 12: Market Context
    # ──────────────────────────────────────────────────────

    def _add_market_context(self, report):
        try:
            from ml.features import compute_regime_features
            from ml.data_collector import collect_market_context

            market = collect_market_context(30)
            if not market.empty:
                regime = compute_regime_features(market)
                if not regime.empty and "spy_regime" in regime.columns:
                    report.market_regime = str(regime["spy_regime"].iloc[-1])
                if not regime.empty and "vix_level" in regime.columns:
                    report.vix_level = float(regime["vix_level"].iloc[-1])
        except Exception as e:
            logger.debug(f"Market context failed: {e}")

    # ──────────────────────────────────────────────────────
    #  Step 13: Suggestions
    # ──────────────────────────────────────────────────────

    def _generate_suggestions(self, report):
        """Generate actionable suggestions based on all analysis."""
        suggestions = []

        # From health score components
        if report.health_components:
            for component, score in report.health_components.items():
                if score < 40:
                    suggestions.append(self._suggestion_for_component(component, score))

        # From weaknesses
        for w in report.weaknesses[:5]:
            dim = w["dimension"]
            val = w["value"]
            metric = w["metric"]
            severity = w["severity"]

            if metric == "avg_pnl" and severity == "high":
                suggestions.append(
                    f"REDUCE exposure to {dim}={val}: losing avg ${abs(w['this_value']):.0f}/trade "
                    f"({w['trade_count']} trades). Consider removing from watchlist or "
                    f"tightening stop loss."
                )
            elif metric == "win_rate" and severity == "high":
                suggestions.append(
                    f"REVIEW {dim}={val}: win rate only {w['this_value']}% "
                    f"(portfolio avg {w['portfolio_avg']}%). Raise the minimum "
                    f"composite score threshold for this category."
                )

        # From classifier
        if report.top_features:
            top_feat = report.top_features[0]
            suggestions.append(
                f"ML MODEL: The strongest predictor of trade success is "
                f"'{top_feat[0]}' (importance={top_feat[1]:.1%}). "
                f"Pay attention to this metric when deciding trades."
            )

        # From Bayesian
        if report.bayesian_summary:
            bs = report.bayesian_summary
            if bs.get("expectancy", 0) < 0:
                suggestions.append(
                    f"EXPECTANCY WARNING: Current Bayesian expectancy is "
                    f"${bs['expectancy']:.2f} per trade (negative). "
                    f"The system is expected to LOSE money at current performance. "
                    f"Review strategy parameters."
                )

        # Monte Carlo warnings
        if report.monte_carlo_1m:
            mc = report.monte_carlo_1m
            if mc.get("prob_below_neg5pct", 0) > 0.20:
                suggestions.append(
                    f"RISK ALERT: Monte Carlo shows {mc['prob_below_neg5pct']:.0%} chance "
                    f"of >5% loss this month. Consider reducing position sizes or "
                    f"tightening the daily loss limit."
                )

        # From anomalies
        if report.anomalous_periods > 3:
            suggestions.append(
                f"INSTABILITY: {report.anomalous_periods} anomalous periods detected "
                f"in the last {report.data_days} days. The portfolio is behaving "
                f"inconsistently — consider pausing to investigate."
            )

        # Market regime
        if report.market_regime == "bear" and report.system == "alpaca":
            suggestions.append(
                "MARKET REGIME: SPY is in a bearish regime. Long-only stock "
                "trading may underperform. Consider reducing position sizes "
                "and raising the minimum score threshold."
            )

        if report.vix_level > 25:
            suggestions.append(
                f"HIGH VOLATILITY: VIX is at {report.vix_level:.1f}. "
                f"Widen stop losses by 20-30% and reduce position sizes "
                f"to account for increased whipsaw risk."
            )

        report.suggestions = suggestions

    def _suggestion_for_component(self, component: str, score: float) -> str:
        msg = {
            "win_rate": (
                f"WIN RATE is low (score={score:.0f}/100). Focus on higher-conviction "
                f"trades by raising minimum composite score from 4 to 5."
            ),
            "profit_factor": (
                f"PROFIT FACTOR is weak (score={score:.0f}/100). Winners aren't "
                f"covering losers. Widen take-profit targets or tighten stop losses."
            ),
            "sharpe_ratio": (
                f"RISK-ADJUSTED RETURNS poor (Sharpe score={score:.0f}/100). "
                f"Returns are volatile relative to their magnitude. "
                f"Reduce position sizing for more consistency."
            ),
            "max_drawdown": (
                f"DRAWDOWN too deep (score={score:.0f}/100). Tighten the daily "
                f"loss limit and consider reducing max concurrent positions."
            ),
            "avg_r_multiple": (
                f"AVERAGE R-MULTIPLE is low (score={score:.0f}/100). Trades aren't "
                f"reaching targets. Review take-profit distances and hold durations."
            ),
            "risk_reward": (
                f"RISK/REWARD ratio poor (score={score:.0f}/100). "
                f"Ensure all trades have minimum 1.5:1 R:R before entry."
            ),
            "consistency": (
                f"INCONSISTENT returns (score={score:.0f}/100). Large swings "
                f"between winning and losing days. Reduce position size variance."
            ),
        }
        return msg.get(component, f"{component} needs improvement (score={score:.0f}/100)")
