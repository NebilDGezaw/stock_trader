"""
ML Models — Ensemble of forecasting and classification models.
===============================================================
Models:
  1. ARIMA/Exponential Smoothing  — equity curve forecasting
  2. Monte Carlo Simulation       — probabilistic return projections
  3. Random Forest / GBT          — trade outcome classification
  4. Isolation Forest              — anomaly detection (unhealthy periods)
  5. Changepoint Detection         — regime shift identification
  6. Bayesian Rolling Estimator    — adaptive win rate & expectancy

All models degrade gracefully with limited data — they switch to
statistical fallbacks when fewer than ~30 data points are available.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_SAMPLES_ML = 30      # Minimum trades for ML models
MIN_SAMPLES_STATS = 10   # Minimum for statistical methods


# ══════════════════════════════════════════════════════════
#  1. Equity Curve Forecasting (ARIMA + Exponential Smoothing)
# ══════════════════════════════════════════════════════════

class EquityForecaster:
    """
    Forecasts equity curve using statsmodels ARIMA or
    Exponential Smoothing with automatic fallback to
    simple trend extrapolation.
    """

    def __init__(self, equity_series: pd.Series):
        """equity_series: daily cumulative P&L or equity values."""
        self.series = equity_series.dropna()
        self.model = None
        self.method = "none"

    def fit(self):
        """Fit the best available model."""
        if len(self.series) < MIN_SAMPLES_STATS:
            self.method = "insufficient_data"
            return self

        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(
                self.series.values,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            self.model = model
            self.method = "exponential_smoothing"
            logger.info(f"Fitted ExponentialSmoothing (AIC={model.aic:.1f})")
            return self
        except Exception as e:
            logger.debug(f"ExponentialSmoothing failed: {e}")

        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(self.series.values, order=(1, 1, 1)).fit()
            self.model = model
            self.method = "arima"
            logger.info(f"Fitted ARIMA(1,1,1) (AIC={model.aic:.1f})")
            return self
        except Exception as e:
            logger.debug(f"ARIMA failed: {e}")

        # Fallback: linear trend
        self.method = "linear_trend"
        return self

    def forecast(self, periods: int = 30) -> dict:
        """
        Forecast future equity values.

        Returns dict with:
          forecast: array of predicted values
          lower_ci: lower 80% confidence interval
          upper_ci: upper 80% confidence interval
          method: which model was used
        """
        if self.method == "insufficient_data":
            return {
                "forecast": np.array([]),
                "lower_ci": np.array([]),
                "upper_ci": np.array([]),
                "method": "insufficient_data",
            }

        if self.method in ("exponential_smoothing",):
            fcast = self.model.forecast(periods)
            # Estimate CI from residuals
            residuals = self.model.resid
            std = np.std(residuals) if len(residuals) > 0 else 0
            steps = np.arange(1, periods + 1)
            ci_width = 1.28 * std * np.sqrt(steps)  # 80% CI
            return {
                "forecast": fcast,
                "lower_ci": fcast - ci_width,
                "upper_ci": fcast + ci_width,
                "method": self.method,
            }

        if self.method == "arima":
            fcast_result = self.model.get_forecast(periods)
            fcast = fcast_result.predicted_mean
            ci = fcast_result.conf_int(alpha=0.20)  # 80% CI
            return {
                "forecast": fcast,
                "lower_ci": ci.iloc[:, 0].values,
                "upper_ci": ci.iloc[:, 1].values,
                "method": self.method,
            }

        # Linear trend fallback
        y = self.series.values
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(len(y), len(y) + periods)
        fcast = slope * future_x + intercept

        # CI from residuals of trend
        residuals = y - (slope * x + intercept)
        std = np.std(residuals)
        steps = np.arange(1, periods + 1)
        ci_width = 1.28 * std * np.sqrt(steps)

        return {
            "forecast": fcast,
            "lower_ci": fcast - ci_width,
            "upper_ci": fcast + ci_width,
            "method": "linear_trend",
        }


# ══════════════════════════════════════════════════════════
#  2. Monte Carlo Return Simulation
# ══════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """
    Simulates future portfolio returns using bootstrapped
    historical daily returns with regime-adjusted parameters.
    """

    def __init__(self, daily_returns: pd.Series, n_simulations: int = 10000):
        self.returns = daily_returns.dropna().values
        self.n_sims = n_simulations

    def simulate(self, horizon_days: int = 30) -> dict:
        """
        Run Monte Carlo simulation.

        Returns dict with:
          median_return: median cumulative return
          percentile_10: 10th percentile (pessimistic)
          percentile_25: 25th percentile
          percentile_75: 75th percentile
          percentile_90: 90th percentile (optimistic)
          prob_positive: probability of positive return
          prob_above_1pct: probability of >1% return
          all_paths: array of (n_sims, horizon_days) — optional
        """
        if len(self.returns) < MIN_SAMPLES_STATS:
            return self._empty_result()

        # Bootstrap: sample with replacement from historical returns
        # Add slight regime adjustment — recent returns weighted more
        n = len(self.returns)
        weights = np.linspace(0.5, 1.5, n)  # Recent data weighted 3x first
        weights /= weights.sum()

        paths = np.zeros((self.n_sims, horizon_days))
        for sim in range(self.n_sims):
            sampled = np.random.choice(self.returns, size=horizon_days, p=weights)
            paths[sim] = np.cumsum(sampled)

        final_returns = paths[:, -1]

        return {
            "horizon_days": horizon_days,
            "n_simulations": self.n_sims,
            "median_return": float(np.median(final_returns)),
            "mean_return": float(np.mean(final_returns)),
            "percentile_5": float(np.percentile(final_returns, 5)),
            "percentile_10": float(np.percentile(final_returns, 10)),
            "percentile_25": float(np.percentile(final_returns, 25)),
            "percentile_75": float(np.percentile(final_returns, 75)),
            "percentile_90": float(np.percentile(final_returns, 90)),
            "percentile_95": float(np.percentile(final_returns, 95)),
            "prob_positive": float((final_returns > 0).mean()),
            "prob_above_1pct": float((final_returns > 0.01).mean()),
            "prob_below_neg5pct": float((final_returns < -0.05).mean()),
            "max_drawdown_median": float(np.median(np.min(paths, axis=1))),
        }

    def _empty_result(self):
        return {
            "horizon_days": 0, "n_simulations": 0,
            "median_return": 0, "mean_return": 0,
            "percentile_5": 0, "percentile_10": 0,
            "percentile_25": 0, "percentile_75": 0,
            "percentile_90": 0, "percentile_95": 0,
            "prob_positive": 0.5, "prob_above_1pct": 0,
            "prob_below_neg5pct": 0, "max_drawdown_median": 0,
        }


# ══════════════════════════════════════════════════════════
#  3. Trade Outcome Classifier (Random Forest / GBT)
# ══════════════════════════════════════════════════════════

class TradeClassifier:
    """
    Predicts trade outcome (win/loss) from trade features
    using Random Forest with fallback to logistic regression.
    Also provides feature importance for weakness analysis.
    """

    FEATURE_COLS = [
        "hour", "day_of_week", "hold_hours_log",
        "win_rate_5", "win_rate_10", "avg_r_5",
        "loss_streak", "win_streak",
        "drawdown_pct", "pnl_std_10",
    ]

    def __init__(self):
        self.model = None
        self.feature_importances = {}
        self.accuracy = 0.0
        self.method = "none"

    def fit(self, trades_df: pd.DataFrame) -> "TradeClassifier":
        """Train on historical trades."""
        if len(trades_df) < MIN_SAMPLES_ML:
            self.method = "insufficient_data"
            return self

        df = trades_df.dropna(subset=["is_winner"])
        available_cols = [c for c in self.FEATURE_COLS if c in df.columns]

        if len(available_cols) < 3:
            self.method = "insufficient_features"
            return self

        X = df[available_cols].fillna(0).values
        y = df["is_winner"].values

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score

            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_samples_leaf=5,
                random_state=42,
            )

            # Cross-validated accuracy
            scores = cross_val_score(model, X, y, cv=min(5, len(df) // 5), scoring="accuracy")
            self.accuracy = float(scores.mean())

            # Fit on all data
            model.fit(X, y)
            self.model = model
            self.method = "gradient_boosting"

            # Feature importances
            for col, imp in zip(available_cols, model.feature_importances_):
                self.feature_importances[col] = float(imp)

            logger.info(
                f"TradeClassifier: {self.method}, accuracy={self.accuracy:.1%}, "
                f"features={len(available_cols)}"
            )

        except ImportError:
            logger.info("scikit-learn not available — using statistical classifier")
            self.method = "statistical"
            # Simple: use win_rate_10 as predictor
            if "win_rate_10" in df.columns:
                threshold = df["win_rate_10"].median()
                predictions = (df["win_rate_10"] > threshold).astype(int)
                self.accuracy = float((predictions == y).mean())

        return self

    def predict_proba(self, features: dict) -> float:
        """Predict win probability for a trade."""
        if self.model is None:
            return 0.5

        available = [c for c in self.FEATURE_COLS if c in features]
        X = np.array([[features.get(c, 0) for c in self.FEATURE_COLS]])

        try:
            return float(self.model.predict_proba(X)[0, 1])
        except Exception:
            return 0.5

    def get_top_features(self, n: int = 5) -> list[tuple[str, float]]:
        """Return top N most important features."""
        sorted_feats = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1], reverse=True,
        )
        return sorted_feats[:n]


# ══════════════════════════════════════════════════════════
#  4. Anomaly Detector (Isolation Forest)
# ══════════════════════════════════════════════════════════

class PortfolioAnomalyDetector:
    """
    Detects anomalous portfolio behavior using Isolation Forest.
    Flags days/periods where behavior deviates significantly from normal.
    """

    FEATURES = [
        "daily_pnl", "trade_count", "win_rate",
        "sharpe_21d", "drawdown",
    ]

    def __init__(self):
        self.model = None
        self.threshold = -0.5

    def fit_predict(self, portfolio_df: pd.DataFrame) -> pd.Series:
        """
        Fit and return anomaly scores.
        -1 = anomaly, 1 = normal. Continuous score also available.
        """
        available = [c for c in self.FEATURES if c in portfolio_df.columns]
        if len(available) < 2 or len(portfolio_df) < MIN_SAMPLES_ML:
            return pd.Series(dtype=float)

        X = portfolio_df[available].fillna(0).values

        try:
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(
                contamination=0.1,  # expect 10% anomalous days
                random_state=42,
                n_estimators=100,
            )
            scores = model.fit_predict(X)
            self.model = model
            return pd.Series(scores, index=portfolio_df.index, name="anomaly_score")
        except ImportError:
            # Fallback: z-score based
            z_scores = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
            max_z = z_scores.max(axis=1)
            anomalies = np.where(max_z > 2.5, -1, 1)
            return pd.Series(anomalies, index=portfolio_df.index, name="anomaly_score")


# ══════════════════════════════════════════════════════════
#  5. Changepoint Detection
# ══════════════════════════════════════════════════════════

def detect_changepoints(series: pd.Series, min_segment: int = 7) -> list[dict]:
    """
    Detect structural breaks in a time series using CUSUM.

    Returns list of dicts with: index, timestamp, direction (improvement/degradation),
    magnitude, before_mean, after_mean.
    """
    if len(series) < min_segment * 3:
        return []

    values = series.dropna().values
    n = len(values)
    mean = values.mean()
    std = values.std()

    if std == 0:
        return []

    # CUSUM
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    threshold = 4 * std  # Sensitivity threshold

    changepoints = []
    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i-1] + (values[i] - mean) - 0.5 * std)
        cusum_neg[i] = min(0, cusum_neg[i-1] + (values[i] - mean) + 0.5 * std)

        if cusum_pos[i] > threshold:
            # Positive shift detected
            before_mean = values[max(0, i-min_segment):i].mean()
            after_mean = values[i:min(n, i+min_segment)].mean()
            changepoints.append({
                "index": i,
                "timestamp": str(series.index[i]) if hasattr(series.index, '__getitem__') else "",
                "direction": "improvement",
                "magnitude": float(after_mean - before_mean),
                "before_mean": float(before_mean),
                "after_mean": float(after_mean),
            })
            cusum_pos[i] = 0

        if cusum_neg[i] < -threshold:
            before_mean = values[max(0, i-min_segment):i].mean()
            after_mean = values[i:min(n, i+min_segment)].mean()
            changepoints.append({
                "index": i,
                "timestamp": str(series.index[i]) if hasattr(series.index, '__getitem__') else "",
                "direction": "degradation",
                "magnitude": float(after_mean - before_mean),
                "before_mean": float(before_mean),
                "after_mean": float(after_mean),
            })
            cusum_neg[i] = 0

    return changepoints


# ══════════════════════════════════════════════════════════
#  6. Bayesian Rolling Estimator
# ══════════════════════════════════════════════════════════

class BayesianEstimator:
    """
    Bayesian updating of win rate and expected R-multiple.
    Uses conjugate priors for online learning.
    """

    def __init__(self, prior_wins: int = 5, prior_losses: int = 5,
                 prior_r_mean: float = 0.5, prior_r_var: float = 1.0):
        # Beta prior for win rate
        self.alpha = prior_wins
        self.beta = prior_losses
        # Normal prior for R-multiple
        self.r_mean = prior_r_mean
        self.r_var = prior_r_var
        self.r_n = 2  # pseudocount

    def update(self, won: bool, r_multiple: float = 0.0):
        """Update beliefs with a new trade outcome."""
        if won:
            self.alpha += 1
        else:
            self.beta += 1

        # Online mean/var update for R
        self.r_n += 1
        delta = r_multiple - self.r_mean
        self.r_mean += delta / self.r_n
        self.r_var += delta * (r_multiple - self.r_mean)

    def win_rate(self) -> float:
        """Posterior mean win rate."""
        return self.alpha / (self.alpha + self.beta)

    def win_rate_ci(self, confidence: float = 0.80) -> tuple[float, float]:
        """Credible interval for win rate."""
        from scipy import stats as scipy_stats
        lower = scipy_stats.beta.ppf((1 - confidence) / 2, self.alpha, self.beta)
        upper = scipy_stats.beta.ppf(1 - (1 - confidence) / 2, self.alpha, self.beta)
        return (float(lower), float(upper))

    def expected_r(self) -> float:
        """Posterior mean R-multiple."""
        return self.r_mean

    def expectancy(self) -> float:
        """Expected value per trade: WR × avg_win_R - (1-WR) × avg_loss_R."""
        wr = self.win_rate()
        # Simplified: assume avg win = +R_mean, avg loss = -1R
        return wr * max(self.r_mean, 0) - (1 - wr) * 1.0

    def summary(self) -> dict:
        return {
            "win_rate": self.win_rate(),
            "expected_r": self.expected_r(),
            "expectancy": self.expectancy(),
            "total_trades": self.alpha + self.beta - 10,  # subtract priors
            "confidence": "high" if (self.alpha + self.beta) > 50 else
                         "medium" if (self.alpha + self.beta) > 20 else "low",
        }


# ══════════════════════════════════════════════════════════
#  7. Health Score Model (Weighted Multi-Factor)
# ══════════════════════════════════════════════════════════

def compute_health_score(metrics: dict) -> dict:
    """
    Compute a 0-100 health score from multiple portfolio metrics.

    Factors and weights:
      Win Rate (20%):        >55% = 100, <40% = 0
      Profit Factor (20%):   >2.0 = 100, <0.8 = 0
      Sharpe Ratio (15%):    >2.0 = 100, <0 = 0
      Max Drawdown (15%):    <5% = 100, >20% = 0
      Avg R-Multiple (15%):  >0.5 = 100, <-0.5 = 0
      Risk/Reward (10%):     >1.5 = 100, <1.0 = 0
      Consistency (5%):      Low PnL std = 100
    """
    def _score(value, low, high):
        """Linear interpolation: low→0, high→100, clamped."""
        if high == low:
            return 50
        return max(0, min(100, (value - low) / (high - low) * 100))

    scores = {}
    weights = {}

    # Win Rate
    wr = metrics.get("win_rate", 0.5)
    scores["win_rate"] = _score(wr, 0.35, 0.60)
    weights["win_rate"] = 20

    # Profit Factor
    pf = metrics.get("profit_factor", 1.0)
    scores["profit_factor"] = _score(pf, 0.8, 2.5)
    weights["profit_factor"] = 20

    # Sharpe Ratio
    sharpe = metrics.get("sharpe_ratio", 0)
    scores["sharpe_ratio"] = _score(sharpe, -0.5, 2.5)
    weights["sharpe_ratio"] = 15

    # Max Drawdown (inverted — lower is better)
    mdd = abs(metrics.get("max_drawdown_pct", 0))
    scores["max_drawdown"] = _score(-mdd, -25, -3)
    weights["max_drawdown"] = 15

    # Average R-Multiple
    avg_r = metrics.get("avg_r_multiple", 0)
    scores["avg_r_multiple"] = _score(avg_r, -0.5, 0.8)
    weights["avg_r_multiple"] = 15

    # Risk/Reward Ratio
    rr = metrics.get("avg_risk_reward", 1.0)
    scores["risk_reward"] = _score(rr, 0.8, 2.0)
    weights["risk_reward"] = 10

    # Consistency (lower std of daily returns = more consistent)
    pnl_std = metrics.get("daily_pnl_std", 0)
    avg_pnl = abs(metrics.get("avg_daily_pnl", 1))
    cv = pnl_std / avg_pnl if avg_pnl > 0 else 5  # coefficient of variation
    scores["consistency"] = _score(-cv, -5, -0.5)
    weights["consistency"] = 5

    # Weighted average
    total_weight = sum(weights.values())
    overall = sum(scores[k] * weights[k] for k in scores) / total_weight

    # Grade
    if overall >= 80:
        grade = "Excellent"
    elif overall >= 65:
        grade = "Good"
    elif overall >= 50:
        grade = "Fair"
    elif overall >= 35:
        grade = "Poor"
    else:
        grade = "Critical"

    return {
        "overall_score": round(overall, 1),
        "grade": grade,
        "component_scores": {k: round(v, 1) for k, v in scores.items()},
        "component_weights": weights,
    }
