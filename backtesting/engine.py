"""
Backtesting Engine for ICT / SMC Strategy
==========================================
Runs the strategy on historical data day-by-day, simulates trade outcomes,
and returns performance metrics.

Usage:
    from backtesting.engine import Backtester
    bt = Backtester("EURUSD=X", "2026-01-26", "2026-01-31",
                    interval="1h", lookback="3mo")
    report = bt.run()
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from data.fetcher import StockDataFetcher
from strategies.smc_strategy import SMCStrategy
from models.signals import TradeAction


# ──────────────────────────────────────────────────────────
#  Data models
# ──────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """A single simulated trade."""
    ticker: str
    date: str                          # signal date
    action: str                        # BUY / SELL / STRONG BUY / STRONG SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    planned_rr: float                  # planned risk:reward
    outcome: str = "OPEN"              # WIN / LOSS / TIMEOUT / OPEN
    exit_price: float = 0.0
    exit_date: str = ""
    pnl_pct: float = 0.0              # % return on the trade
    actual_rr: float = 0.0            # achieved risk:reward
    bars_held: int = 0
    score: int = 0
    bias: str = ""
    signals_count: int = 0
    confidence: str = ""


@dataclass
class BacktestReport:
    """Aggregated backtest results."""
    ticker: str
    start_date: str
    end_date: str
    interval: str
    total_signals: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    total_pnl_pct: float = 0.0
    avg_rr_achieved: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_bars_held: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)


# ──────────────────────────────────────────────────────────
#  Backtester
# ──────────────────────────────────────────────────────────

class Backtester:
    """
    Run the SMC strategy on historical data and simulate trade outcomes.

    Parameters
    ----------
    ticker      : ticker symbol
    start_date  : first day of the backtest window (YYYY-MM-DD)
    end_date    : last day of the backtest window (YYYY-MM-DD)
    interval    : candle interval ('1h', '1d', etc.)
    lookback    : how much extra history to fetch before start_date so the
                  strategy has enough data for structure detection
    stock_mode  : use stock-mode overrides
    max_hold    : max bars to hold a trade before timing out
    """

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1h",
        lookback: str = "3mo",
        stock_mode: bool = False,
        max_hold: int = 50,
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.lookback = lookback
        self.stock_mode = stock_mode
        self.max_hold = max_hold

    # ── public API ────────────────────────────────────────

    def run(self) -> BacktestReport:
        """Execute the backtest and return a report."""
        report = BacktestReport(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.interval,
        )

        # 1. Fetch a wide window of data
        full_df = self._fetch_data()
        if full_df is None or len(full_df) < 30:
            return report

        # 2. Find the bar indices that fall within the backtest window
        bt_start = pd.Timestamp(self.start_date)
        bt_end = pd.Timestamp(self.end_date) + pd.Timedelta(days=1)  # inclusive

        signal_mask = (full_df.index >= bt_start) & (full_df.index < bt_end)
        signal_indices = full_df.index[signal_mask]

        if len(signal_indices) == 0:
            return report

        # 3. For each trading day in the window, run the strategy on all
        #    data up to that point and record signals
        unique_days = sorted(set(idx.date() for idx in signal_indices))

        for day in unique_days:
            # Slice data: everything up to end of this day
            day_end = pd.Timestamp(day) + pd.Timedelta(days=1)
            history = full_df[full_df.index < day_end]

            if len(history) < 30:
                continue

            # Run strategy on the history
            try:
                strat = SMCStrategy(
                    history, ticker=self.ticker, stock_mode=self.stock_mode
                ).run()
            except Exception:
                continue

            setup = strat.trade_setup
            if setup is None:
                continue

            # Only count actionable signals
            if setup.action in (TradeAction.HOLD,):
                report.total_signals += 1
                continue

            report.total_signals += 1

            # Determine confidence
            warned = sum(1 for s in setup.signals if "⚠" in s.details)
            if warned == 0:
                conf = "HIGH"
            elif warned <= 2:
                conf = "MODERATE"
            else:
                conf = "LOW"

            trade = BacktestTrade(
                ticker=self.ticker,
                date=str(day),
                action=setup.action.value,
                entry_price=setup.entry_price,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit,
                planned_rr=setup.risk_reward,
                score=setup.composite_score,
                bias=setup.bias.value,
                signals_count=len(setup.signals),
                confidence=conf,
            )

            # 4. Simulate the trade outcome using future bars
            self._simulate_trade(trade, full_df, day_end)
            report.trades.append(trade)

        # 5. Compute aggregate metrics
        self._compute_metrics(report)
        return report

    # ── internals ─────────────────────────────────────────

    def _fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch data with enough lookback before the backtest window."""
        try:
            # Calculate a start date with lookback buffer
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
            # Map lookback to approximate days
            lookback_days = {
                "1mo": 35, "2mo": 65, "3mo": 95, "6mo": 185, "1y": 370,
            }.get(self.lookback, 95)
            fetch_start = (start_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

            # End date: add buffer after backtest window for trade simulation
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            fetch_end = (end_dt + timedelta(days=30)).strftime("%Y-%m-%d")

            fetcher = StockDataFetcher(self.ticker)
            df = fetcher.fetch(
                start=fetch_start,
                end=fetch_end,
                interval=self.interval,
            )
            return df
        except Exception:
            return None

    def _simulate_trade(
        self, trade: BacktestTrade, df: pd.DataFrame, entry_time: pd.Timestamp
    ):
        """
        Walk forward from entry_time bar by bar.
        Check if price hits TP or SL first.
        """
        future = df[df.index >= entry_time]

        if len(future) == 0:
            trade.outcome = "OPEN"
            return

        is_long = "BUY" in trade.action
        bars = 0

        for idx, row in future.iterrows():
            bars += 1

            if is_long:
                # Check SL first (worst case)
                if row["Low"] <= trade.stop_loss:
                    trade.outcome = "LOSS"
                    trade.exit_price = trade.stop_loss
                    trade.exit_date = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                    trade.bars_held = bars
                    break
                # Check TP
                if row["High"] >= trade.take_profit:
                    trade.outcome = "WIN"
                    trade.exit_price = trade.take_profit
                    trade.exit_date = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                    trade.bars_held = bars
                    break
            else:
                # Short: check SL first
                if row["High"] >= trade.stop_loss:
                    trade.outcome = "LOSS"
                    trade.exit_price = trade.stop_loss
                    trade.exit_date = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                    trade.bars_held = bars
                    break
                # Check TP
                if row["Low"] <= trade.take_profit:
                    trade.outcome = "WIN"
                    trade.exit_price = trade.take_profit
                    trade.exit_date = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                    trade.bars_held = bars
                    break

            if bars >= self.max_hold:
                trade.outcome = "TIMEOUT"
                trade.exit_price = row["Close"]
                trade.exit_date = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                trade.bars_held = bars
                break
        else:
            # Ran out of future data
            if len(future) > 0:
                last = future.iloc[-1]
                trade.outcome = "TIMEOUT"
                trade.exit_price = last["Close"]
                trade.exit_date = str(future.index[-1].date()) if hasattr(future.index[-1], 'date') else str(future.index[-1])
                trade.bars_held = len(future)

        # Calculate P&L
        if trade.exit_price > 0 and trade.entry_price > 0:
            is_long = "BUY" in trade.action
            if is_long:
                trade.pnl_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
            else:
                trade.pnl_pct = ((trade.entry_price - trade.exit_price) / trade.entry_price) * 100

            # Actual R:R achieved
            risk = abs(trade.entry_price - trade.stop_loss)
            if risk > 0:
                if is_long:
                    reward = trade.exit_price - trade.entry_price
                else:
                    reward = trade.entry_price - trade.exit_price
                trade.actual_rr = reward / risk

    def _compute_metrics(self, report: BacktestReport):
        """Compute aggregate metrics from individual trades."""
        report.total_trades = len(report.trades)
        if report.total_trades == 0:
            return

        report.wins = sum(1 for t in report.trades if t.outcome == "WIN")
        report.losses = sum(1 for t in report.trades if t.outcome == "LOSS")
        report.timeouts = sum(1 for t in report.trades if t.outcome == "TIMEOUT")

        report.win_rate = (report.wins / report.total_trades) * 100 if report.total_trades > 0 else 0

        pnls = [t.pnl_pct for t in report.trades]
        report.avg_pnl_pct = np.mean(pnls) if pnls else 0
        report.total_pnl_pct = np.sum(pnls) if pnls else 0
        report.best_trade_pnl = max(pnls) if pnls else 0
        report.worst_trade_pnl = min(pnls) if pnls else 0

        rrs = [t.actual_rr for t in report.trades if t.outcome in ("WIN", "LOSS")]
        report.avg_rr_achieved = np.mean(rrs) if rrs else 0

        bars = [t.bars_held for t in report.trades]
        report.avg_bars_held = np.mean(bars) if bars else 0


# ──────────────────────────────────────────────────────────
#  Multi-ticker backtester
# ──────────────────────────────────────────────────────────

def backtest_session(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1h",
    lookback: str = "3mo",
    stock_mode: bool = False,
    max_hold: int = 50,
    progress_callback=None,
) -> List[BacktestReport]:
    """
    Run backtest for multiple tickers. Returns a list of BacktestReports.

    Parameters
    ----------
    progress_callback : optional callable(i, total, ticker) for progress bars
    """
    reports = []
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, total, ticker)
        bt = Backtester(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            lookback=lookback,
            stock_mode=stock_mode,
            max_hold=max_hold,
        )
        reports.append(bt.run())
    return reports
