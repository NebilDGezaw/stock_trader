"""
Backtesting Engine for ICT / SMC Strategy
==========================================
Comprehensive bar-by-bar backtester using a rolling window approach.
Feeds historical data through the existing SMCStrategy class and
simulates position management with SL/TP/signal-based exits.

Usage:
    from bt_engine.engine import BacktestEngine
    engine = BacktestEngine("SPY", period="2y", interval="1d", stock_mode=True)
    engine.run()
    print(engine.summary())
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import config
from data.fetcher import StockDataFetcher
from strategies.smc_strategy import SMCStrategy
from models.signals import TradeAction, MarketBias
from utils.helpers import compute_atr

# Default strategy class — can be overridden
_DEFAULT_STRATEGY = SMCStrategy


# ──────────────────────────────────────────────────────────
#  Trade Record
# ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker: str
    direction: str              # "long" or "short"
    entry_date: object          # timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    risk_per_share: float
    exit_date: object = None
    exit_price: float = 0.0
    exit_reason: str = ""       # "SL", "TP", "signal", "end"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    # Trailing stop state
    original_sl: float = 0.0
    highest_price: float = 0.0  # for long trailing
    lowest_price: float = 0.0   # for short trailing
    trailing_active: bool = False
    atr_at_entry: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "entry_date": str(self.entry_date),
            "exit_date": str(self.exit_date) if self.exit_date else "",
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "stop_loss": round(self.stop_loss, 4),
            "take_profit": round(self.take_profit, 4),
            "position_size": self.position_size,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "r_multiple": round(self.r_multiple, 2),
            "exit_reason": self.exit_reason,
        }


# ──────────────────────────────────────────────────────────
#  Backtest Engine
# ──────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Comprehensive backtesting engine using rolling-window SMCStrategy.

    Parameters
    ----------
    ticker      : symbol to backtest
    period      : yfinance period string (e.g. '2y')
    interval    : candle interval ('1d', '1h', etc.)
    stock_mode  : use stock-mode overrides in SMCStrategy
    window      : rolling window size (bars of history fed to strategy)
    initial_capital : starting equity
    risk_per_trade  : fraction of capital risked per trade
    max_positions   : max concurrent open positions
    """

    def __init__(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        stock_mode: bool = False,
        window: int = 120,
        initial_capital: float = None,
        risk_per_trade: float = None,
        max_positions: int = None,
        strategy_class=None,
    ):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.stock_mode = stock_mode
        self.window = window
        self.strategy_class = strategy_class or _DEFAULT_STRATEGY
        self.capital = initial_capital or config.INITIAL_CAPITAL
        # Use leveraged risk if running leveraged strategy
        if strategy_class is not None:
            from strategies.leveraged_momentum import LeveragedMomentumStrategy
            if strategy_class is LeveragedMomentumStrategy:
                self.risk_pct = risk_per_trade or config.LEVERAGED_MODE.get("risk_per_trade", config.RISK_PER_TRADE)
            else:
                self.risk_pct = risk_per_trade or config.RISK_PER_TRADE
        else:
            self.risk_pct = risk_per_trade or config.RISK_PER_TRADE
        self.max_positions = max_positions or config.MAX_OPEN_POSITIONS

        self._initial_capital = self.capital
        self._equity = self.capital
        self._trades: List[Trade] = []
        self._open_positions: List[Trade] = []
        self._equity_curve: List[dict] = []
        self._df: Optional[pd.DataFrame] = None
        self._ran = False
        self._last_tp_exit_bar: int = -999  # track last TP exit bar index

        # Detect forex mode from strategy class
        from strategies.forex_ict import ForexICTStrategy
        self._forex = (self.strategy_class == ForexICTStrategy)
        if self._forex:
            self.risk_pct = config.FOREX_MODE.get("risk_per_trade", self.risk_pct)

        # Detect leveraged/crypto mode from strategy class
        from strategies.leveraged_momentum import LeveragedMomentumStrategy
        from strategies.crypto_momentum import CryptoMomentumStrategy
        self._leveraged = (self.strategy_class == LeveragedMomentumStrategy)
        self._crypto = (self.strategy_class == CryptoMomentumStrategy)

        if self._crypto:
            crypto_cfg = config.CRYPTO_MODE
            self.risk_pct = risk_per_trade or crypto_cfg.get("risk_per_trade", config.RISK_PER_TRADE)
            self._trailing_enabled = crypto_cfg.get("trailing_stop", True)
            self._trail_activation_atr = crypto_cfg.get("trail_activation_atr", 1.5)
            self._trail_distance_atr = crypto_cfg.get("trail_distance_atr", 1.0)
        elif self._leveraged:
            lev_cfg = config.LEVERAGED_MODE
            self.risk_pct = lev_cfg.get("risk_per_trade", self.risk_pct)
            self._trailing_enabled = lev_cfg.get("trailing_stop", False)
            self._trail_activation_atr = lev_cfg.get("trail_activation_atr", 1.0)
            self._trail_distance_atr = lev_cfg.get("trail_distance_atr", 0.75)
        else:
            self._trailing_enabled = False
            self._trail_activation_atr = 1.0
            self._trail_distance_atr = 0.75

    # ── public API ────────────────────────────────────────

    def run(self) -> "BacktestEngine":
        """Execute the backtest."""
        # Fetch data
        fetcher = StockDataFetcher(self.ticker)
        self._df = fetcher.fetch(period=self.period, interval=self.interval)

        if len(self._df) < self.window + 10:
            print(f"[!] Not enough data for {self.ticker}: {len(self._df)} bars (need {self.window + 10})")
            self._ran = True
            return self

        df = self._df

        # Walk bar-by-bar starting from `window`
        for i in range(self.window, len(df)):
            bar = df.iloc[i]
            bar_date = df.index[i]
            window_df = df.iloc[i - self.window : i + 1].copy()

            # 1. Check open positions against this bar's price action
            self._check_exits(bar, bar_date)

            # 2. Run strategy on the rolling window
            try:
                strat = self.strategy_class(
                    window_df, ticker=self.ticker, stock_mode=self.stock_mode
                ).run()
            except Exception:
                self._record_equity(bar_date)
                continue

            setup = strat.trade_setup
            if setup is None:
                self._record_equity(bar_date)
                continue

            action = setup.action

            # 3. Process signals
            if action in (TradeAction.BUY, TradeAction.STRONG_BUY):
                self._handle_buy_signal(setup, bar, bar_date)
            elif action in (TradeAction.SELL, TradeAction.STRONG_SELL):
                self._handle_sell_signal(setup, bar, bar_date)

            self._record_equity(bar_date)

        # Close any remaining open positions at last bar
        if len(df) > 0:
            last_bar = df.iloc[-1]
            last_date = df.index[-1]
            for pos in list(self._open_positions):
                self._close_position(pos, last_bar["Close"], last_date, "end")

        self._ran = True
        return self

    @property
    def trades(self) -> List[dict]:
        return [t.to_dict() for t in self._trades]

    @property
    def equity_curve(self) -> pd.DataFrame:
        if not self._equity_curve:
            return pd.DataFrame(columns=["date", "equity"])
        ec = pd.DataFrame(self._equity_curve)
        return ec

    def to_csv(self, path: str):
        """Export trades to CSV."""
        pd.DataFrame(self.trades).to_csv(path, index=False)

    def summary(self) -> str:
        """Return a formatted performance report."""
        if not self._ran:
            return "Backtest has not been run yet."

        closed = self._trades
        if not closed:
            return (
                f"{'═' * 60}\n"
                f"  BACKTEST: {self.ticker} ({self.period}, {self.interval})\n"
                f"{'═' * 60}\n"
                f"  No trades generated.\n"
                f"{'═' * 60}"
            )

        # Basic stats
        total = len(closed)
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total * 100) if total > 0 else 0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        total_pnl = sum(t.pnl for t in closed)
        total_return = (self._equity - self._initial_capital) / self._initial_capital * 100

        # R-multiples
        r_multiples = [t.r_multiple for t in closed]
        avg_r = np.mean(r_multiples) if r_multiples else 0

        # Best/worst
        best = max(closed, key=lambda t: t.pnl)
        worst = min(closed, key=lambda t: t.pnl)

        # Drawdown
        ec = self.equity_curve
        max_dd, max_dd_dur = self._compute_drawdown(ec)

        # Sharpe & Sortino
        sharpe = self._compute_sharpe(ec)
        sortino = self._compute_sortino(ec)

        # CAGR
        cagr = self._compute_cagr(ec)

        # Monthly returns
        monthly = self._compute_monthly_returns(ec)

        # Exit reasons
        exit_reasons = {}
        for t in closed:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        lines = [
            f"{'═' * 60}",
            f"  BACKTEST REPORT: {self.ticker}",
            f"  Period: {self.period} | Interval: {self.interval} | Stock Mode: {self.stock_mode}",
            f"{'═' * 60}",
            f"",
            f"  PERFORMANCE",
            f"  {'─' * 40}",
            f"  Initial Capital   : ${self._initial_capital:,.2f}",
            f"  Final Equity      : ${self._equity:,.2f}",
            f"  Total Return      : {total_return:+.2f}%",
            f"  CAGR              : {cagr:+.2f}%",
            f"  Total P&L         : ${total_pnl:+,.2f}",
            f"",
            f"  RISK METRICS",
            f"  {'─' * 40}",
            f"  Sharpe Ratio      : {sharpe:.2f}",
            f"  Sortino Ratio     : {sortino:.2f}",
            f"  Max Drawdown      : {max_dd:.2f}%",
            f"  Max DD Duration   : {max_dd_dur} bars",
            f"  Profit Factor     : {profit_factor:.2f}",
            f"",
            f"  TRADE STATISTICS",
            f"  {'─' * 40}",
            f"  Total Trades      : {total}",
            f"  Wins              : {win_count} ({win_rate:.1f}%)",
            f"  Losses            : {loss_count}",
            f"  Avg Win           : ${avg_win:+,.2f}",
            f"  Avg Loss          : ${avg_loss:+,.2f}",
            f"  Avg R-Multiple    : {avg_r:+.2f}R",
            f"  Best Trade        : ${best.pnl:+,.2f} ({best.r_multiple:+.2f}R) on {best.entry_date}",
            f"  Worst Trade       : ${worst.pnl:+,.2f} ({worst.r_multiple:+.2f}R) on {worst.entry_date}",
            f"",
            f"  EXIT REASONS",
            f"  {'─' * 40}",
        ]
        for reason, count in sorted(exit_reasons.items()):
            lines.append(f"  {reason:<20}: {count}")

        if monthly:
            lines.append(f"")
            lines.append(f"  MONTHLY RETURNS")
            lines.append(f"  {'─' * 40}")
            for month, ret in monthly.items():
                lines.append(f"  {month}  : {ret:+.2f}%")

        lines.append(f"{'═' * 60}")
        return "\n".join(lines)

    # ── position management ───────────────────────────────

    def _handle_buy_signal(self, setup, bar, bar_date):
        """Open a long position (or close shorts in non-stock mode)."""
        # In stock_mode, close any short positions on buy signal
        if self.stock_mode:
            # Stocks: only long. Close existing longs on opposing signal handled in sell.
            pass
        else:
            # Close shorts on buy signal
            for pos in list(self._open_positions):
                if pos.direction == "short":
                    self._close_position(pos, bar["Close"], bar_date, "signal")

        # Check if we can open a new long
        long_count = sum(1 for p in self._open_positions if p.direction == "long")
        if long_count >= self.max_positions:
            return

        # Don't duplicate: skip if we already have a long for this ticker
        if any(p.direction == "long" and p.ticker == self.ticker for p in self._open_positions):
            return

        self._open_long(setup, bar, bar_date)

    def _handle_sell_signal(self, setup, bar, bar_date):
        """Close longs (stock mode) or open shorts."""
        # Close any open longs
        for pos in list(self._open_positions):
            if pos.direction == "long" and pos.ticker == self.ticker:
                self._close_position(pos, bar["Close"], bar_date, "signal")

        if self.stock_mode:
            return  # No shorting in stock mode

        # Open short if allowed
        short_count = sum(1 for p in self._open_positions if p.direction == "short")
        if short_count >= self.max_positions:
            return
        if any(p.direction == "short" and p.ticker == self.ticker for p in self._open_positions):
            return

        self._open_short(setup, bar, bar_date)

    def _open_long(self, setup, bar, bar_date):
        entry = bar["Close"]
        sl = setup.stop_loss
        tp = setup.take_profit

        # Ensure SL is below entry for long
        if sl >= entry:
            sl = entry * 0.97

        risk_per_share = abs(entry - sl)
        if risk_per_share == 0:
            return

        risk_amount = self._equity * self.risk_pct
        size = int(risk_amount / risk_per_share)
        if size <= 0:
            return

        # Don't exceed available equity
        cost = size * entry
        if cost > self._equity:
            size = int(self._equity / entry)
            if size <= 0:
                return

        trade = Trade(
            ticker=self.ticker,
            direction="long",
            entry_date=bar_date,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            position_size=size,
            risk_per_share=risk_per_share,
            original_sl=sl,
            highest_price=entry,
            lowest_price=entry,
            atr_at_entry=self._get_current_atr(bar_date) if self._trailing_enabled else 0.0,
        )
        self._open_positions.append(trade)

    def _open_short(self, setup, bar, bar_date):
        entry = bar["Close"]
        sl = setup.stop_loss
        tp = setup.take_profit

        # Ensure SL is above entry for short
        if sl <= entry:
            sl = entry * 1.03

        risk_per_share = abs(sl - entry)
        if risk_per_share == 0:
            return

        risk_amount = self._equity * self.risk_pct
        size = int(risk_amount / risk_per_share)
        if size <= 0:
            return

        trade = Trade(
            ticker=self.ticker,
            direction="short",
            entry_date=bar_date,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            position_size=size,
            risk_per_share=risk_per_share,
            original_sl=sl,
            highest_price=entry,
            lowest_price=entry,
            atr_at_entry=self._get_current_atr(bar_date) if self._trailing_enabled else 0.0,
        )
        self._open_positions.append(trade)

    def _check_exits(self, bar, bar_date):
        """Check if any open position hits SL or TP on this bar.
        Uses per-trade ATR for trailing stops (no recomputation per bar)."""
        for pos in list(self._open_positions):
            if pos.direction == "long":
                # Update high watermark
                if bar["High"] > pos.highest_price:
                    pos.highest_price = bar["High"]

                # Trailing stop using stored ATR
                if self._trailing_enabled and pos.atr_at_entry > 0:
                    atr = pos.atr_at_entry
                    profit_atr = (pos.highest_price - pos.entry_price) / atr

                    # Stage 1: breakeven after 1x ATR
                    if profit_atr >= self._trail_activation_atr:
                        be_sl = pos.entry_price
                        if be_sl > pos.stop_loss:
                            pos.stop_loss = be_sl
                            pos.trailing_active = True

                    # Stage 2: trail at 0.75x ATR behind peak after 1.5x ATR
                    if profit_atr >= self._trail_activation_atr * 1.5 and pos.trailing_active:
                        new_sl = pos.highest_price - (atr * self._trail_distance_atr)
                        if new_sl > pos.stop_loss:
                            pos.stop_loss = new_sl

                # Check exits
                if bar["Low"] <= pos.stop_loss:
                    reason = "trailing_SL" if pos.trailing_active else "SL"
                    self._close_position(pos, pos.stop_loss, bar_date, reason)
                elif bar["High"] >= pos.take_profit:
                    self._close_position(pos, pos.take_profit, bar_date, "TP")

            else:  # short
                # Update low watermark
                if pos.lowest_price == 0 or bar["Low"] < pos.lowest_price:
                    pos.lowest_price = bar["Low"]

                if self._trailing_enabled and pos.atr_at_entry > 0:
                    atr = pos.atr_at_entry
                    profit_atr = (pos.entry_price - pos.lowest_price) / atr

                    if profit_atr >= self._trail_activation_atr:
                        be_sl = pos.entry_price
                        if be_sl < pos.stop_loss:
                            pos.stop_loss = be_sl
                            pos.trailing_active = True

                    if profit_atr >= self._trail_activation_atr * 1.5 and pos.trailing_active:
                        new_sl = pos.lowest_price + (atr * self._trail_distance_atr)
                        if new_sl < pos.stop_loss:
                            pos.stop_loss = new_sl

                if bar["High"] >= pos.stop_loss:
                    reason = "trailing_SL" if pos.trailing_active else "SL"
                    self._close_position(pos, pos.stop_loss, bar_date, reason)
                elif bar["Low"] <= pos.take_profit:
                    self._close_position(pos, pos.take_profit, bar_date, "TP")

    def _is_leveraged_strategy(self) -> bool:
        """Check if we're using the leveraged strategy."""
        from strategies.leveraged_momentum import LeveragedMomentumStrategy
        return self.strategy_class is LeveragedMomentumStrategy

    def _get_current_atr(self, bar_date, period: int = 14) -> float:
        """Get ATR at the current bar for trailing stop."""
        if self._df is None:
            return 0.0
        try:
            idx = self._df.index.get_loc(bar_date)
            if idx < period:
                return 0.0
            from utils.helpers import compute_atr
            atr_series = compute_atr(self._df.iloc[:idx + 1], period=period)
            return float(atr_series.iloc[-1])
        except Exception:
            return 0.0

    def _close_position(self, pos: Trade, exit_price: float, exit_date, reason: str):
        """Close a position and record the trade."""
        pos.exit_price = exit_price
        pos.exit_date = exit_date
        pos.exit_reason = reason

        if pos.direction == "long":
            pos.pnl = (exit_price - pos.entry_price) * pos.position_size
        else:
            pos.pnl = (pos.entry_price - exit_price) * pos.position_size

        pos.pnl_pct = pos.pnl / (pos.entry_price * pos.position_size) * 100 if pos.entry_price else 0

        if pos.risk_per_share > 0:
            if pos.direction == "long":
                pos.r_multiple = (exit_price - pos.entry_price) / pos.risk_per_share
            else:
                pos.r_multiple = (pos.entry_price - exit_price) / pos.risk_per_share

        self._equity += pos.pnl
        self._trades.append(pos)
        if pos in self._open_positions:
            self._open_positions.remove(pos)

    def _record_equity(self, bar_date):
        """Snapshot equity including unrealized P&L."""
        unrealized = 0
        if self._df is not None and len(self._df) > 0:
            # Use bar_date price for open positions
            try:
                bar = self._df.loc[bar_date]
                price = bar["Close"] if not isinstance(bar, pd.DataFrame) else bar["Close"].iloc[0]
                for pos in self._open_positions:
                    if pos.direction == "long":
                        unrealized += (price - pos.entry_price) * pos.position_size
                    else:
                        unrealized += (pos.entry_price - price) * pos.position_size
            except (KeyError, IndexError):
                pass

        self._equity_curve.append({
            "date": bar_date,
            "equity": self._equity + unrealized,
        })

    # ── metrics ───────────────────────────────────────────

    def _compute_drawdown(self, ec: pd.DataFrame):
        """Max drawdown % and duration in bars."""
        if ec.empty:
            return 0.0, 0
        equity = ec["equity"].values
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak * 100
        max_dd = float(np.min(dd))

        # Duration: longest streak below previous peak
        in_dd = equity < peak
        max_dur = 0
        cur_dur = 0
        for v in in_dd:
            if v:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        return max_dd, max_dur

    def _compute_sharpe(self, ec: pd.DataFrame, rf: float = 0.0) -> float:
        """Annualized Sharpe ratio."""
        if len(ec) < 2:
            return 0.0
        returns = ec["equity"].pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        # Annualize based on interval
        factor = self._annualization_factor()
        excess = returns.mean() - rf / factor
        return float(excess / returns.std() * np.sqrt(factor))

    def _compute_sortino(self, ec: pd.DataFrame, rf: float = 0.0) -> float:
        """Annualized Sortino ratio."""
        if len(ec) < 2:
            return 0.0
        returns = ec["equity"].pct_change().dropna()
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        factor = self._annualization_factor()
        excess = returns.mean() - rf / factor
        return float(excess / downside.std() * np.sqrt(factor))

    def _compute_cagr(self, ec: pd.DataFrame) -> float:
        """Compound annual growth rate."""
        if ec.empty or len(ec) < 2:
            return 0.0
        start_eq = ec["equity"].iloc[0]
        end_eq = ec["equity"].iloc[-1]
        if start_eq <= 0:
            return 0.0

        start_date = ec["date"].iloc[0]
        end_date = ec["date"].iloc[-1]
        try:
            years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
        except Exception:
            years = len(ec) / self._annualization_factor()

        if years <= 0:
            return 0.0
        return float(((end_eq / start_eq) ** (1 / years) - 1) * 100)

    def _compute_monthly_returns(self, ec: pd.DataFrame) -> Dict[str, float]:
        """Monthly return breakdown."""
        if ec.empty:
            return {}
        ec2 = ec.copy()
        ec2["date"] = pd.to_datetime(ec2["date"])
        ec2 = ec2.set_index("date")

        monthly = {}
        for name, group in ec2["equity"].groupby(pd.Grouper(freq="ME")):
            if len(group) < 2:
                continue
            ret = (group.iloc[-1] / group.iloc[0] - 1) * 100
            monthly[name.strftime("%Y-%m")] = round(ret, 2)
        return monthly

    def _annualization_factor(self) -> float:
        """Approximate number of bars per year."""
        mapping = {
            "1m": 252 * 390, "5m": 252 * 78, "15m": 252 * 26,
            "30m": 252 * 13, "1h": 252 * 6.5, "60m": 252 * 6.5,
            "1d": 252, "5d": 52, "1wk": 52, "1mo": 12,
        }
        return mapping.get(self.interval, 252)


# ──────────────────────────────────────────────────────────
#  Multi-ticker comparison
# ──────────────────────────────────────────────────────────

def run_multi(
    tickers: List[str],
    period: str = "2y",
    interval: str = "1d",
    stock_mode: bool = False,
    window: int = 120,
    strategy_class=None,
) -> Dict[str, BacktestEngine]:
    """Run backtests for multiple tickers and return dict of engines."""
    results = {}
    for ticker in tickers:
        print(f"  Backtesting {ticker}...")
        try:
            engine = BacktestEngine(
                ticker, period=period, interval=interval,
                stock_mode=stock_mode, window=window,
                strategy_class=strategy_class,
            )
            engine.run()
            results[ticker] = engine
        except Exception as e:
            print(f"  [!] {ticker} failed: {e}")
    return results


def comparison_table(engines: Dict[str, BacktestEngine]) -> str:
    """Format a comparison table across multiple tickers."""
    lines = [
        f"{'═' * 90}",
        f"  {'TICKER':<8} {'TRADES':>6} {'WIN%':>6} {'TOTAL RET':>10} "
        f"{'SHARPE':>7} {'MAX DD':>8} {'PF':>6} {'AVG R':>6}",
        f"{'─' * 90}",
    ]
    for ticker, eng in engines.items():
        closed = eng._trades
        total = len(closed)
        wins = sum(1 for t in closed if t.pnl > 0)
        wr = (wins / total * 100) if total else 0
        ret = (eng._equity - eng._initial_capital) / eng._initial_capital * 100
        ec = eng.equity_curve
        sharpe = eng._compute_sharpe(ec)
        dd, _ = eng._compute_drawdown(ec)
        gross_w = sum(t.pnl for t in closed if t.pnl > 0)
        gross_l = abs(sum(t.pnl for t in closed if t.pnl <= 0))
        pf = gross_w / gross_l if gross_l > 0 else float('inf')
        avg_r = np.mean([t.r_multiple for t in closed]) if closed else 0

        lines.append(
            f"  {ticker:<8} {total:>6} {wr:>5.1f}% {ret:>+9.2f}% "
            f"{sharpe:>7.2f} {dd:>+7.2f}% {pf:>6.2f} {avg_r:>+5.2f}R"
        )
    lines.append(f"{'═' * 90}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
#  Compatibility layer — old API used by the dashboard UI
# ══════════════════════════════════════════════════════════

@dataclass
class BacktestTrade:
    """A single simulated trade (old API)."""
    ticker: str
    date: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    planned_rr: float = 0.0
    outcome: str = "OPEN"
    exit_price: float = 0.0
    exit_date: str = ""
    pnl_pct: float = 0.0
    actual_rr: float = 0.0
    bars_held: int = 0
    score: int = 0
    bias: str = ""
    signals_count: int = 0
    confidence: str = ""


@dataclass
class BacktestReport:
    """Aggregated backtest results (old API)."""
    ticker: str
    start_date: str = ""
    end_date: str = ""
    interval: str = ""
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
    trades: list = field(default_factory=list)


class Backtester:
    """Old-API backtester used by the dashboard."""

    def __init__(self, ticker, start_date, end_date, interval="1h",
                 lookback="3mo", stock_mode=False, max_hold=50):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.lookback = lookback
        self.stock_mode = stock_mode
        self.max_hold = max_hold

    def run(self):
        report = BacktestReport(
            ticker=self.ticker, start_date=self.start_date,
            end_date=self.end_date, interval=self.interval,
        )
        try:
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
            lookback_days = {"1mo": 35, "2mo": 65, "3mo": 95,
                             "6mo": 185, "1y": 370}.get(self.lookback, 95)
            fetch_start = (start_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
            fetch_end = (end_dt + timedelta(days=30)).strftime("%Y-%m-%d")
            fetcher = StockDataFetcher(self.ticker)
            full_df = fetcher.fetch(start=fetch_start, end=fetch_end,
                                    interval=self.interval)
        except Exception:
            return report

        if full_df is None or len(full_df) < 30:
            return report

        bt_start = pd.Timestamp(self.start_date)
        bt_end = pd.Timestamp(self.end_date) + pd.Timedelta(days=1)
        signal_mask = (full_df.index >= bt_start) & (full_df.index < bt_end)
        signal_indices = full_df.index[signal_mask]
        if len(signal_indices) == 0:
            return report

        unique_days = sorted(set(idx.date() for idx in signal_indices))
        for day in unique_days:
            day_end = pd.Timestamp(day) + pd.Timedelta(days=1)
            history = full_df[full_df.index < day_end]
            if len(history) < 30:
                continue
            try:
                strat = SMCStrategy(history, ticker=self.ticker,
                                    stock_mode=self.stock_mode).run()
            except Exception:
                continue
            setup = strat.trade_setup
            if setup is None:
                continue
            if setup.action == TradeAction.HOLD:
                report.total_signals += 1
                continue
            report.total_signals += 1
            warned = sum(1 for s in setup.signals if "⚠" in s.details)
            conf = "HIGH" if warned == 0 else ("MODERATE" if warned <= 2 else "LOW")
            trade = BacktestTrade(
                ticker=self.ticker, date=str(day), action=setup.action.value,
                entry_price=setup.entry_price, stop_loss=setup.stop_loss,
                take_profit=setup.take_profit, planned_rr=setup.risk_reward,
                score=setup.composite_score, bias=setup.bias.value,
                signals_count=len(setup.signals), confidence=conf,
            )
            self._simulate_trade(trade, full_df, day_end)
            report.trades.append(trade)

        self._compute_metrics(report)
        return report

    def _simulate_trade(self, trade, df, entry_time):
        future = df[df.index >= entry_time]
        if len(future) == 0:
            trade.outcome = "OPEN"
            return
        is_long = "BUY" in trade.action
        bars = 0
        for idx, row in future.iterrows():
            bars += 1
            if is_long:
                if row["Low"] <= trade.stop_loss:
                    trade.outcome, trade.exit_price = "LOSS", trade.stop_loss
                    trade.exit_date = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                    trade.bars_held = bars
                    break
                if row["High"] >= trade.take_profit:
                    trade.outcome, trade.exit_price = "WIN", trade.take_profit
                    trade.exit_date = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                    trade.bars_held = bars
                    break
            else:
                if row["High"] >= trade.stop_loss:
                    trade.outcome, trade.exit_price = "LOSS", trade.stop_loss
                    trade.exit_date = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                    trade.bars_held = bars
                    break
                if row["Low"] <= trade.take_profit:
                    trade.outcome, trade.exit_price = "WIN", trade.take_profit
                    trade.exit_date = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                    trade.bars_held = bars
                    break
            if bars >= self.max_hold:
                trade.outcome, trade.exit_price = "TIMEOUT", row["Close"]
                trade.exit_date = str(idx.date()) if hasattr(idx, 'date') else str(idx)
                trade.bars_held = bars
                break
        else:
            if len(future) > 0:
                last = future.iloc[-1]
                trade.outcome, trade.exit_price = "TIMEOUT", last["Close"]
                trade.exit_date = str(future.index[-1].date()) if hasattr(future.index[-1], 'date') else str(future.index[-1])
                trade.bars_held = len(future)
        if trade.exit_price > 0 and trade.entry_price > 0:
            if "BUY" in trade.action:
                trade.pnl_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
            else:
                trade.pnl_pct = ((trade.entry_price - trade.exit_price) / trade.entry_price) * 100
            risk = abs(trade.entry_price - trade.stop_loss)
            if risk > 0:
                reward = (trade.exit_price - trade.entry_price) if "BUY" in trade.action else (trade.entry_price - trade.exit_price)
                trade.actual_rr = reward / risk

    def _compute_metrics(self, report):
        report.total_trades = len(report.trades)
        if report.total_trades == 0:
            return
        report.wins = sum(1 for t in report.trades if t.outcome == "WIN")
        report.losses = sum(1 for t in report.trades if t.outcome == "LOSS")
        report.timeouts = sum(1 for t in report.trades if t.outcome == "TIMEOUT")
        report.win_rate = (report.wins / report.total_trades) * 100
        pnls = [t.pnl_pct for t in report.trades]
        report.avg_pnl_pct = np.mean(pnls) if pnls else 0
        report.total_pnl_pct = np.sum(pnls) if pnls else 0
        report.best_trade_pnl = max(pnls) if pnls else 0
        report.worst_trade_pnl = min(pnls) if pnls else 0
        rrs = [t.actual_rr for t in report.trades if t.outcome in ("WIN", "LOSS")]
        report.avg_rr_achieved = np.mean(rrs) if rrs else 0
        bars = [t.bars_held for t in report.trades]
        report.avg_bars_held = np.mean(bars) if bars else 0


def backtest_session(tickers, start_date, end_date, interval="1h",
                     lookback="3mo", stock_mode=False, max_hold=50,
                     progress_callback=None):
    """Run backtest for multiple tickers (old API)."""
    reports = []
    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, len(tickers), ticker)
        bt = Backtester(ticker=ticker, start_date=start_date,
                        end_date=end_date, interval=interval,
                        lookback=lookback, stock_mode=stock_mode,
                        max_hold=max_hold)
        reports.append(bt.run())
    return reports
