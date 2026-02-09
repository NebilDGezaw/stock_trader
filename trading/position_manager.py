"""
Position Manager — monitors and manages open HFM MT5 positions.
================================================================
Handles:
    - Trailing stops (move SL to breakeven after 1R, then trail by ATR)
    - Signal reversal detection (re-run strategy, close if flipped)
    - Partial close (50% at 1R, let remainder run to TP)
    - Position status reporting for Telegram

Designed to run periodically via GitHub Actions (every few hours).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config
from data.fetcher import StockDataFetcher
from models.signals import TradeAction
from trading.mt5_client import MT5Client, PositionInfo
from trading.symbols import to_yfinance, get_asset_type

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Data models
# ──────────────────────────────────────────────────────────

@dataclass
class PositionUpdate:
    """Record of a position management action."""
    ticket: int
    symbol: str
    action_taken: str     # "trail_stop", "partial_close", "full_close", "no_change"
    old_sl: float = 0.0
    new_sl: float = 0.0
    pnl: float = 0.0
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ──────────────────────────────────────────────────────────
#  ATR helper (works without full strategy import)
# ──────────────────────────────────────────────────────────

def _get_current_atr(ticker_yf: str, period: int = 14) -> float:
    """Fetch recent data and compute ATR for a ticker."""
    try:
        asset_type = get_asset_type(ticker_yf)
        if asset_type == "forex":
            df = StockDataFetcher(ticker_yf).fetch(period="1mo", interval="1h")
        else:
            df = StockDataFetcher(ticker_yf).fetch(period="3mo", interval="1d")

        if df is None or len(df) < period + 1:
            return 0.0

        high = df["High"]
        low = df["Low"]
        close = df["Close"].shift(1)
        tr = (high - low).combine(abs(high - close), max).combine(abs(low - close), max)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if atr == atr else 0.0  # NaN check
    except Exception as e:
        logger.warning(f"ATR fetch failed for {ticker_yf}: {e}")
        return 0.0


def _rerun_strategy(ticker_yf: str) -> Optional[TradeAction]:
    """Re-run the appropriate strategy to check if signal has flipped."""
    try:
        from bt_engine.engine import _detect_asset_type, _run_strategy

        asset_type = _detect_asset_type(ticker_yf)

        if asset_type == "forex":
            df = StockDataFetcher(ticker_yf).fetch(period="1mo", interval="1h")
        else:
            df = StockDataFetcher(ticker_yf).fetch(period="6mo", interval="1d")

        if df is None or len(df) < 30:
            return None

        stock_mode = (asset_type == "stocks")
        strat = _run_strategy(df, ticker_yf, stock_mode=stock_mode)

        if strat and strat.trade_setup:
            return strat.trade_setup.action
        return TradeAction.HOLD
    except Exception as e:
        logger.warning(f"Strategy re-run failed for {ticker_yf}: {e}")
        return None


# ──────────────────────────────────────────────────────────
#  Position Manager
# ──────────────────────────────────────────────────────────

class PositionManager:
    """
    Manages open positions placed by the bot.

    Usage:
        client = MT5Client()
        client.connect(...)
        manager = PositionManager(client)
        updates = manager.manage_all()
    """

    def __init__(
        self,
        client: MT5Client,
        trail_activation_r: float = 1.0,
        trail_atr_multiplier: float = 1.0,
        partial_close_at_r: float = 1.0,
        partial_close_pct: float = 0.5,
        enable_reversal_close: bool = True,
        dry_run: bool = False,
    ):
        self.client = client
        self.trail_activation_r = trail_activation_r
        self.trail_atr_multiplier = trail_atr_multiplier
        self.partial_close_at_r = partial_close_at_r
        self.partial_close_pct = partial_close_pct
        self.enable_reversal_close = enable_reversal_close
        self.dry_run = dry_run

    def manage_all(self) -> list[PositionUpdate]:
        """
        Check and manage all open bot positions.
        Returns a list of actions taken.
        """
        positions = self.client.get_open_positions()
        if not positions:
            logger.info("No open bot positions to manage.")
            return []

        updates = []
        for pos in positions:
            update = self._manage_position(pos)
            if update:
                updates.append(update)

        return updates

    def get_summary(self) -> dict:
        """Get a summary of all open positions for Telegram reporting."""
        positions = self.client.get_open_positions()
        acct = self.client.get_account_info()

        total_pnl = sum(p.profit for p in positions)
        winners = sum(1 for p in positions if p.profit > 0)
        losers = sum(1 for p in positions if p.profit < 0)

        return {
            "total_positions": len(positions),
            "total_pnl": total_pnl,
            "winners": winners,
            "losers": losers,
            "balance": acct.balance if acct else 0,
            "equity": acct.equity if acct else 0,
            "positions": [
                {
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "type": p.type,
                    "volume": p.volume,
                    "open_price": p.open_price,
                    "sl": p.sl,
                    "tp": p.tp,
                    "profit": p.profit,
                }
                for p in positions
            ],
        }

    # ── Internal ──────────────────────────────────────────

    def _manage_position(self, pos: PositionInfo) -> Optional[PositionUpdate]:
        """Manage a single position: trail, partial close, or reversal close."""
        ticker_yf = to_yfinance(pos.symbol)
        is_buy = pos.type == "BUY"

        # Get current price
        sym_info = self.client.get_symbol_info(pos.symbol)
        if sym_info is None:
            return None
        current_price = sym_info["bid"] if is_buy else sym_info["ask"]

        # Calculate R-multiple (how many R's of profit)
        risk = abs(pos.open_price - pos.sl) if pos.sl > 0 else 0
        if risk == 0:
            logger.warning(f"Position {pos.ticket} has no SL — skipping")
            return None

        if is_buy:
            profit_distance = current_price - pos.open_price
        else:
            profit_distance = pos.open_price - current_price

        r_multiple = profit_distance / risk

        logger.info(
            f"Position {pos.ticket} {pos.symbol} {pos.type}: "
            f"entry={pos.open_price}, current={current_price}, "
            f"SL={pos.sl}, TP={pos.tp}, R={r_multiple:.2f}, PnL={pos.profit:.2f}"
        )

        # ── Step 1: Check for signal reversal ────────────
        if self.enable_reversal_close and r_multiple < 0.5:
            # Only check reversal when trade is near breakeven or losing
            current_signal = _rerun_strategy(ticker_yf)
            if current_signal is not None:
                signal_is_buy = current_signal in (
                    TradeAction.BUY, TradeAction.STRONG_BUY
                )
                signal_is_sell = current_signal in (
                    TradeAction.SELL, TradeAction.STRONG_SELL
                )

                if (is_buy and signal_is_sell) or (not is_buy and signal_is_buy):
                    logger.info(
                        f"Signal REVERSED for {pos.symbol}: "
                        f"position={pos.type}, new_signal={current_signal.value}"
                    )
                    if not self.dry_run:
                        result = self.client.close_position(
                            pos.ticket, comment="reversal_close"
                        )
                        if result.success:
                            return PositionUpdate(
                                ticket=pos.ticket, symbol=pos.symbol,
                                action_taken="full_close",
                                pnl=pos.profit,
                                reason=f"Signal reversed to {current_signal.value}",
                            )
                    else:
                        return PositionUpdate(
                            ticket=pos.ticket, symbol=pos.symbol,
                            action_taken="full_close",
                            pnl=pos.profit,
                            reason=f"[DRY RUN] Signal reversed to {current_signal.value}",
                        )

        # ── Step 2: Partial close at 1R ──────────────────
        # Only if the position hasn't already been partially closed
        if (r_multiple >= self.partial_close_at_r
                and "partial" not in pos.comment
                and self.partial_close_pct > 0):
            close_vol = round(pos.volume * self.partial_close_pct, 2)
            min_vol = 0.01
            if close_vol >= min_vol:
                logger.info(
                    f"Partial close {pos.ticket}: {close_vol} lots at {r_multiple:.1f}R"
                )
                if not self.dry_run:
                    result = self.client.close_position(
                        pos.ticket, volume=close_vol, comment="partial_1R"
                    )
                    if result.success:
                        # Note: partial close doesn't return a PositionUpdate
                        # because the position is still open (with reduced volume)
                        pass

        # ── Step 3: Trail stop ───────────────────────────
        if r_multiple >= self.trail_activation_r:
            atr = _get_current_atr(ticker_yf)
            if atr <= 0:
                # Fallback: use a fraction of the risk as trail distance
                trail_distance = risk * 0.75
            else:
                trail_distance = atr * self.trail_atr_multiplier

            if is_buy:
                new_sl = current_price - trail_distance
                # Never move SL backwards
                if new_sl > pos.sl:
                    logger.info(
                        f"Trailing stop {pos.ticket}: "
                        f"SL {pos.sl:.5f} → {new_sl:.5f}"
                    )
                    if not self.dry_run:
                        result = self.client.modify_position(
                            pos.ticket, sl=new_sl
                        )
                        if result.success:
                            return PositionUpdate(
                                ticket=pos.ticket, symbol=pos.symbol,
                                action_taken="trail_stop",
                                old_sl=pos.sl, new_sl=new_sl,
                                pnl=pos.profit,
                                reason=f"Trailed at {r_multiple:.1f}R",
                            )
                    else:
                        return PositionUpdate(
                            ticket=pos.ticket, symbol=pos.symbol,
                            action_taken="trail_stop",
                            old_sl=pos.sl, new_sl=new_sl,
                            pnl=pos.profit,
                            reason=f"[DRY RUN] Trail at {r_multiple:.1f}R",
                        )
            else:
                new_sl = current_price + trail_distance
                if new_sl < pos.sl or pos.sl == 0:
                    logger.info(
                        f"Trailing stop {pos.ticket}: "
                        f"SL {pos.sl:.5f} → {new_sl:.5f}"
                    )
                    if not self.dry_run:
                        result = self.client.modify_position(
                            pos.ticket, sl=new_sl
                        )
                        if result.success:
                            return PositionUpdate(
                                ticket=pos.ticket, symbol=pos.symbol,
                                action_taken="trail_stop",
                                old_sl=pos.sl, new_sl=new_sl,
                                pnl=pos.profit,
                                reason=f"Trailed at {r_multiple:.1f}R",
                            )
                    else:
                        return PositionUpdate(
                            ticket=pos.ticket, symbol=pos.symbol,
                            action_taken="trail_stop",
                            old_sl=pos.sl, new_sl=new_sl,
                            pnl=pos.profit,
                            reason=f"[DRY RUN] Trail at {r_multiple:.1f}R",
                        )

        # No action needed
        return PositionUpdate(
            ticket=pos.ticket, symbol=pos.symbol,
            action_taken="no_change",
            pnl=pos.profit,
            reason=f"R={r_multiple:.2f}, holding",
        )
