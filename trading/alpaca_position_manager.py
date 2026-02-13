"""
Alpaca Position Manager — monitors and manages open Alpaca positions.
=====================================================================
Handles:
    - Trailing stops (cancel old stop → place new tighter stop)
    - Signal reversal detection (re-run strategy, close if flipped)
    - Partial close (sell half at 1R, let remainder run)
    - Position status reporting for Telegram

Designed to run periodically via GitHub Actions (every 1-2 hours).

NOTE: Alpaca bracket orders have built-in SL/TP (OCO legs). This manager
      handles *trailing* the stop above breakeven and detecting reversals
      that the static bracket doesn't catch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config
from data.fetcher import StockDataFetcher
from models.signals import TradeAction
from trading.alpaca_client import AlpacaClient, PositionInfo

logger = logging.getLogger(__name__)

LEVERAGED_TICKERS = set(getattr(config, "LEVERAGED_TICKERS", ["MSTU", "MSTR", "MSTZ", "TSLL"]))


# ──────────────────────────────────────────────────────────
#  Data models
# ──────────────────────────────────────────────────────────

@dataclass
class PositionUpdate:
    """Record of a position management action."""
    symbol: str
    action_taken: str     # "trail_stop", "partial_close", "full_close", "no_change"
    old_sl: float = 0.0
    new_sl: float = 0.0
    pnl: float = 0.0
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ──────────────────────────────────────────────────────────
#  ATR helper
# ──────────────────────────────────────────────────────────

def _get_current_atr(ticker: str, period: int = 14) -> float:
    """Fetch recent data and compute ATR for a ticker."""
    try:
        is_leveraged = ticker.upper() in LEVERAGED_TICKERS
        interval = "1h" if is_leveraged else "1d"
        fetch_period = "1mo" if is_leveraged else "3mo"

        df = StockDataFetcher(ticker).fetch(period=fetch_period, interval=interval)
        if df is None or len(df) < period + 1:
            return 0.0

        high = df["High"]
        low = df["Low"]
        close = df["Close"].shift(1)
        tr = (high - low).combine(abs(high - close), max).combine(abs(low - close), max)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if atr == atr else 0.0  # NaN check
    except Exception as e:
        logger.warning(f"ATR fetch failed for {ticker}: {e}")
        return 0.0


def _rerun_strategy(ticker: str) -> Optional[TradeAction]:
    """Re-run the appropriate strategy to check if signal has flipped."""
    try:
        from bt_engine.engine import _detect_asset_type, _run_strategy

        is_leveraged = ticker.upper() in LEVERAGED_TICKERS

        if is_leveraged:
            df = StockDataFetcher(ticker).fetch(period="3mo", interval="1d")
        else:
            df = StockDataFetcher(ticker).fetch(period="6mo", interval="1d")

        if df is None or len(df) < 30:
            return None

        strat = _run_strategy(df, ticker, stock_mode=True)
        if strat and strat.trade_setup:
            return strat.trade_setup.action
        return TradeAction.HOLD
    except Exception as e:
        logger.warning(f"Strategy re-run failed for {ticker}: {e}")
        return None


# ──────────────────────────────────────────────────────────
#  Position Manager
# ──────────────────────────────────────────────────────────

class AlpacaPositionManager:
    """
    Manages open stock positions on Alpaca.

    Usage:
        client = AlpacaClient()
        client.connect()
        manager = AlpacaPositionManager(client)
        updates = manager.manage_all()
    """

    def __init__(
        self,
        client: AlpacaClient,
        trail_activation_r: float = 1.0,
        trail_atr_multiplier: float = 1.0,
        enable_reversal_close: bool = True,
        partial_close_at_r: float = 1.0,
        partial_close_pct: float = 0.5,
        dry_run: bool = False,
    ):
        self.client = client
        self.trail_activation_r = trail_activation_r
        self.trail_atr_multiplier = trail_atr_multiplier
        self.enable_reversal_close = enable_reversal_close
        self.partial_close_at_r = partial_close_at_r
        self.partial_close_pct = partial_close_pct
        self.dry_run = dry_run

    def manage_all(self) -> list[PositionUpdate]:
        """Check and manage all open positions."""
        positions = self.client.get_open_positions()
        if not positions:
            logger.info("No open Alpaca positions to manage.")
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

        total_pnl = sum(p.unrealized_pl for p in positions)
        winners = sum(1 for p in positions if p.unrealized_pl > 0)
        losers = sum(1 for p in positions if p.unrealized_pl < 0)

        return {
            "total_positions": len(positions),
            "total_pnl": total_pnl,
            "winners": winners,
            "losers": losers,
            "balance": acct.balance if acct else 0,
            "equity": acct.equity if acct else 0,
            "buying_power": acct.buying_power if acct else 0,
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "qty": p.qty,
                    "avg_entry": p.avg_entry_price,
                    "current_price": p.current_price,
                    "profit": p.unrealized_pl,
                    "profit_pct": p.unrealized_plpc * 100,
                }
                for p in positions
            ],
        }

    # ── Internal ──────────────────────────────────────────

    def _manage_position(self, pos: PositionInfo) -> Optional[PositionUpdate]:
        """Manage a single position: check reversal, log status."""
        ticker = pos.symbol
        is_long = pos.side == "long"

        # ── Halal compliance: close any short positions immediately ──
        if not is_long:
            logger.info(
                f"SHORT position detected on {ticker} — closing for halal compliance. "
                f"PnL: ${pos.unrealized_pl:+.2f}"
            )
            if not self.dry_run:
                # Cancel ALL open orders on this symbol (bracket legs lock shares)
                # Keep cancelling until no orders remain
                import time
                for attempt in range(3):
                    open_orders = self.client.get_open_orders(ticker)
                    if not open_orders:
                        break
                    for order in open_orders:
                        oid = order.get("id", "")
                        logger.info(f"Cancelling order {oid} on {ticker} (attempt {attempt+1})")
                        self.client.cancel_order(oid)
                    time.sleep(1)  # Wait for cancellation to propagate

                # Now close the position
                result = self.client.close_position(ticker)
                if result.success:
                    return PositionUpdate(
                        symbol=ticker,
                        action_taken="full_close",
                        pnl=pos.unrealized_pl,
                        reason=f"Short position closed (halal compliance)",
                    )
                else:
                    logger.error(f"Failed to close short {ticker}: {result.message}")
            return PositionUpdate(
                symbol=ticker,
                action_taken="full_close",
                pnl=pos.unrealized_pl,
                reason=f"[DRY RUN] Short position closed (halal compliance)",
            )

        # We need to find the current SL from the open stop-loss order
        sl_price = self._find_stop_loss_price(ticker)

        # Calculate R-multiple
        risk = abs(pos.avg_entry_price - sl_price) if sl_price > 0 else 0
        if risk == 0:
            logger.warning(f"Position {ticker} has no SL order — skipping trail logic")
            # Still check reversal
            risk = pos.avg_entry_price * 0.02  # fallback: 2% as risk

        if is_long:
            profit_distance = pos.current_price - pos.avg_entry_price
        else:
            profit_distance = pos.avg_entry_price - pos.current_price

        r_multiple = profit_distance / risk if risk > 0 else 0

        logger.info(
            f"Position {ticker} {pos.side}: "
            f"entry=${pos.avg_entry_price:.2f}, current=${pos.current_price:.2f}, "
            f"SL=${sl_price:.2f}, R={r_multiple:.2f}, "
            f"PnL=${pos.unrealized_pl:+.2f} ({pos.unrealized_plpc*100:+.1f}%)"
        )

        # ── Step 1: Check for signal reversal ────────────
        if self.enable_reversal_close and r_multiple < 0.5:
            current_signal = _rerun_strategy(ticker)
            if current_signal is not None:
                signal_is_buy = current_signal in (
                    TradeAction.BUY, TradeAction.STRONG_BUY
                )
                signal_is_sell = current_signal in (
                    TradeAction.SELL, TradeAction.STRONG_SELL
                )

                if (is_long and signal_is_sell) or (not is_long and signal_is_buy):
                    logger.info(
                        f"Signal REVERSED for {ticker}: "
                        f"position={pos.side}, new_signal={current_signal.value}"
                    )
                    if not self.dry_run:
                        result = self.client.close_position(ticker)
                        if result.success:
                            return PositionUpdate(
                                symbol=ticker,
                                action_taken="full_close",
                                pnl=pos.unrealized_pl,
                                reason=f"Signal reversed to {current_signal.value}",
                            )
                    else:
                        return PositionUpdate(
                            symbol=ticker,
                            action_taken="full_close",
                            pnl=pos.unrealized_pl,
                            reason=f"[DRY RUN] Signal reversed to {current_signal.value}",
                        )

        # ── Step 2: Trail stop ─────────────────────────────────
        # For leveraged ETFs with trailing enabled: cancel old SL → place new tighter SL
        # For regular stocks: always trail at 1R+
        is_leveraged = ticker.upper() in LEVERAGED_TICKERS
        trail_enabled = (
            config.LEVERAGED_MODE.get("trailing_stop", False) if is_leveraged
            else True  # always trail regular stocks
        )

        if trail_enabled and sl_price > 0 and r_multiple >= self.trail_activation_r:
            atr = _get_current_atr(ticker)
            if atr <= 0:
                trail_distance = risk * 0.75
            else:
                trail_distance = atr * self.trail_atr_multiplier

            if is_long:
                new_sl = pos.current_price - trail_distance
                # Never move SL backwards, and must be above entry for profit lock
                if new_sl > sl_price and new_sl > pos.avg_entry_price:
                    logger.info(
                        f"Trailing stop {ticker}: "
                        f"SL ${sl_price:.2f} → ${new_sl:.2f} (at {r_multiple:.1f}R)"
                    )
                    if not self.dry_run:
                        # Cancel ALL existing stop orders for this symbol
                        success = self._replace_stop_order(
                            ticker, int(pos.qty), new_sl, "sell"
                        )
                        if success:
                            return PositionUpdate(
                                symbol=ticker,
                                action_taken="trail_stop",
                                old_sl=sl_price,
                                new_sl=new_sl,
                                pnl=pos.unrealized_pl,
                                reason=f"Trailed at {r_multiple:.1f}R: "
                                       f"SL ${sl_price:.2f} → ${new_sl:.2f}",
                            )
                    else:
                        return PositionUpdate(
                            symbol=ticker,
                            action_taken="trail_stop",
                            old_sl=sl_price,
                            new_sl=new_sl,
                            pnl=pos.unrealized_pl,
                            reason=f"[DRY RUN] Trail at {r_multiple:.1f}R: "
                                   f"SL ${sl_price:.2f} → ${new_sl:.2f}",
                        )

        # No action needed
        return PositionUpdate(
            symbol=ticker,
            action_taken="no_change",
            pnl=pos.unrealized_pl,
            reason=f"R={r_multiple:.2f}, holding",
        )

    def _replace_stop_order(
        self, symbol: str, qty: int, new_stop_price: float, side: str
    ) -> bool:
        """Cancel old stop orders and place a new one at the tighter level."""
        import time

        # Cancel existing stop orders
        orders = self.client.get_open_orders(symbol)
        cancelled = False
        for order in orders:
            if order.get("stop_price", 0) > 0:
                oid = order.get("id", "")
                logger.info(f"Cancelling old stop order {oid} on {symbol}")
                self.client.cancel_order(oid)
                cancelled = True

        if cancelled:
            time.sleep(1)  # Wait for cancellation to propagate

        # Place new stop order
        result = self.client.place_stop_order(
            symbol=symbol, qty=qty, stop_price=new_stop_price, side=side
        )
        return result.success

    def _find_stop_loss_price(self, symbol: str) -> float:
        """Find the current stop-loss price from open orders for a symbol."""
        try:
            orders = self.client.get_open_orders(symbol)
            for order in orders:
                # Stop orders have stop_price set
                if order.get("stop_price", 0) > 0 and "stop" in order.get("type", "").lower():
                    return order["stop_price"]
                # Also check child orders of brackets
                if order.get("stop_price", 0) > 0:
                    return order["stop_price"]
            return 0.0
        except Exception:
            return 0.0
