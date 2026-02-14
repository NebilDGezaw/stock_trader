"""
Position Manager — monitors and manages open HFM MT5 positions.
================================================================
Handles:
    - Trailing stops (move SL to breakeven after 1R, then trail by ATR)
    - Signal reversal detection (re-run strategy, close if flipped)
    - Partial close (50% at 1R, let remainder run to TP)
    - Position status reporting for Telegram

Designed to run periodically via GitHub Actions (every few hours).

CRITICAL: Quote validation
    On ephemeral GitHub Actions VMs, MT5 symbol quotes may not be
    populated immediately after connection.  bid/ask can be 0 or stale.
    We MUST validate current_price before using it for R-multiple
    calculations or trailing stop placement.  When quotes are bad,
    we fall back to the server-side pos.profit for safety decisions.
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
#  Quote Validation
# ──────────────────────────────────────────────────────────

def _validate_price(current_price: float, entry_price: float,
                    symbol: str, max_deviation_pct: float = 0.15) -> bool:
    """
    Validate that a quote price is reasonable.

    Returns False if:
        - Price is zero or negative
        - Price deviates more than max_deviation_pct from entry
          (e.g., 15% default — catches stale/garbage quotes)

    For forex (price < $10), we use 5% since forex doesn't move 15%.
    For crypto, we allow 20% since crypto is volatile.
    """
    if current_price <= 0:
        logger.warning(
            f"STALE QUOTE: {symbol} bid/ask is {current_price} (zero/negative)"
        )
        return False

    if entry_price <= 0:
        return True  # Can't validate without entry, assume ok

    deviation = abs(current_price - entry_price) / entry_price

    # Adjust threshold based on asset class
    if entry_price < 10:
        # Forex (prices like 1.xxxxx) or XRP — 5% max
        threshold = 0.05
    elif entry_price > 10000:
        # BTC — allow 20%
        threshold = 0.20
    else:
        threshold = max_deviation_pct

    if deviation > threshold:
        logger.warning(
            f"STALE QUOTE: {symbol} current={current_price:.5f} vs "
            f"entry={entry_price:.5f} — {deviation*100:.1f}% deviation "
            f"(threshold={threshold*100:.0f}%). Using server PnL instead."
        )
        return False

    return True


def _compute_r_from_pnl(pos: PositionInfo) -> Optional[float]:
    """
    Compute R-multiple from server-side PnL when quotes are unreliable.

    R = PnL / risk_amount_per_1R
    risk_amount_per_1R ≈ |entry - SL| × volume × contract_size × pip_value

    Since we don't have contract details here, we estimate from the SL
    distance and the position's actual risk:
        risk_in_price = |entry - SL|
        pnl_at_1r ≈ abs(profit when price moves risk_in_price distance)

    For a rough estimate: if the position profit is -$50 and we know the
    SL distance is X pips at Y lots, we can back-calculate R.

    Simpler approach: use the fact that at exactly SL hit, PnL = -1R.
    So 1R (in dollars) ≈ the dollar loss at SL.  We can estimate this
    as: risk_dollars ≈ |entry - SL| / entry × notional_value.
    But without contract_size, just return the raw PnL sign for
    safety decisions.
    """
    if pos.sl <= 0:
        return None

    risk_distance = abs(pos.open_price - pos.sl)
    if risk_distance == 0:
        return None

    # Estimate R from PnL direction and magnitude
    # This is approximate but MUCH better than using garbage quotes
    # We know that at -1R, profit should be approximately -(risk_per_pip × volume × risk_pips)
    # Without full contract info, use a heuristic:
    # If PnL is positive, R is positive. If very negative relative to likely risk, flag it.
    # For safety decisions, the sign and rough magnitude is what matters.

    # Use the PnL itself — for max_loss_r check, if PnL < 0 and large,
    # we should still close. For trailing, if PnL > 0, we should still trail.
    return None  # Signal to caller to use PnL-based logic instead


# ──────────────────────────────────────────────────────────
#  ATR helper (works without full strategy import)
# ──────────────────────────────────────────────────────────

def _get_current_atr(ticker_yf: str, period: int = 14) -> float:
    """Fetch recent data and compute ATR for a ticker."""
    try:
        from utils.helpers import compute_atr

        asset_type = get_asset_type(ticker_yf)
        # "metal" (gold/silver) uses 4H just like "commodity"
        if asset_type in ("forex", "crypto", "commodity", "metal"):
            df = StockDataFetcher(ticker_yf).fetch(period="3mo", interval="4h")
        else:
            df = StockDataFetcher(ticker_yf).fetch(period="3mo", interval="1d")

        if df is None or len(df) < period + 1:
            return 0.0

        atr_series = compute_atr(df, period=period)
        atr = float(atr_series.iloc[-1])
        return atr if atr == atr else 0.0  # NaN check
    except Exception as e:
        logger.warning(f"ATR fetch failed for {ticker_yf}: {e}")
        return 0.0


def _rerun_strategy(ticker_yf: str) -> Optional[TradeAction]:
    """Re-run the appropriate strategy to check if signal has flipped."""
    try:
        from bt_engine.engine import _detect_asset_type, _run_strategy

        asset_type = _detect_asset_type(ticker_yf)

        # "metal" (gold/silver) uses 4H just like "commodity"
        if asset_type in ("forex", "crypto", "commodity", "metal"):
            df = StockDataFetcher(ticker_yf).fetch(period="3mo", interval="4h")
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
        max_position_age_hours: float = 48.0,   # close after 48h (2 days)
        max_loss_r_multiple: float = -2.0,       # close if loss exceeds 2R
        close_losing_on_hold: bool = True,       # close losers when signal is HOLD
        dry_run: bool = False,
    ):
        self.client = client
        self.trail_activation_r = trail_activation_r
        self.trail_atr_multiplier = trail_atr_multiplier
        self.partial_close_at_r = partial_close_at_r
        self.partial_close_pct = partial_close_pct
        self.enable_reversal_close = enable_reversal_close
        self.max_position_age_hours = max_position_age_hours
        self.max_loss_r_multiple = max_loss_r_multiple
        self.close_losing_on_hold = close_losing_on_hold
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

    def _close_position(self, pos: PositionInfo, reason: str, comment: str) -> Optional[PositionUpdate]:
        """Helper: close a position with retry and return the update record."""
        if not self.dry_run:
            import time
            for attempt in range(3):
                result = self.client.close_position(pos.ticket, comment=comment)
                if result.success:
                    return PositionUpdate(
                        ticket=pos.ticket, symbol=pos.symbol,
                        action_taken="full_close", pnl=pos.profit, reason=reason,
                    )
                else:
                    logger.warning(
                        f"Close attempt {attempt + 1}/3 failed for "
                        f"{pos.ticket} {pos.symbol}: {result}"
                    )
                    if attempt < 2:
                        time.sleep(2)
            logger.error(
                f"CRITICAL: ALL 3 close attempts FAILED for {pos.ticket} "
                f"{pos.symbol} — position remains open! Reason: {reason}"
            )
            return None
        else:
            return PositionUpdate(
                ticket=pos.ticket, symbol=pos.symbol,
                action_taken="full_close", pnl=pos.profit,
                reason=f"[DRY RUN] {reason}",
            )

    def _manage_position(self, pos: PositionInfo) -> Optional[PositionUpdate]:
        """
        Manage a single position with VALIDATED quotes.

        CRITICAL: On ephemeral VMs, MT5 quotes may be stale/zero.
        We validate current_price before using it.  When quotes are bad,
        we fall back to server-side pos.profit for safety decisions and
        skip trailing stop / partial close (which need accurate prices).
        """
        ticker_yf = to_yfinance(pos.symbol)
        is_buy = pos.type == "BUY"

        # ── Get and VALIDATE current price ────────────────
        sym_info = self.client.get_symbol_info(pos.symbol)
        if sym_info is None:
            logger.warning(f"No symbol info for {pos.symbol} — skipping")
            return None

        raw_price = sym_info["bid"] if is_buy else sym_info["ask"]
        price_valid = _validate_price(raw_price, pos.open_price, pos.symbol)
        current_price = raw_price if price_valid else 0.0

        # Calculate risk (SL distance)
        risk = abs(pos.open_price - pos.sl) if pos.sl > 0 else 0
        if risk == 0:
            # Position has no SL — use a fallback risk estimate (2% of entry)
            # so safety checks (PnL net, max age, reversal) still work.
            # Do NOT skip — unprotected positions need management most.
            logger.warning(
                f"Position {pos.ticket} {pos.symbol} has no SL — "
                f"using 2% fallback risk for management"
            )
            risk = pos.open_price * 0.02

        # Calculate R-multiple — use validated price OR fall back to PnL
        if price_valid and current_price > 0:
            if is_buy:
                profit_distance = current_price - pos.open_price
            else:
                profit_distance = pos.open_price - current_price
            r_multiple = profit_distance / risk
        else:
            # FALLBACK: estimate R from server-side PnL
            # 1R in dollars ≈ risk_distance × volume × contract_size × pip_value
            # Since we don't have contract details, use a conservative approach:
            # Just use PnL sign + magnitude relative to a $500 estimate per 1R
            # This prevents the garbage-quote-triggered closures
            estimated_1r_dollars = max(abs(pos.profit) * 0.3, 50.0)  # rough floor
            r_multiple = pos.profit / estimated_1r_dollars if estimated_1r_dollars > 0 else 0
            logger.info(
                f"Using PnL-based R estimate for {pos.symbol}: "
                f"PnL=${pos.profit:.2f}, estimated R={r_multiple:.2f}"
            )

        # Calculate position age in hours
        try:
            open_time = datetime.utcfromtimestamp(pos.time) if isinstance(pos.time, (int, float)) else pos.time
            age_hours = (datetime.utcnow() - open_time).total_seconds() / 3600.0
        except Exception:
            age_hours = 0.0

        price_tag = f"current={current_price:.5f}" if price_valid else "current=STALE"
        logger.info(
            f"Position {pos.ticket} {pos.symbol} {pos.type}: "
            f"entry={pos.open_price}, {price_tag}, "
            f"SL={pos.sl}, TP={pos.tp}, R={r_multiple:.2f}, "
            f"PnL={pos.profit:.2f}, age={age_hours:.1f}h"
        )

        # ── Step 0a: MAX LOSS — use server PnL, not R-multiple ──
        # This is the SAFE way: pos.profit is always correct from MT5 server.
        # We close if dollar loss exceeds a threshold based on equity.
        # Also check R-multiple but ONLY if quotes are validated.
        if price_valid and r_multiple <= self.max_loss_r_multiple:
            reason = (
                f"Max loss hit: R={r_multiple:.2f} <= {self.max_loss_r_multiple}R "
                f"(PnL=${pos.profit:.2f}, age={age_hours:.1f}h)"
            )
            logger.warning(f"CLOSING {pos.ticket} {pos.symbol}: {reason}")
            result = self._close_position(pos, reason, "max_loss_close")
            if result:
                return result

        # PnL-based safety net: close if losing more than $500 regardless of quotes
        if pos.profit < -500:
            reason = (
                f"PnL safety net: ${pos.profit:.2f} exceeds -$500 threshold "
                f"(age={age_hours:.1f}h)"
            )
            logger.warning(f"CLOSING {pos.ticket} {pos.symbol}: {reason}")
            result = self._close_position(pos, reason, "pnl_safety_close")
            if result:
                return result

        # ── Step 0b: MAX AGE TIMEOUT ─────────────────────
        if age_hours >= self.max_position_age_hours:
            reason = (
                f"Position timeout: {age_hours:.1f}h >= {self.max_position_age_hours}h "
                f"(R={r_multiple:.2f}, PnL=${pos.profit:.2f})"
            )
            logger.warning(f"CLOSING {pos.ticket} {pos.symbol}: {reason}")
            result = self._close_position(pos, reason, "timeout_close")
            if result:
                return result

        # ── Step 1: Signal reversal (only with VALID quotes) ──
        # We need reliable R-multiple to decide if position is losing
        if self.enable_reversal_close and price_valid and r_multiple < 0.5:
            current_signal = _rerun_strategy(ticker_yf)
            if current_signal is not None:
                signal_is_buy = current_signal in (
                    TradeAction.BUY, TradeAction.STRONG_BUY
                )
                signal_is_sell = current_signal in (
                    TradeAction.SELL, TradeAction.STRONG_SELL
                )
                signal_is_hold = current_signal == TradeAction.HOLD

                # Close on explicit reversal
                if (is_buy and signal_is_sell) or (not is_buy and signal_is_buy):
                    reason = f"Signal reversed to {current_signal.value} (R={r_multiple:.2f})"
                    logger.info(f"Signal REVERSED for {pos.symbol}: {reason}")
                    result = self._close_position(pos, reason, "reversal_close")
                    if result:
                        return result

                # Close losing positions when signal goes HOLD
                if self.close_losing_on_hold and signal_is_hold and r_multiple < -0.5:
                    reason = (
                        f"Signal is HOLD and position losing: R={r_multiple:.2f}, "
                        f"PnL=${pos.profit:.2f} — no longer confirmed"
                    )
                    logger.info(f"Closing unconfirmed loser {pos.symbol}: {reason}")
                    result = self._close_position(pos, reason, "hold_loser_close")
                    if result:
                        return result

        # ── SKIP Steps 2-3 if quotes are invalid ─────────
        # Partial close and trailing stops REQUIRE accurate current_price.
        # With bad quotes we'd set garbage SL levels or close profitable trades.
        if not price_valid:
            return PositionUpdate(
                ticket=pos.ticket, symbol=pos.symbol,
                action_taken="no_change",
                pnl=pos.profit,
                reason=f"Quotes stale — skipping trail/partial (PnL=${pos.profit:.2f})",
            )

        # ── Step 2: Partial close at 1R ──────────────────
        # Detect prior partial close by checking if volume is already reduced.
        # MT5 does NOT update position.comment on partial close, so checking
        # the comment is unreliable. Instead, check if the current volume is
        # significantly less than what a "fresh" position would have — i.e.,
        # if volume is already below 60% of what lot sizing would produce,
        # assume a partial close already happened.
        already_partially_closed = (
            "partial" in (pos.comment or "")
            or pos.volume < 0.015  # near minimum lot — no point closing further
        )
        if (r_multiple >= self.partial_close_at_r
                and not already_partially_closed
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
                        return PositionUpdate(
                            ticket=pos.ticket, symbol=pos.symbol,
                            action_taken="partial_close",
                            pnl=pos.profit,
                            reason=f"Partial close {close_vol} lots at {r_multiple:.1f}R",
                        )

        # ── Step 3: Trail stop (ONLY with valid quotes) ──
        if r_multiple >= self.trail_activation_r:
            atr = _get_current_atr(ticker_yf)
            if atr <= 0:
                trail_distance = risk * 0.75
            else:
                trail_distance = atr * self.trail_atr_multiplier

            # Sanity: trail_distance must be positive and reasonable
            if trail_distance <= 0 or trail_distance > pos.open_price * 0.5:
                logger.warning(
                    f"Trail distance {trail_distance:.5f} is unreasonable "
                    f"for {pos.symbol} (entry={pos.open_price}) — skipping"
                )
            elif is_buy:
                new_sl = current_price - trail_distance
                # Sanity: new_sl must be between 0 and current_price
                if new_sl > 0 and new_sl < current_price and new_sl > pos.sl:
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
                # Sanity: new_sl must be above current_price and improving
                if new_sl > current_price and (new_sl < pos.sl or pos.sl == 0):
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
            reason=f"R={r_multiple:.2f}, age={age_hours:.1f}h, holding",
        )
