"""
Alpaca Client — wrapper around the alpaca-py SDK.
===================================================
Handles connection, bracket order placement, position management,
and account queries for Alpaca Markets (stocks & leveraged ETFs).

Runs on any OS (Linux, Mac, Windows) — no native terminal needed.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopLimitOrderRequest,
        GetOrdersRequest,
        ReplaceOrderRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        OrderClass,
        OrderStatus,
        TimeInForce,
        QueryOrderStatus,
    )
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False


# ──────────────────────────────────────────────────────────
#  Data models (mirrors MT5 client interface for consistency)
# ──────────────────────────────────────────────────────────

@dataclass
class AccountInfo:
    account_id: str
    balance: float          # cash
    equity: float           # equity (cash + positions)
    buying_power: float
    currency: str
    pattern_day_trader: bool
    trading_blocked: bool
    account_blocked: bool


@dataclass
class PositionInfo:
    symbol: str
    side: str               # "long" or "short"
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float  # PnL percent


@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    message: str = ""
    filled_qty: float = 0.0
    filled_price: float = 0.0
    status: str = ""


# ──────────────────────────────────────────────────────────
#  Alpaca Client
# ──────────────────────────────────────────────────────────

CLIENT_TAG = "stock_trader_bot"


class AlpacaClient:
    """
    Wrapper around the alpaca-py TradingClient.

    Usage:
        client = AlpacaClient()
        client.connect()
        result = client.place_bracket_order("AAPL", "BUY", 10, sl=145.0, tp=165.0)
        client.disconnect()
    """

    def __init__(self):
        if not HAS_ALPACA:
            raise ImportError(
                "alpaca-py package not installed. "
                "Run: pip install alpaca-py"
            )
        self._client: Optional[TradingClient] = None
        self._connected = False

    # ── Connection ────────────────────────────────────────

    def connect(
        self,
        api_key: str = None,
        secret_key: str = None,
        paper: bool = True,
    ) -> bool:
        """
        Connect to Alpaca Markets API.

        Parameters can be passed directly or read from environment variables:
            ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
        """
        api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        paper_env = os.environ.get("ALPACA_PAPER", "true").lower()
        paper = paper if api_key else (paper_env in ("true", "1", "yes"))

        if not api_key or not secret_key:
            logger.error(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY."
            )
            return False

        try:
            self._client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper,
            )
            # Verify connection by fetching account
            acct = self._client.get_account()
            self._connected = True

            mode = "PAPER" if paper else "LIVE"
            logger.info(
                f"Connected to Alpaca [{mode}] | "
                f"Account: {acct.id} | "
                f"Equity: ${float(acct.equity):,.2f} | "
                f"Cash: ${float(acct.cash):,.2f} | "
                f"Buying Power: ${float(acct.buying_power):,.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            return False

    def disconnect(self):
        """Clean up (Alpaca REST client doesn't need explicit disconnect)."""
        self._connected = False
        self._client = None
        logger.info("Alpaca client disconnected.")

    def is_connected(self) -> bool:
        return self._connected

    # ── Account Info ──────────────────────────────────────

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get current account balance, equity, buying power."""
        self._ensure_connected()
        try:
            acct = self._client.get_account()
            return AccountInfo(
                account_id=str(acct.id),
                balance=float(acct.cash),
                equity=float(acct.equity),
                buying_power=float(acct.buying_power),
                currency=acct.currency or "USD",
                pattern_day_trader=bool(acct.pattern_day_trader),
                trading_blocked=bool(acct.trading_blocked),
                account_blocked=bool(acct.account_blocked),
            )
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None

    # ── Check Market Hours ────────────────────────────────

    def is_market_open(self) -> bool:
        """Check if the US stock market is currently open."""
        self._ensure_connected()
        try:
            clock = self._client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market clock: {e}")
            return False

    def get_clock(self) -> Optional[dict]:
        """Get market clock info."""
        self._ensure_connected()
        try:
            clock = self._client.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": str(clock.next_open),
                "next_close": str(clock.next_close),
                "timestamp": str(clock.timestamp),
            }
        except Exception as e:
            logger.error(f"Failed to get clock: {e}")
            return None

    # ── Asset Info ────────────────────────────────────────

    def get_asset(self, symbol: str) -> Optional[dict]:
        """Check if an asset is tradeable."""
        self._ensure_connected()
        try:
            asset = self._client.get_asset(symbol)
            return {
                "symbol": asset.symbol,
                "name": asset.name,
                "tradable": asset.tradable,
                "shortable": asset.shortable,
                "fractionable": asset.fractionable,
                "status": str(asset.status),
                "exchange": str(asset.exchange),
            }
        except Exception as e:
            logger.warning(f"Asset {symbol} not found: {e}")
            return None

    # ── Place Bracket Order ───────────────────────────────

    def place_bracket_order(
        self,
        symbol: str,
        action: str,
        qty: int,
        sl: float,
        tp: float,
        client_order_id: str = None,
    ) -> OrderResult:
        """
        Place a bracket order (market entry + stop loss + take profit).

        Parameters
        ----------
        symbol   : Stock symbol (e.g., "AAPL", "MSTU")
        action   : "BUY" or "SELL"
        qty      : Number of shares
        sl       : Stop loss price
        tp       : Take profit price
        """
        self._ensure_connected()

        try:
            is_buy = action.upper() in ("BUY", "STRONG BUY")
            side = OrderSide.BUY if is_buy else OrderSide.SELL

            from alpaca.trading.requests import (
                TakeProfitRequest,
                StopLossRequest,
            )

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(tp, 2)),
                stop_loss=StopLossRequest(stop_price=round(sl, 2)),
                client_order_id=client_order_id,
            )

            order = self._client.submit_order(order_data)

            logger.info(
                f"Bracket order placed: {action} {qty} {symbol} | "
                f"SL=${sl:.2f} TP=${tp:.2f} | "
                f"Order ID={order.id} | Status={order.status}"
            )

            return OrderResult(
                success=True,
                order_id=str(order.id),
                message=f"Order submitted: {order.status}",
                filled_qty=float(order.filled_qty or 0),
                filled_price=float(order.filled_avg_price or 0),
                status=str(order.status),
            )

        except Exception as e:
            msg = f"Bracket order failed for {symbol}: {e}"
            logger.error(msg)
            return OrderResult(success=False, message=msg)

    # ── Place Simple Market Order ─────────────────────────

    def place_market_order(
        self,
        symbol: str,
        action: str,
        qty: int,
        client_order_id: str = None,
    ) -> OrderResult:
        """Place a simple market order (no SL/TP)."""
        self._ensure_connected()

        try:
            is_buy = action.upper() in ("BUY", "STRONG BUY")
            side = OrderSide.BUY if is_buy else OrderSide.SELL

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )

            order = self._client.submit_order(order_data)

            logger.info(
                f"Market order placed: {action} {qty} {symbol} | "
                f"Order ID={order.id} | Status={order.status}"
            )

            return OrderResult(
                success=True,
                order_id=str(order.id),
                message=f"Order submitted: {order.status}",
                filled_qty=float(order.filled_qty or 0),
                filled_price=float(order.filled_avg_price or 0),
                status=str(order.status),
            )

        except Exception as e:
            msg = f"Market order failed for {symbol}: {e}"
            logger.error(msg)
            return OrderResult(success=False, message=msg)

    # ── Query Positions ───────────────────────────────────

    def get_open_positions(self) -> list[PositionInfo]:
        """Get all open positions."""
        self._ensure_connected()
        try:
            positions = self._client.get_all_positions()
            result = []
            for p in positions:
                result.append(PositionInfo(
                    symbol=p.symbol,
                    side=str(p.side),
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc),
                ))
            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get a specific position by symbol."""
        self._ensure_connected()
        try:
            p = self._client.get_open_position(symbol)
            return PositionInfo(
                symbol=p.symbol,
                side=str(p.side),
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
            )
        except Exception:
            return None

    # ── Close Position ────────────────────────────────────

    def close_position(
        self, symbol: str, qty: int = None
    ) -> OrderResult:
        """
        Close an open position (full or partial).

        Parameters
        ----------
        symbol : Stock symbol
        qty    : Shares to close (None = close all)
        """
        self._ensure_connected()
        try:
            if qty:
                # Partial close via opposite market order
                pos = self.get_position(symbol)
                if pos is None:
                    return OrderResult(
                        success=False, message=f"No position on {symbol}"
                    )
                action = "SELL" if pos.side == "long" else "BUY"
                return self.place_market_order(symbol, action, qty)
            else:
                # Full close
                order = self._client.close_position(symbol)
                logger.info(f"Position {symbol} closed: {order}")
                return OrderResult(
                    success=True,
                    order_id=str(order.id) if hasattr(order, "id") else "",
                    message="Position closed",
                    status=str(order.status) if hasattr(order, "status") else "closed",
                )
        except Exception as e:
            msg = f"Close position failed for {symbol}: {e}"
            logger.error(msg)
            return OrderResult(success=False, message=msg)

    # ── Query Orders ──────────────────────────────────────

    def get_open_orders(self, symbol: str = None) -> list[dict]:
        """Get all open (pending) orders."""
        self._ensure_connected()
        try:
            request_params = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[symbol] if symbol else None,
                nested=True,
            )
            orders = self._client.get_orders(request_params)
            result = []
            for o in orders:
                result.append({
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "side": str(o.side),
                    "type": str(o.type),
                    "qty": float(o.qty) if o.qty else 0,
                    "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
                    "status": str(o.status),
                    "order_class": str(o.order_class) if o.order_class else "",
                    "limit_price": float(o.limit_price) if o.limit_price else 0,
                    "stop_price": float(o.stop_price) if o.stop_price else 0,
                    "created_at": str(o.created_at),
                })
            return result
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        self._ensure_connected()
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Cancel order {order_id} failed: {e}")
            return False

    # ── Internals ─────────────────────────────────────────

    def _ensure_connected(self):
        if not self._connected or self._client is None:
            raise ConnectionError(
                "Not connected to Alpaca. Call connect() first."
            )
