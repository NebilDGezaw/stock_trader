"""
MT5 Client — thin wrapper around the MetaTrader5 Python library.
================================================================
Handles connection, order management, and position queries for HFM.

Supports two modes:
    1. Native: MetaTrader5 package on Windows (pip install MetaTrader5)
    2. RPyC bridge: mt5linux on Linux/Docker (pip install mt5linux)
       Connects to a Wine-based MT5 terminal via RPyC on localhost:18812
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # Will fail gracefully with clear error

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Data models
# ──────────────────────────────────────────────────────────

@dataclass
class AccountInfo:
    login: int
    balance: float
    equity: float
    margin: float
    free_margin: float
    currency: str
    leverage: int
    server: str


@dataclass
class PositionInfo:
    ticket: int
    symbol: str
    type: str           # "BUY" or "SELL"
    volume: float
    open_price: float
    sl: float
    tp: float
    profit: float
    comment: str
    magic: int
    time_str: str


@dataclass
class OrderResult:
    success: bool
    ticket: int = 0
    retcode: int = 0
    message: str = ""
    volume: float = 0.0
    price: float = 0.0


# ──────────────────────────────────────────────────────────
#  MT5 Client
# ──────────────────────────────────────────────────────────

MAGIC_NUMBER = 202602  # Identifies orders placed by this bot


class MT5Client:
    """
    Wrapper around the MetaTrader5 Python package.

    Usage:
        client = MT5Client()
        client.connect(login=123, password="pw", server="HFMarketsIntl-Live")
        result = client.place_order("EURUSD", "BUY", 0.01, sl=1.0800, tp=1.0950)
        client.disconnect()
    """

    def __init__(self):
        if mt5 is None:
            raise ImportError(
                "MetaTrader5 package not installed. "
                "Run: pip install MetaTrader5"
            )
        self._connected = False

    # ── Connection ────────────────────────────────────────

    def connect(
        self,
        login: int = None,
        password: str = None,
        server: str = None,
        path: str = None,
        retries: int = 3,
        retry_delay: int = 20,
    ) -> bool:
        """
        Connect to an already-running MT5 terminal, or start one.

        Parameters can be passed directly or read from environment variables:
            HFM_MT5_LOGIN, HFM_MT5_PASSWORD, HFM_MT5_SERVER

        On GitHub Actions the terminal should already be running (started
        via command-line with /login /password /server flags in a previous
        workflow step).  mt5.initialize(path=...) just attaches to it.
        """
        import time

        login = login or int(os.environ.get("HFM_MT5_LOGIN", 0))
        password = password or os.environ.get("HFM_MT5_PASSWORD", "")
        server = server or os.environ.get("HFM_MT5_SERVER", "")

        if not all([login, password, server]):
            logger.error("Missing MT5 credentials. Set HFM_MT5_LOGIN, "
                         "HFM_MT5_PASSWORD, HFM_MT5_SERVER.")
            return False

        # ── Try initialize with retries ──────────────────────
        # Strategy: first try minimal init (just path) to attach to
        # an already-running terminal. If that fails, try full init
        # with credentials so initialize() can start the terminal itself.

        initialized = False
        for attempt in range(1, retries + 1):
            logger.info(f"MT5 initialize attempt {attempt}/{retries}...")

            # Attempt A: minimal init — attach to running terminal
            init_kwargs = {"timeout": 60_000}
            if path:
                init_kwargs["path"] = path

            if mt5.initialize(**init_kwargs):
                logger.info("Attached to running MT5 terminal")
                initialized = True
                break

            err_a = mt5.last_error()
            logger.warning(f"  Attach failed: {err_a}")
            mt5.shutdown()

            # Attempt B: full init — let initialize() start terminal
            logger.info("  Trying full initialize with credentials...")
            init_kwargs_full = {
                "login": login,
                "password": password,
                "server": server,
                "timeout": 120_000,
                "portable": True,
            }
            if path:
                init_kwargs_full["path"] = path

            if mt5.initialize(**init_kwargs_full):
                logger.info("MT5 started and connected via full init")
                initialized = True
                break

            err_b = mt5.last_error()
            logger.warning(f"  Full init also failed: {err_b}")
            mt5.shutdown()

            if attempt < retries:
                wait = retry_delay * attempt
                logger.info(f"  Waiting {wait}s before retry...")
                time.sleep(wait)

        if not initialized:
            err = mt5.last_error()
            logger.error(f"MT5 init failed after {retries} attempts: {err}")
            return False

        # ── Ensure we're logged in to the right account ──────
        acct = mt5.account_info()
        if acct is None or acct.login != login:
            logger.info(f"Logging in to account {login}...")
            if not mt5.login(login=login, password=password, server=server):
                err = mt5.last_error()
                logger.error(f"MT5 login failed: {err}")
                mt5.shutdown()
                return False

        self._connected = True
        info = mt5.account_info()
        if info:
            logger.info(
                f"Connected to {server} | Account {login} | "
                f"Balance: {info.balance} {info.currency}"
            )
        else:
            logger.info("Connected to MT5 (account info not yet available)")
        return True

    def disconnect(self):
        """Shut down MT5 terminal connection."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected.")

    def is_connected(self) -> bool:
        return self._connected

    # ── Account Info ──────────────────────────────────────

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get current account balance, equity, margin, etc."""
        self._ensure_connected()
        info = mt5.account_info()
        if info is None:
            return None
        return AccountInfo(
            login=info.login,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            currency=info.currency,
            leverage=info.leverage,
            server=info.server,
        )

    # ── Symbol Info ───────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get symbol details (tick size, lot constraints, spread)."""
        self._ensure_connected()
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(f"Symbol {symbol} not found. Enabling...")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to enable symbol {symbol}")
                return None
            info = mt5.symbol_info(symbol)
            if info is None:
                return None

        return {
            "symbol": info.name,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
            "point": info.point,
            "digits": info.digits,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_contract_size": info.trade_contract_size,
            "currency_profit": info.currency_profit,
            "filling_mode": info.filling_mode,
        }

    def list_symbols(self, group: str = None) -> list[str]:
        """List all available symbols (optionally filtered by group)."""
        self._ensure_connected()
        if group:
            symbols = mt5.symbols_get(group=group)
        else:
            symbols = mt5.symbols_get()
        if symbols is None:
            return []
        return [s.name for s in symbols]

    # ── Place Order ───────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        action: str,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "stock_trader_bot",
        deviation: int = 20,
    ) -> OrderResult:
        """
        Place a market order (BUY or SELL).

        Parameters
        ----------
        symbol   : MT5 symbol name (e.g., "EURUSD", "XAUUSD", "BTCUSD")
        action   : "BUY" or "SELL"
        volume   : Lot size (e.g., 0.01)
        sl       : Stop loss price (0 = no SL)
        tp       : Take profit price (0 = no TP)
        comment  : Order comment
        deviation: Max price deviation in points
        """
        self._ensure_connected()

        # Enable symbol if needed
        if not mt5.symbol_select(symbol, True):
            return OrderResult(
                success=False,
                message=f"Failed to enable symbol {symbol}",
            )

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(
                success=False,
                message=f"No tick data for {symbol}",
            )

        is_buy = action.upper() in ("BUY", "STRONG BUY")
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        price = tick.ask if is_buy else tick.bid

        # Normalize volume to broker's step
        sym_info = mt5.symbol_info(symbol)
        if sym_info:
            step = sym_info.volume_step
            volume = max(sym_info.volume_min,
                         min(sym_info.volume_max,
                             round(volume / step) * step))
            # Ensure clean float
            volume = round(volume, 2)

        # Detect supported filling mode from symbol info
        filling_type = mt5.ORDER_FILLING_FOK  # default
        if sym_info:
            fm = sym_info.filling_mode
            if fm & 1:    # SYMBOL_FILLING_FOK
                filling_type = mt5.ORDER_FILLING_FOK
            elif fm & 2:  # SYMBOL_FILLING_IOC
                filling_type = mt5.ORDER_FILLING_IOC
            else:
                filling_type = mt5.ORDER_FILLING_RETURN
            logger.info(f"Symbol {symbol}: filling_mode={fm}, using {filling_type}")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        # Pre-check
        check = mt5.order_check(request)
        if check is None or check.retcode != 0:
            msg = f"Order check failed: {check}"
            logger.warning(msg)
            # Try all filling modes as fallback
            for alt_fill in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC,
                             mt5.ORDER_FILLING_RETURN]:
                if alt_fill == filling_type:
                    continue
                request["type_filling"] = alt_fill
                check = mt5.order_check(request)
                if check and check.retcode == 0:
                    logger.info(f"Filling mode fallback worked: {alt_fill}")
                    break
            else:
                return OrderResult(
                    success=False,
                    retcode=check.retcode if check else -1,
                    message=str(check),
                )

        # Send order
        result = mt5.order_send(request)
        if result is None:
            return OrderResult(success=False, message="order_send returned None")

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"Order placed: {action} {volume} {symbol} @ {result.price} "
                f"| SL={sl} TP={tp} | Ticket={result.order}"
            )
            return OrderResult(
                success=True,
                ticket=result.order,
                retcode=result.retcode,
                message="Order executed",
                volume=result.volume,
                price=result.price,
            )
        else:
            msg = f"Order failed: retcode={result.retcode}, comment={result.comment}"
            logger.error(msg)
            return OrderResult(
                success=False,
                retcode=result.retcode,
                message=msg,
            )

    # ── Modify Position ───────────────────────────────────

    def modify_position(
        self, ticket: int, sl: float = None, tp: float = None
    ) -> OrderResult:
        """
        Modify SL/TP on an open position (e.g., trail stop).
        """
        self._ensure_connected()

        position = self._get_position_by_ticket(ticket)
        if position is None:
            return OrderResult(
                success=False, message=f"Position {ticket} not found"
            )

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
            "magic": MAGIC_NUMBER,
        }

        result = mt5.order_send(request)
        if result is None:
            return OrderResult(success=False, message="modify returned None")

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"Position {ticket} modified: SL={request['sl']}, TP={request['tp']}"
            )
            return OrderResult(
                success=True, ticket=ticket, retcode=result.retcode,
                message="Position modified",
            )
        else:
            msg = f"Modify failed: retcode={result.retcode}, comment={result.comment}"
            logger.error(msg)
            return OrderResult(success=False, retcode=result.retcode, message=msg)

    # ── Close Position ────────────────────────────────────

    def close_position(
        self, ticket: int, volume: float = None, comment: str = "bot_close"
    ) -> OrderResult:
        """
        Close an open position (full or partial).

        Parameters
        ----------
        ticket  : Position ticket number
        volume  : Lots to close (None = close all)
        """
        self._ensure_connected()

        position = self._get_position_by_ticket(ticket)
        if position is None:
            return OrderResult(
                success=False, message=f"Position {ticket} not found"
            )

        close_volume = volume if volume else position.volume
        is_buy = (position.type == mt5.ORDER_TYPE_BUY)

        # Close a BUY with a SELL and vice versa
        order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(position.symbol)
        price = tick.bid if is_buy else tick.ask

        # Detect filling mode
        sym_info = mt5.symbol_info(position.symbol)
        filling_type = mt5.ORDER_FILLING_FOK
        if sym_info:
            fm = sym_info.filling_mode
            if fm & 1:
                filling_type = mt5.ORDER_FILLING_FOK
            elif fm & 2:
                filling_type = mt5.ORDER_FILLING_IOC
            else:
                filling_type = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": close_volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,
        }

        result = mt5.order_send(request)
        if result is None:
            return OrderResult(success=False, message="close returned None")

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"Position {ticket} closed: {close_volume} lots @ {result.price}"
            )
            return OrderResult(
                success=True, ticket=result.order, retcode=result.retcode,
                message="Position closed", volume=result.volume,
                price=result.price,
            )
        else:
            # Try all filling modes as fallback
            for alt_fill in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC,
                             mt5.ORDER_FILLING_RETURN]:
                if alt_fill == filling_type:
                    continue
                request["type_filling"] = alt_fill
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    return OrderResult(
                        success=True, ticket=result.order, retcode=result.retcode,
                        message="Position closed",
                        volume=result.volume, price=result.price,
                    )
            msg = f"Close failed: retcode={result.retcode if result else 'None'}"
            logger.error(msg)
            return OrderResult(success=False, message=msg)

    # ── Query Positions ───────────────────────────────────

    def get_open_positions(self, symbol: str = None) -> list[PositionInfo]:
        """Get all open positions (optionally filtered by symbol)."""
        self._ensure_connected()

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result = []
        for p in positions:
            # Only return positions opened by this bot
            if p.magic != MAGIC_NUMBER:
                continue
            result.append(PositionInfo(
                ticket=p.ticket,
                symbol=p.symbol,
                type="BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                volume=p.volume,
                open_price=p.price_open,
                sl=p.sl,
                tp=p.tp,
                profit=p.profit,
                comment=p.comment,
                magic=p.magic,
                time_str=str(p.time),
            ))
        return result

    def get_all_positions(self) -> list[PositionInfo]:
        """Get ALL open positions (including non-bot ones)."""
        self._ensure_connected()
        positions = mt5.positions_get()
        if positions is None:
            return []
        result = []
        for p in positions:
            result.append(PositionInfo(
                ticket=p.ticket,
                symbol=p.symbol,
                type="BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                volume=p.volume,
                open_price=p.price_open,
                sl=p.sl,
                tp=p.tp,
                profit=p.profit,
                comment=p.comment,
                magic=p.magic,
                time_str=str(p.time),
            ))
        return result

    # ── Internals ─────────────────────────────────────────

    def _ensure_connected(self):
        if not self._connected:
            raise ConnectionError(
                "Not connected to MT5. Call connect() first."
            )

    def _get_position_by_ticket(self, ticket: int):
        """Get a raw MT5 position by ticket."""
        positions = mt5.positions_get(ticket=ticket)
        if positions and len(positions) > 0:
            return positions[0]
        return None
