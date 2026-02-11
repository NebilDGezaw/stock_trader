#!/usr/bin/env python3
"""
Alpaca Account Audit — pull EVERY transaction from the API.
============================================================
Shows:
  1. Account overview (cash, equity, buying power, margin status)
  2. All closed/filled orders with exact fill prices
  3. All current open positions with unrealized PnL
  4. Trade-level activities (FILL events) for exact buy/sell proof
  5. Non-trade activities (dividends, transfers, interest, etc.)

Usage:
    python scripts/alpaca_audit.py
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
except ImportError:
    print("ERROR: alpaca-py not installed. Run: pip install alpaca-py")
    sys.exit(1)


def main():
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    paper = os.environ.get("ALPACA_PAPER", "true").lower() in ("true", "1", "yes")

    if not api_key or not secret_key:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        sys.exit(1)

    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
    mode = "PAPER" if paper else "LIVE"

    # ═══════════════════════════════════════════════════
    # 1. ACCOUNT OVERVIEW
    # ═══════════════════════════════════════════════════
    acct = client.get_account()

    print("=" * 70)
    print(f"  ALPACA ACCOUNT AUDIT  [{mode}]")
    print(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    print(f"\n{'─' * 70}")
    print("  1. ACCOUNT OVERVIEW")
    print(f"{'─' * 70}")
    print(f"  Account ID       : {acct.id}")
    print(f"  Status           : {acct.status}")
    print(f"  Currency         : {acct.currency}")
    print(f"  ")
    print(f"  Cash             : ${float(acct.cash):>12,.2f}")
    print(f"  Portfolio Value  : ${float(acct.portfolio_value):>12,.2f}")
    print(f"  Equity           : ${float(acct.equity):>12,.2f}")
    print(f"  Long Market Value: ${float(acct.long_market_value):>12,.2f}")
    print(f"  Short Market Val : ${float(acct.short_market_value):>12,.2f}")
    print(f"  ")
    print(f"  Buying Power     : ${float(acct.buying_power):>12,.2f}")
    print(f"  Regt Buying Pwr  : ${float(acct.regt_buying_power):>12,.2f}")
    print(f"  Daytrading BP    : ${float(acct.daytrading_buying_power):>12,.2f}")
    print(f"  Initial Margin   : ${float(acct.initial_margin):>12,.2f}")
    print(f"  Maintenance Mrgn : ${float(acct.maintenance_margin):>12,.2f}")
    print(f"  Last Equity      : ${float(acct.last_equity):>12,.2f}")
    print(f"  ")
    print(f"  Multiplier       : {acct.multiplier}x")
    print(f"  PDT Flag         : {acct.pattern_day_trader}")
    print(f"  Day Trade Count  : {acct.daytrade_count}")
    print(f"  Trading Blocked  : {acct.trading_blocked}")
    print(f"  Account Blocked  : {acct.account_blocked}")

    # Explain buying power
    multiplier = int(acct.multiplier) if acct.multiplier else 1
    equity = float(acct.equity)
    bp = float(acct.buying_power)
    print(f"\n  ⚡ BUYING POWER EXPLANATION:")
    print(f"     Equity × {multiplier} = ${equity:,.2f} × {multiplier} = ${equity * multiplier:,.2f}")
    print(f"     Actual BP = ${bp:,.2f}")
    if multiplier > 1:
        print(f"     → You have a MARGIN account with {multiplier}x leverage.")
        print(f"     → ${bp:,.2f} buying power does NOT mean you have that much cash.")
        print(f"     → Your real money (equity) is ${equity:,.2f}.")

    # ═══════════════════════════════════════════════════
    # 2. ALL OPEN POSITIONS
    # ═══════════════════════════════════════════════════
    positions = client.get_all_positions()

    print(f"\n{'─' * 70}")
    print(f"  2. OPEN POSITIONS ({len(positions)} total)")
    print(f"{'─' * 70}")

    total_market_val = 0.0
    total_unrealized = 0.0
    total_cost_basis = 0.0

    if positions:
        print(f"  {'Symbol':<8} {'Side':<6} {'Qty':>6} {'Avg Entry':>10} {'Current':>10} "
              f"{'Mkt Value':>12} {'Unrl PnL':>12} {'PnL%':>8}")
        print(f"  {'─'*8} {'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*12} {'─'*12} {'─'*8}")
        for p in sorted(positions, key=lambda x: x.symbol):
            entry = float(p.avg_entry_price)
            current = float(p.current_price)
            mkt_val = float(p.market_value)
            pnl = float(p.unrealized_pl)
            pnl_pct = float(p.unrealized_plpc) * 100
            qty = float(p.qty)
            cost = entry * qty

            total_market_val += abs(mkt_val)
            total_unrealized += pnl
            total_cost_basis += cost

            emoji = "🟢" if pnl >= 0 else "🔴"
            print(f"  {p.symbol:<8} {str(p.side):<6} {qty:>6.0f} "
                  f"${entry:>9.2f} ${current:>9.2f} "
                  f"${mkt_val:>11,.2f} {emoji}${pnl:>10,.2f} {pnl_pct:>+7.2f}%")

        print(f"\n  TOTALS:")
        print(f"    Cost Basis        : ${total_cost_basis:>12,.2f}")
        print(f"    Market Value      : ${total_market_val:>12,.2f}")
        print(f"    Unrealized PnL    : ${total_unrealized:>12,.2f}")
    else:
        print("  (no open positions)")

    # ═══════════════════════════════════════════════════
    # 3. ALL CLOSED / FILLED ORDERS
    # ═══════════════════════════════════════════════════
    # Fetch ALL closed orders (includes filled, cancelled, expired)
    closed_req = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        limit=500,
        nested=True,
    )
    closed_orders = client.get_orders(closed_req)

    # Also get open orders
    open_req = GetOrdersRequest(
        status=QueryOrderStatus.OPEN,
        nested=True,
    )
    open_orders = client.get_orders(open_req)

    print(f"\n{'─' * 70}")
    print(f"  3. ORDER HISTORY ({len(closed_orders)} closed, {len(open_orders)} open)")
    print(f"{'─' * 70}")

    # Group by parent order (bracket orders have legs)
    filled_orders = []
    cancelled_orders = []
    other_orders = []

    for o in closed_orders:
        status = str(o.status).lower()
        if "filled" in status:
            filled_orders.append(o)
        elif "cancel" in status:
            cancelled_orders.append(o)
        else:
            other_orders.append(o)

    # Show FILLED orders with exact prices
    print(f"\n  FILLED ORDERS ({len(filled_orders)}):")
    print(f"  {'Time':<20} {'Symbol':<8} {'Side':<6} {'Qty':>6} "
          f"{'Fill Price':>11} {'Total $':>12} {'Type':<14} {'Class':<10}")
    print(f"  {'─'*20} {'─'*8} {'─'*6} {'─'*6} {'─'*11} {'─'*12} {'─'*14} {'─'*10}")

    realized_trades = {}  # symbol -> list of fills for P&L calc

    for o in sorted(filled_orders, key=lambda x: x.filled_at or x.created_at):
        filled_at = o.filled_at or o.created_at
        time_str = str(filled_at)[:19] if filled_at else "?"
        symbol = o.symbol
        side = str(o.side)
        qty = float(o.filled_qty) if o.filled_qty else 0
        fill_price = float(o.filled_avg_price) if o.filled_avg_price else 0
        total = qty * fill_price
        otype = str(o.type) if o.type else "?"
        oclass = str(o.order_class) if o.order_class else ""

        print(f"  {time_str:<20} {symbol:<8} {side:<6} {qty:>6.0f} "
              f"${fill_price:>10.4f} ${total:>11,.2f} {otype:<14} {oclass:<10}")

        # Track for P&L
        if symbol not in realized_trades:
            realized_trades[symbol] = []
        realized_trades[symbol].append({
            "time": time_str,
            "side": side,
            "qty": qty,
            "price": fill_price,
            "total": total,
            "type": otype,
            "class": oclass,
        })

        # Show bracket legs if present
        if hasattr(o, "legs") and o.legs:
            for leg in o.legs:
                leg_status = str(leg.status).lower()
                leg_filled = float(leg.filled_avg_price) if leg.filled_avg_price else 0
                leg_qty = float(leg.filled_qty) if leg.filled_qty else 0
                leg_type = str(leg.type) if leg.type else "?"
                leg_side = str(leg.side)
                stop_px = float(leg.stop_price) if leg.stop_price else 0
                limit_px = float(leg.limit_price) if leg.limit_price else 0

                if "filled" in leg_status and leg_filled > 0:
                    leg_total = leg_qty * leg_filled
                    print(f"    └─ LEG: {leg_side:<5} {leg_qty:>4.0f} "
                          f"@ ${leg_filled:>10.4f} (${leg_total:>10,.2f}) "
                          f"[{leg_type}, {leg_status}]")

                    if symbol not in realized_trades:
                        realized_trades[symbol] = []
                    realized_trades[symbol].append({
                        "time": str(leg.filled_at or "")[:19],
                        "side": leg_side,
                        "qty": leg_qty,
                        "price": leg_filled,
                        "total": leg_total,
                        "type": leg_type,
                        "class": "leg",
                    })
                else:
                    target = f"stop=${stop_px:.2f}" if stop_px else f"limit=${limit_px:.2f}"
                    print(f"    └─ LEG: {leg_side:<5} {target} "
                          f"[{leg_type}, {leg_status}]")

    # Show CANCELLED orders count
    if cancelled_orders:
        print(f"\n  CANCELLED ORDERS ({len(cancelled_orders)}):")
        for o in sorted(cancelled_orders, key=lambda x: x.created_at or "")[:20]:
            created = str(o.created_at)[:19] if o.created_at else "?"
            symbol = o.symbol
            side = str(o.side)
            qty = float(o.qty) if o.qty else 0
            otype = str(o.type) if o.type else "?"
            stop_px = float(o.stop_price) if o.stop_price else 0
            limit_px = float(o.limit_price) if o.limit_price else 0

            price_info = ""
            if stop_px:
                price_info = f"stop=${stop_px:.2f}"
            elif limit_px:
                price_info = f"limit=${limit_px:.2f}"

            print(f"  {created:<20} {symbol:<8} {side:<6} {qty:>6.0f} "
                  f"{otype:<14} {price_info}")

        if len(cancelled_orders) > 20:
            print(f"  ... and {len(cancelled_orders) - 20} more cancelled orders")

    # Show OPEN orders
    if open_orders:
        print(f"\n  PENDING/OPEN ORDERS ({len(open_orders)}):")
        for o in open_orders:
            created = str(o.created_at)[:19] if o.created_at else "?"
            symbol = o.symbol
            side = str(o.side)
            qty = float(o.qty) if o.qty else 0
            otype = str(o.type) if o.type else "?"
            status = str(o.status)
            stop_px = float(o.stop_price) if o.stop_price else 0
            limit_px = float(o.limit_price) if o.limit_price else 0

            price_info = ""
            if stop_px:
                price_info = f"stop=${stop_px:.2f}"
            elif limit_px:
                price_info = f"limit=${limit_px:.2f}"

            print(f"  {created:<20} {symbol:<8} {side:<6} {qty:>6.0f} "
                  f"{otype:<14} {status:<12} {price_info}")

    # ═══════════════════════════════════════════════════
    # 4. REALIZED P&L PER SYMBOL (from fills)
    # ═══════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("  4. REALIZED P&L RECONSTRUCTION (from actual fills)")
    print(f"{'─' * 70}")

    grand_realized = 0.0
    symbols_with_closed = []

    for symbol in sorted(realized_trades.keys()):
        fills = realized_trades[symbol]

        # Separate buys and sells
        buys = [f for f in fills if "buy" in f["side"].lower()]
        sells = [f for f in fills if "sell" in f["side"].lower()]

        total_bought = sum(f["total"] for f in buys)
        total_sold = sum(f["total"] for f in sells)
        qty_bought = sum(f["qty"] for f in buys)
        qty_sold = sum(f["qty"] for f in sells)

        if qty_bought > 0 and qty_sold > 0:
            # Some round trips
            closed_qty = min(qty_bought, qty_sold)
            avg_buy = total_bought / qty_bought if qty_bought else 0
            avg_sell = total_sold / qty_sold if qty_sold else 0
            realized = (avg_sell - avg_buy) * closed_qty
            remaining = qty_bought - qty_sold

            grand_realized += realized
            symbols_with_closed.append(symbol)

            emoji = "🟢" if realized >= 0 else "🔴"
            print(f"\n  {symbol}:")
            print(f"    Bought : {qty_bought:>6.0f} shares, avg ${avg_buy:.4f}, total ${total_bought:,.2f}")
            print(f"    Sold   : {qty_sold:>6.0f} shares, avg ${avg_sell:.4f}, total ${total_sold:,.2f}")
            print(f"    Closed : {closed_qty:.0f} shares")
            if remaining > 0:
                print(f"    Still open: {remaining:.0f} shares")
            print(f"    {emoji} Realized P&L: ${realized:+,.2f}")
        elif qty_bought > 0 and qty_sold == 0:
            avg_buy = total_bought / qty_bought if qty_bought else 0
            print(f"\n  {symbol}:")
            print(f"    Bought : {qty_bought:>6.0f} shares, avg ${avg_buy:.4f}, total ${total_bought:,.2f}")
            print(f"    (Still fully open — no sells yet)")
        elif qty_sold > 0 and qty_bought == 0:
            avg_sell = total_sold / qty_sold if qty_sold else 0
            print(f"\n  {symbol}:")
            print(f"    Sold short: {qty_sold:>6.0f} shares, avg ${avg_sell:.4f}, total ${total_sold:,.2f}")
            print(f"    (Short position — no covers yet)")

    print(f"\n  {'═' * 60}")
    print(f"  GRAND TOTAL REALIZED P&L: ${grand_realized:+,.2f}")
    print(f"  UNREALIZED P&L (open)   : ${total_unrealized:+,.2f}")
    print(f"  COMBINED                : ${grand_realized + total_unrealized:+,.2f}")
    print(f"  {'═' * 60}")

    # ═══════════════════════════════════════════════════
    # 5. ACCOUNT ACTIVITIES (transfers, dividends, etc.)
    # ═══════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("  5. NON-TRADE ACTIVITIES (transfers, dividends, fees, interest)")
    print(f"{'─' * 70}")

    try:
        # Get non-trade activities
        activities = client.get_account_activities(activity_types="TRANS,DIV,INT,FEE,CFEE,CSD,CSW")
        if activities:
            for a in activities[:30]:
                date = str(getattr(a, "date", getattr(a, "transaction_time", "")))[:19]
                atype = getattr(a, "activity_type", "?")
                net = getattr(a, "net_amount", getattr(a, "qty", "?"))
                desc = getattr(a, "description", "")
                symbol = getattr(a, "symbol", "")

                print(f"  {date:<20} {atype:<8} ${float(net):>12,.2f}  {symbol} {desc}")
        else:
            print("  (no non-trade activities found)")
    except Exception as e:
        print(f"  (Could not fetch activities: {e})")

    # ═══════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'═' * 70}")
    cash = float(acct.cash)
    equity = float(acct.equity)
    last_eq = float(acct.last_equity)
    portfolio = float(acct.portfolio_value)
    bp = float(acct.buying_power)

    print(f"  Cash in account   : ${cash:>12,.2f}")
    print(f"  Portfolio value   : ${portfolio:>12,.2f}")
    print(f"  Equity            : ${equity:>12,.2f}")
    print(f"  Last equity (EOD) : ${last_eq:>12,.2f}")
    print(f"  Today's change    : ${equity - last_eq:>+12,.2f}")
    print(f"  Buying power      : ${bp:>12,.2f}  (= equity × {int(acct.multiplier)})")
    print(f"  Margin multiplier : {acct.multiplier}x")
    print(f"\n  ⚠️  Buying power ≠ your money. It includes margin leverage.")
    print(f"     Your actual money = equity = ${equity:,.2f}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
