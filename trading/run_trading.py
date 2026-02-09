#!/usr/bin/env python3
"""
HFM Trading Runner — entry point for GitHub Actions.
=====================================================
Modes:
    --mode entry     : Analyze tickers and place new trades
    --mode monitor   : Manage open positions (trail, partial close, reversal)
    --mode summary   : Send daily summary to Telegram
    --mode verify    : Verify symbol availability on broker (first-run check)

Usage:
    python -m trading.run_trading --mode entry --session london
    python -m trading.run_trading --mode monitor
    python -m trading.run_trading --mode summary
    python -m trading.run_trading --mode verify
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from data.fetcher import StockDataFetcher
from models.signals import TradeAction, MarketBias
from trading.mt5_client import MT5Client
from trading.executor import TradeExecutor, ExecutorConfig, ExecutionRecord
from trading.position_manager import PositionManager, PositionUpdate
from trading.symbols import to_mt5, verify_symbols

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trading.runner")


# ──────────────────────────────────────────────────────────
#  Session Definitions (mirrors Telegram bot + UI)
# ──────────────────────────────────────────────────────────

SESSIONS = {
    "london": {
        "label": "🇬🇧 London Open — Forex & Metals",
        "tickers": ["EURUSD=X", "GBPUSD=X", "EURGBP=X", "GBPJPY=X",
                     "USDCHF=X", "GC=F", "SI=F"],
        "interval": "1h",
        "period": "1mo",
        "stock_mode": False,
    },
    "ny": {
        "label": "🇺🇸 NY Session — Crypto, Metals & Forex",
        "tickers": ["GC=F", "SI=F", "BTC-USD", "ETH-USD", "SOL-USD",
                     "XRP-USD", "USDJPY=X", "USDCAD=X", "USDCHF=X",
                     "GBPUSD=X"],
        "interval": "1h",
        "period": "1mo",
        "stock_mode": False,
    },
    "overlap": {
        "label": "🔥 London/NY Overlap",
        "tickers": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "SI=F",
                     "BTC-USD", "ETH-USD", "SOL-USD"],
        "interval": "1h",
        "period": "1mo",
        "stock_mode": False,
    },
}


# ──────────────────────────────────────────────────────────
#  Telegram notification helper
# ──────────────────────────────────────────────────────────

def send_telegram(text: str, chat_id: str = None):
    """Send a message via Telegram Bot API."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("Telegram credentials not set — skipping notification")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            if not result.get("ok"):
                logger.error(f"Telegram error: {result}")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")


# ──────────────────────────────────────────────────────────
#  Analysis (reuses existing strategies)
# ──────────────────────────────────────────────────────────

def analyze_ticker(ticker: str, period: str, interval: str, stock_mode: bool):
    """Run the appropriate strategy on a ticker. Returns (strategy, setup)."""
    from bt_engine.engine import _detect_asset_type, _run_strategy

    try:
        asset_type = _detect_asset_type(ticker)

        # Override interval for forex (1h is optimal)
        if asset_type == "forex" and interval == "1d":
            interval = "1h"
            period = "1mo"

        df = StockDataFetcher(ticker).fetch(period=period, interval=interval)
        if df is None or len(df) < 30:
            logger.warning(f"Insufficient data for {ticker}")
            return None, None

        strat = _run_strategy(df, ticker, stock_mode=stock_mode)
        return strat, strat.trade_setup if strat else None
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}")
        return None, None


# ──────────────────────────────────────────────────────────
#  Mode: Entry — analyze and place trades
# ──────────────────────────────────────────────────────────

def mode_entry(client: MT5Client, session_name: str, dry_run: bool):
    """Analyze session tickers and place trades."""
    session = SESSIONS.get(session_name)
    if not session:
        logger.error(f"Unknown session: {session_name}")
        return

    logger.info(f"=== ENTRY MODE: {session['label']} ===")
    logger.info(f"Tickers: {', '.join(session['tickers'])}")

    executor = TradeExecutor(client, ExecutorConfig(
        max_concurrent_positions=config.MAX_OPEN_POSITIONS,
        default_risk_pct=config.RISK_PER_TRADE,
        min_risk_reward=config.RISK_REWARD_MIN,
        max_risk_per_trade=0.0,   # demo: no cap. For live, set to e.g. $1
        dry_run=dry_run,
    ))

    records = []
    for ticker in session["tickers"]:
        logger.info(f"Analyzing {ticker}...")
        strat, setup = analyze_ticker(
            ticker, session["period"], session["interval"],
            stock_mode=session["stock_mode"],
        )

        if setup is None:
            logger.info(f"  {ticker}: No setup")
            continue

        logger.info(
            f"  {ticker}: {setup.action.value} | Score={setup.composite_score} "
            f"| R:R=1:{setup.risk_reward:.1f}"
        )

        if setup.action in (TradeAction.HOLD,):
            continue

        record = executor.execute_single(setup)
        records.append(record)

    # Send Telegram summary
    _send_entry_telegram(session["label"], records, dry_run)

    logger.info(f"=== Entry complete: {len(records)} trades processed ===")


def mode_entry_local(session_name: str):
    """
    Local test mode: run analysis and print what would be traded.
    No MT5 connection needed — works on Mac/Linux.
    """
    session = SESSIONS.get(session_name)
    if not session:
        logger.error(f"Unknown session: {session_name}")
        return

    print(f"\n{'═' * 60}")
    print(f"  LOCAL TEST — {session['label']}")
    print(f"{'═' * 60}")
    print(f"  Tickers: {', '.join(session['tickers'])}")
    print(f"  Interval: {session['interval']}, Period: {session['period']}")
    print(f"{'─' * 60}\n")

    from trading.symbols import to_mt5

    actionable = 0
    for ticker in session["tickers"]:
        print(f"  Analyzing {ticker}...", end=" ")
        strat, setup = analyze_ticker(
            ticker, session["period"], session["interval"],
            stock_mode=session["stock_mode"],
        )

        if setup is None:
            print("No data / no setup")
            continue

        mt5_sym = to_mt5(ticker)
        action = setup.action.value
        rr = setup.risk_reward
        score = setup.composite_score

        # Lot size estimate (assuming $1000 equity, 2% risk)
        sl_dist = abs(setup.entry_price - setup.stop_loss)
        risk_amt = 1000 * 0.02  # $20

        is_actionable = (
            setup.action not in (TradeAction.HOLD,)
            and rr >= config.RISK_REWARD_MIN
        )

        if is_actionable:
            actionable += 1
            emoji = "🟢" if "BUY" in action else "🔴"
            print(f"{emoji} {action}")
            print(f"    MT5 Symbol : {mt5_sym}")
            print(f"    Entry      : {setup.entry_price:.5f}")
            print(f"    Stop Loss  : {setup.stop_loss:.5f}")
            print(f"    Take Profit: {setup.take_profit:.5f}")
            print(f"    R:R        : 1:{rr:.1f}")
            print(f"    Score      : {score}")
            print(f"    Signals    : {len(setup.signals)}")
        else:
            reason = "HOLD" if setup.action == TradeAction.HOLD else f"R:R {rr:.1f} < {config.RISK_REWARD_MIN}"
            print(f"⏭  {action} (score={score}, R:R=1:{rr:.1f}) — {reason}")

    print(f"\n{'─' * 60}")
    print(f"  Actionable signals: {actionable} / {len(session['tickers'])}")
    print(f"  (Would place up to {min(actionable, config.MAX_OPEN_POSITIONS)} orders)")
    print(f"{'═' * 60}\n")


def _send_entry_telegram(label: str, records: list[ExecutionRecord], dry_run: bool):
    """Format and send entry results to Telegram."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    mode_tag = " [DRY RUN]" if dry_run else ""

    executed = [r for r in records if r.executed]
    skipped = [r for r in records if not r.executed]

    lines = [
        f"🤖 <b>HFM Auto-Trader{mode_tag}</b>",
        f"📊 {label}",
        f"🕐 {now}",
        f"{'─' * 28}",
        f"Executed: <b>{len(executed)}</b> | Skipped: <b>{len(skipped)}</b>",
        "",
    ]

    for r in executed:
        lines.append(
            f"✅ <b>{r.action}</b> {r.mt5_symbol}\n"
            f"   📍 Entry: <code>{r.entry_price:.5f}</code>\n"
            f"   🛑 SL: <code>{r.sl:.5f}</code>\n"
            f"   🎯 TP: <code>{r.tp:.5f}</code>\n"
            f"   📐 Vol: <code>{r.volume}</code> | R:R 1:{r.risk_reward:.1f}\n"
            f"   🎫 Ticket: <code>{r.ticket}</code>"
        )

    for r in skipped:
        lines.append(f"⏭ {r.mt5_symbol}: {r.reason}")

    send_telegram("\n".join(lines))


# ──────────────────────────────────────────────────────────
#  Mode: Monitor — manage open positions
# ──────────────────────────────────────────────────────────

def mode_monitor(client: MT5Client, dry_run: bool):
    """Check and manage all open bot positions."""
    logger.info("=== MONITOR MODE ===")

    manager = PositionManager(
        client,
        trail_activation_r=1.0,
        trail_atr_multiplier=1.0,
        partial_close_at_r=1.0,
        partial_close_pct=0.5,
        enable_reversal_close=True,
        dry_run=dry_run,
    )

    updates = manager.manage_all()

    # Only notify if something happened
    actions = [u for u in updates if u.action_taken != "no_change"]
    if actions:
        _send_monitor_telegram(actions, dry_run)

    logger.info(f"=== Monitor complete: {len(actions)} actions taken ===")


def _send_monitor_telegram(updates: list[PositionUpdate], dry_run: bool):
    """Format and send monitor updates to Telegram."""
    mode_tag = " [DRY RUN]" if dry_run else ""
    now = datetime.utcnow().strftime("%H:%M UTC")

    lines = [
        f"🔄 <b>Position Update{mode_tag}</b> — {now}",
        f"{'─' * 28}",
    ]

    for u in updates:
        if u.action_taken == "trail_stop":
            lines.append(
                f"📈 <b>Trail</b> {u.symbol} #{u.ticket}\n"
                f"   SL: <code>{u.old_sl:.5f}</code> → <code>{u.new_sl:.5f}</code>\n"
                f"   {u.reason}"
            )
        elif u.action_taken == "full_close":
            emoji = "✅" if u.pnl >= 0 else "❌"
            lines.append(
                f"{emoji} <b>Closed</b> {u.symbol} #{u.ticket}\n"
                f"   PnL: <code>{u.pnl:+.2f}</code>\n"
                f"   {u.reason}"
            )
        elif u.action_taken == "partial_close":
            lines.append(
                f"✂️ <b>Partial Close</b> {u.symbol} #{u.ticket}\n"
                f"   {u.reason}"
            )

    send_telegram("\n".join(lines))


# ──────────────────────────────────────────────────────────
#  Mode: Summary — daily report
# ──────────────────────────────────────────────────────────

def mode_summary(client: MT5Client):
    """Send end-of-day summary to Telegram."""
    logger.info("=== SUMMARY MODE ===")

    manager = PositionManager(client)
    summary = manager.get_summary()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"📋 <b>Daily Trading Summary</b>",
        f"🕐 {now}",
        f"{'─' * 28}",
        f"💰 Balance: <code>${summary['balance']:,.2f}</code>",
        f"📊 Equity: <code>${summary['equity']:,.2f}</code>",
        f"📈 Open Positions: <b>{summary['total_positions']}</b>",
        f"   🟢 Winning: {summary['winners']}",
        f"   🔴 Losing: {summary['losers']}",
        f"   💵 Total PnL: <code>${summary['total_pnl']:+,.2f}</code>",
        "",
    ]

    if summary["positions"]:
        lines.append("<b>Open Positions:</b>")
        for p in summary["positions"]:
            emoji = "🟢" if p["profit"] >= 0 else "🔴"
            lines.append(
                f"  {emoji} {p['type']} {p['symbol']} "
                f"({p['volume']} lots) "
                f"PnL: <code>${p['profit']:+.2f}</code>"
            )
    else:
        lines.append("No open positions.")

    send_telegram("\n".join(lines))


# ──────────────────────────────────────────────────────────
#  Mode: Verify — check symbol availability
# ──────────────────────────────────────────────────────────

def mode_verify(client: MT5Client):
    """Verify all session tickers are available on the broker."""
    logger.info("=== VERIFY MODE ===")

    all_tickers = set()
    for session in SESSIONS.values():
        all_tickers.update(session["tickers"])

    results = verify_symbols(client, sorted(all_tickers))

    available = sum(1 for v in results.values() if v["available"])
    missing = sum(1 for v in results.values() if not v["available"])

    logger.info(f"Available: {available}, Missing: {missing}")

    if missing > 0:
        logger.warning("Missing symbols — check your HFM account type!")
        for ticker, info in results.items():
            if not info["available"]:
                logger.warning(f"  ✗ {ticker} → {info['mt5_symbol']}")

    # Also list all broker symbols for reference
    all_symbols = client.list_symbols()
    logger.info(f"Total broker symbols: {len(all_symbols)}")

    # Log forex/metals/crypto symbols specifically
    for group_name, pattern in [("Forex", "*USD*"), ("Metals", "*XAU*"),
                                 ("Crypto", "*BTC*")]:
        matches = client.list_symbols(group=pattern)
        if matches:
            logger.info(f"  {group_name} symbols: {', '.join(matches[:20])}")


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HFM Trading Runner")
    parser.add_argument(
        "--mode", required=True,
        choices=["entry", "monitor", "summary", "verify"],
        help="Operating mode",
    )
    parser.add_argument(
        "--session", default="london",
        choices=list(SESSIONS.keys()),
        help="Trading session (for entry mode)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log actions without placing real orders",
    )
    parser.add_argument(
        "--mt5-path", default=None,
        help="Path to MT5 terminal (for Docker setups)",
    )
    parser.add_argument(
        "--local-test", action="store_true",
        help="Run analysis only (no MT5 connection) — for local testing on Mac/Linux",
    )
    args = parser.parse_args()

    # ── Local test mode: analysis only, no MT5 ───────────
    if args.local_test:
        if args.mode == "entry":
            mode_entry_local(args.session)
        else:
            print("Local test mode only supports --mode entry")
        return

    # Connect to MT5
    client = MT5Client()
    if not client.connect(path=args.mt5_path):
        logger.error("Failed to connect to MT5. Exiting.")
        send_telegram("❌ <b>HFM Bot Error</b>\nFailed to connect to MT5 terminal.")
        sys.exit(1)

    try:
        if args.mode == "entry":
            mode_entry(client, args.session, args.dry_run)
        elif args.mode == "monitor":
            mode_monitor(client, args.dry_run)
        elif args.mode == "summary":
            mode_summary(client)
        elif args.mode == "verify":
            mode_verify(client)
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        send_telegram(f"❌ <b>HFM Bot Error</b>\n<code>{e}</code>")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
