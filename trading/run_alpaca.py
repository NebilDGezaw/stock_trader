#!/usr/bin/env python3
"""
Alpaca Trading Runner — entry point for GitHub Actions.
========================================================
Modes:
    --mode entry     : Analyze stocks and place new trades
    --mode monitor   : Manage open positions (trail, reversal detection)
    --mode summary   : Send daily summary to Telegram
    --mode verify    : Verify asset availability on Alpaca

Sessions:
    --session leveraged        : Leveraged ETFs (MSTU, MSTR, MSTZ, TSLL, TQQQ, SOXL, FNGU)
    --session tech             : Mega Tech (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD)
    --session semis            : Semiconductors (AVGO, QCOM, ASML, MU)
    --session healthcare       : Healthcare (ABBV, LLY, ISRG, JNJ)
    --session energy           : Energy — crash hedge (XOM, CVX, COP)
    --session consumer_staples : Consumer Staples — defensive (PG, KO, PEP, CL)
    --session industrials      : Industrials — real economy (CAT, DE, WM, UNP)
    --session clean_energy     : Clean Energy (ENPH, FSLR)
    --session consumer         : Consumer Discretionary (COST, TGT)
    --session stocks           : All non-leveraged stocks combined

Usage:
    python -m trading.run_alpaca --mode entry --session leveraged
    python -m trading.run_alpaca --mode entry --session tech
    python -m trading.run_alpaca --mode monitor
    python -m trading.run_alpaca --mode summary
    python -m trading.run_alpaca --mode verify
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
from trading.alpaca_client import AlpacaClient
from trading.alpaca_executor import (
    AlpacaExecutor, AlpacaExecutorConfig, ExecutionRecord,
    LEVERAGED_TICKERS, TICKER_CATEGORY, CATEGORY_CAPS, TICKER_CAPS,
)
from trading.alpaca_position_manager import AlpacaPositionManager, PositionUpdate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trading.alpaca_runner")


# ──────────────────────────────────────────────────────────
#  Session Definitions
# ──────────────────────────────────────────────────────────

SESSIONS = {
    # ── Leveraged ETFs (hourly scan, 25% portfolio cap) ──
    "leveraged": {
        "label": "⚡ Leveraged ETFs",
        "tickers": ["MSTU", "MSTR", "MSTZ", "TSLL", "TQQQ", "SOXL", "FNGU"],
        "interval": "1h",
        "period": "3mo",
        "stock_mode": True,
    },
    # ── Mega Tech (2h scan, 20% cap) ──
    "tech": {
        "label": "💻 Mega Tech",
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD"],
        "interval": "1d",
        "period": "6mo",
        "stock_mode": True,
    },
    # ── Semiconductors (2h scan, 10% cap) ──
    "semis": {
        "label": "🔬 Semiconductors",
        "tickers": ["AVGO", "QCOM", "ASML", "MU"],
        "interval": "1d",
        "period": "6mo",
        "stock_mode": True,
    },
    # ── Healthcare (2h scan, 10% cap) ──
    "healthcare": {
        "label": "🏥 Healthcare",
        "tickers": ["ABBV", "LLY", "ISRG", "JNJ"],
        "interval": "1d",
        "period": "6mo",
        "stock_mode": True,
    },
    # ── Energy (2h scan, 15% cap) ── *** NEW: PRIMARY CRASH HEDGE ***
    # Oil & gas — goes UP when tech goes DOWN.
    # XOM +80% during 2022 tech crash. Halal (natural resources).
    "energy": {
        "label": "🛢️ Energy",
        "tickers": ["XOM", "CVX", "COP"],
        "interval": "1d",
        "period": "6mo",
        "stock_mode": True,
    },
    # ── Consumer Staples (2h scan, 10% cap) ── *** NEW: DEFENSIVE ***
    # Ultra-low-beta, recession-proof. Halal (household goods, beverages).
    "consumer_staples": {
        "label": "🧴 Consumer Staples",
        "tickers": ["PG", "KO", "PEP", "CL"],
        "interval": "1d",
        "period": "6mo",
        "stock_mode": True,
    },
    # ── Industrials (2h scan, 10% cap) ── *** NEW: REAL ECONOMY ***
    # Infrastructure, logistics, agriculture. Halal (machinery, railroads).
    "industrials": {
        "label": "🏗️ Industrials",
        "tickers": ["CAT", "DE", "WM", "UNP"],
        "interval": "1d",
        "period": "6mo",
        "stock_mode": True,
    },
    # ── Clean Energy (2h scan, 5% cap) ──
    # Trimmed: dropped NIO/RIVN (speculative, 1% caps, no diversification value)
    "clean_energy": {
        "label": "🌿 Clean Energy",
        "tickers": ["ENPH", "FSLR"],
        "interval": "1d",
        "period": "6mo",
        "stock_mode": True,
    },
    # ── Consumer Discretionary (2h scan, 5% cap) ──
    "consumer": {
        "label": "🛒 Consumer Discretionary",
        "tickers": ["COST", "TGT"],
        "interval": "1d",
        "period": "6mo",
        "stock_mode": True,
    },
}

# Legacy alias so existing --session stocks still works
# Includes ALL non-leveraged sessions for a combined scan
SESSIONS["stocks"] = {
    "label": "📈 All Regular Stocks",
    "tickers": (
        SESSIONS["tech"]["tickers"]
        + SESSIONS["semis"]["tickers"]
        + SESSIONS["healthcare"]["tickers"]
        + SESSIONS["energy"]["tickers"]
        + SESSIONS["consumer_staples"]["tickers"]
        + SESSIONS["industrials"]["tickers"]
        + SESSIONS["clean_energy"]["tickers"]
        + SESSIONS["consumer"]["tickers"]
    ),
    "interval": "1d",
    "period": "6mo",
    "stock_mode": True,
}


# ──────────────────────────────────────────────────────────
#  Telegram notification helper
# ──────────────────────────────────────────────────────────

def send_telegram(text: str, chat_id: str = None):
    """
    Send a message via Telegram Bot API.

    Stocks go to PRIVATE chat (TELEGRAM_CHAT_ID), not the group.
    Telegram limits messages to 4096 characters — truncate if needed.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    # Stock alerts → private chat (not group)
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("Telegram credentials not set — skipping notification")
        return

    # Telegram max message length is 4096 chars
    if len(text) > 4000:
        text = text[:3950] + "\n\n... (truncated)"

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
        # Retry without HTML parse mode (in case HTML entities broke it)
        try:
            import re
            clean_text = re.sub(r'<[^>]+>', '', text)
            payload = json.dumps({
                "chat_id": chat_id,
                "text": clean_text,
                "disable_web_page_preview": True,
            }).encode("utf-8")
            req2 = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req2, timeout=15) as resp:
                pass  # just send it
        except Exception:
            pass  # truly failed


# ──────────────────────────────────────────────────────────
#  Analysis (reuses existing strategies)
# ──────────────────────────────────────────────────────────

def analyze_ticker(ticker: str, period: str, interval: str, stock_mode: bool):
    """Run the appropriate strategy on a ticker. Returns (strategy, setup)."""
    from bt_engine.engine import _detect_asset_type, _run_strategy

    try:
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

def _rank_by_relative_strength(tickers: list[str], period: str = "1mo") -> list[str]:
    """
    Rank tickers by relative strength (momentum) over the given period.

    Returns tickers sorted best-to-worst by recent performance.
    Strongest tickers get analyzed first and have priority for position slots.
    """
    try:
        from data.fetcher import StockDataFetcher
        momentum = []
        for ticker in tickers:
            try:
                df = StockDataFetcher(ticker).fetch(period=period, interval="1d")
                if df is not None and len(df) >= 10:
                    # Use 20-day return as relative strength
                    lookback = min(20, len(df) - 1)
                    ret = (df["Close"].iloc[-1] / df["Close"].iloc[-lookback] - 1) * 100
                    momentum.append((ticker, ret))
                else:
                    momentum.append((ticker, 0.0))
            except Exception:
                momentum.append((ticker, 0.0))

        # Sort by highest momentum first
        momentum.sort(key=lambda x: -x[1])

        ranked = [t for t, r in momentum]
        logger.info(
            "Relative strength ranking: " +
            ", ".join(f"{t}({r:+.1f}%)" for t, r in momentum)
        )
        return ranked
    except Exception as e:
        logger.warning(f"Relative strength ranking failed: {e} — using original order")
        return tickers


def mode_entry(client: AlpacaClient, session_name: str, dry_run: bool):
    """Analyze session tickers and place trades."""
    session = SESSIONS.get(session_name)
    if not session:
        logger.error(f"Unknown session: {session_name}")
        return

    # Check if market is open
    if not client.is_market_open():
        logger.warning("Market is CLOSED. Bracket orders will queue for next open.")

    logger.info(f"=== ALPACA ENTRY: {session['label']} ===")

    # Rank tickers by relative strength — strongest first
    tickers = _rank_by_relative_strength(session["tickers"])
    logger.info(f"Tickers (ranked): {', '.join(tickers)}")

    executor = AlpacaExecutor(client, AlpacaExecutorConfig(
        max_concurrent_positions=12,
        max_daily_loss_pct=config.MAX_DAILY_LOSS_PCT,
        default_risk_pct=config.RISK_PER_TRADE,
        leveraged_risk_pct=config.LEVERAGED_MODE.get("risk_per_trade", 0.02),
        min_risk_reward=config.RISK_REWARD_MIN,
        dry_run=dry_run,
    ))

    records = []
    for ticker in tickers:
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

    # Send Telegram summary (to private chat for stocks)
    _send_entry_telegram(session["label"], records, dry_run)

    logger.info(f"=== Entry complete: {len(records)} trades processed ===")


# ──────────────────────────────────────────────────────────
#  Mode: Entry Local (no Alpaca connection)
# ──────────────────────────────────────────────────────────

def mode_entry_local(session_name: str):
    """
    Local test mode: run analysis and print what would be traded.
    No Alpaca connection needed.
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

        action = setup.action.value
        rr = setup.risk_reward
        score = setup.composite_score

        is_actionable = (
            setup.action not in (TradeAction.HOLD,)
            and rr >= config.RISK_REWARD_MIN
        )

        if is_actionable:
            actionable += 1
            emoji = "🟢" if "BUY" in action else "🔴"
            print(f"{emoji} {action}")
            print(f"    Entry      : ${setup.entry_price:.2f}")
            print(f"    Stop Loss  : ${setup.stop_loss:.2f}")
            print(f"    Take Profit: ${setup.take_profit:.2f}")
            print(f"    R:R        : 1:{rr:.1f}")
            print(f"    Score      : {score}")
            print(f"    Signals    : {len(setup.signals)}")
        else:
            reason = "HOLD" if setup.action == TradeAction.HOLD else f"R:R {rr:.1f} < {config.RISK_REWARD_MIN}"
            print(f"⏭  {action} (score={score}, R:R=1:{rr:.1f}) — {reason}")

    print(f"\n{'─' * 60}")
    print(f"  Actionable signals: {actionable} / {len(session['tickers'])}")
    print(f"{'═' * 60}\n")


def _send_entry_telegram(label: str, records: list[ExecutionRecord], dry_run: bool):
    """Format and send entry results to Telegram (private chat)."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    mode_tag = " [DRY RUN]" if dry_run else ""

    executed = [r for r in records if r.executed]
    skipped = [r for r in records if not r.executed]

    lines = [
        f"📊 <b>Alpaca Stock Trader{mode_tag}</b>",
        f"🏷 {label}",
        f"🕐 {now}",
        f"{'─' * 28}",
        f"Executed: <b>{len(executed)}</b> | Skipped: <b>{len(skipped)}</b>",
        "",
    ]

    for r in executed:
        lines.append(
            f"✅ <b>{r.action}</b> {r.ticker}\n"
            f"   📍 Entry: <code>${r.entry_price:.2f}</code>\n"
            f"   🛑 SL: <code>${r.sl:.2f}</code>\n"
            f"   🎯 TP: <code>${r.tp:.2f}</code>\n"
            f"   📐 Shares: <code>{r.qty}</code> | R:R 1:{r.risk_reward:.1f}\n"
            f"   🆔 {r.order_id[:8]}..."
        )

    for r in skipped:
        lines.append(f"⏭ {r.ticker}: {r.reason}")

    send_telegram("\n".join(lines))


# ──────────────────────────────────────────────────────────
#  Mode: Monitor — manage open positions
# ──────────────────────────────────────────────────────────

def mode_monitor(client: AlpacaClient, dry_run: bool):
    """Check and manage all open stock positions."""
    logger.info("=== ALPACA MONITOR MODE ===")

    manager = AlpacaPositionManager(
        client,
        trail_activation_r=1.0,
        trail_atr_multiplier=1.0,
        enable_reversal_close=True,
        partial_close_at_r=1.0,
        partial_close_pct=0.5,
        dry_run=dry_run,
    )

    updates = manager.manage_all()

    # Only notify if something happened
    actions = [u for u in updates if u.action_taken != "no_change"]
    if actions:
        _send_monitor_telegram(actions, dry_run)

    logger.info(f"=== Monitor complete: {len(actions)} actions taken ===")


def _send_monitor_telegram(updates: list[PositionUpdate], dry_run: bool):
    """Format and send monitor updates to Telegram (private chat)."""
    mode_tag = " [DRY RUN]" if dry_run else ""
    now = datetime.utcnow().strftime("%H:%M UTC")

    lines = [
        f"🔄 <b>Alpaca Position Update{mode_tag}</b> — {now}",
        f"{'─' * 28}",
    ]

    for u in updates:
        if u.action_taken == "trail_stop":
            lines.append(
                f"📈 <b>Trail</b> {u.symbol}\n"
                f"   SL: <code>${u.old_sl:.2f}</code> → <code>${u.new_sl:.2f}</code>\n"
                f"   {u.reason}"
            )
        elif u.action_taken == "full_close":
            emoji = "✅" if u.pnl >= 0 else "❌"
            lines.append(
                f"{emoji} <b>Closed</b> {u.symbol}\n"
                f"   PnL: <code>${u.pnl:+.2f}</code>\n"
                f"   {u.reason}"
            )
        elif u.action_taken == "partial_close":
            lines.append(
                f"✂️ <b>Partial Close</b> {u.symbol}\n"
                f"   {u.reason}"
            )

    send_telegram("\n".join(lines))


# ──────────────────────────────────────────────────────────
#  Mode: Summary — daily report
# ──────────────────────────────────────────────────────────

def mode_summary(client: AlpacaClient):
    """Send end-of-day summary to Telegram (private chat)."""
    logger.info("=== ALPACA SUMMARY MODE ===")

    manager = AlpacaPositionManager(client)
    summary = manager.get_summary()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"📋 <b>Alpaca Daily Summary</b>",
        f"🕐 {now}",
        f"{'─' * 28}",
        f"💰 Cash: <code>${summary['balance']:,.2f}</code>",
        f"📊 Equity: <code>${summary['equity']:,.2f}</code>",
        f"🔓 Buying Power: <code>${summary['buying_power']:,.2f}</code>",
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
                f"  {emoji} {p['side'].upper()} {p['symbol']} "
                f"({int(p['qty'])} shares @ ${p['avg_entry']:.2f}) "
                f"Now: ${p['current_price']:.2f} "
                f"PnL: <code>${p['profit']:+.2f}</code> ({p['profit_pct']:+.1f}%)"
            )
    else:
        lines.append("No open positions.")

    send_telegram("\n".join(lines))

    # ── ML Data Collection: save daily snapshot for analytics ──
    try:
        from ml.data_collector import AccountSnapshot, _append_snapshot, collect_alpaca_trades
        snap = AccountSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            system="alpaca",
            equity=summary["equity"],
            balance=summary["balance"],
            open_positions=summary["total_positions"],
            daily_pnl=summary.get("total_pnl", 0),
            unrealized_pnl=summary.get("total_pnl", 0),
        )
        _append_snapshot(snap)
        # Also collect recent trades for ML training data
        collect_alpaca_trades(days=7)
        logger.info("ML snapshot saved for Alpaca")
    except Exception as e:
        logger.debug(f"ML snapshot failed (non-critical): {e}")


# ──────────────────────────────────────────────────────────
#  Mode: Verify — check asset availability
# ──────────────────────────────────────────────────────────

def mode_verify(client: AlpacaClient):
    """Verify all session tickers are available on Alpaca."""
    logger.info("=== ALPACA VERIFY MODE ===")

    all_tickers = set()
    for session in SESSIONS.values():
        all_tickers.update(session["tickers"])

    available = 0
    missing = 0
    for ticker in sorted(all_tickers):
        asset = client.get_asset(ticker)
        if asset and asset["tradable"]:
            logger.info(
                f"  ✓ {ticker} — {asset['name']} "
                f"({asset['exchange']}, "
                f"{'fractionable' if asset['fractionable'] else 'whole shares only'})"
            )
            available += 1
        else:
            logger.warning(f"  ✗ {ticker} — NOT TRADEABLE")
            missing += 1

    logger.info(f"Available: {available}, Missing: {missing}")

    # Account info
    acct = client.get_account_info()
    if acct:
        logger.info(
            f"Account: equity=${acct.equity:,.2f}, "
            f"cash=${acct.balance:,.2f}, "
            f"buying_power=${acct.buying_power:,.2f}, "
            f"PDT={acct.pattern_day_trader}"
        )

    # Market clock
    clock = client.get_clock()
    if clock:
        logger.info(
            f"Market {'OPEN' if clock['is_open'] else 'CLOSED'} | "
            f"Next open: {clock['next_open']} | "
            f"Next close: {clock['next_close']}"
        )


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alpaca Stock Trading Runner")
    parser.add_argument(
        "--mode", required=True,
        choices=["entry", "monitor", "summary", "verify"],
        help="Operating mode",
    )
    parser.add_argument(
        "--session", default="leveraged",
        choices=list(SESSIONS.keys()),
        help="Trading session (for entry mode): leveraged, tech, semis, healthcare, "
             "energy, consumer_staples, industrials, clean_energy, consumer, stocks (all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log actions without placing real orders",
    )
    parser.add_argument(
        "--local-test", action="store_true",
        help="Run analysis only (no Alpaca connection) — for local testing",
    )
    args = parser.parse_args()

    # ── Local test mode: analysis only ────────────────────
    if args.local_test:
        if args.mode == "entry":
            mode_entry_local(args.session)
        else:
            print("Local test mode only supports --mode entry")
        return

    # Connect to Alpaca
    client = AlpacaClient()
    if not client.connect():
        logger.error("Failed to connect to Alpaca. Exiting.")
        send_telegram("❌ <b>Alpaca Bot Error</b>\nFailed to connect to Alpaca API.")
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
        send_telegram(f"❌ <b>Alpaca Bot Error</b>\n<code>{e}</code>")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
