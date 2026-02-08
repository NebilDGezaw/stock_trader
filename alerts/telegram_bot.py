#!/usr/bin/env python3
"""
Telegram Alert Bot — sends ICT/SMC buy/sell signals to Telegram.

Designed to run headlessly via GitHub Actions on a cron schedule.
Requires two environment variables:
    TELEGRAM_BOT_TOKEN  — from @BotFather
    TELEGRAM_CHAT_ID    — your user or group chat ID

Usage:
    python -m alerts.telegram_bot                        # scan default watchlist
    python -m alerts.telegram_bot --tickers AAPL TSLA    # specific tickers
    python -m alerts.telegram_bot --asset crypto         # crypto watchlist
    python -m alerts.telegram_bot --all                  # scan all asset classes
"""

import sys
import os
import argparse
import urllib.request
import urllib.parse
import json
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from data.fetcher import StockDataFetcher
from strategies.smc_strategy import SMCStrategy
from models.signals import TradeAction, MarketBias


# ──────────────────────────────────────────────────────────
#  Telegram API (using urllib — no extra dependencies)
# ──────────────────────────────────────────────────────────

def send_telegram(token: str, chat_id: str, text: str, parse_mode: str = "HTML"):
    """Send a message via the Telegram Bot API."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            if not result.get("ok"):
                print(f"  [!] Telegram API error: {result}")
            return result
    except Exception as e:
        print(f"  [!] Failed to send Telegram message: {e}")
        return None


# ──────────────────────────────────────────────────────────
#  Analysis + Alert Logic
# ──────────────────────────────────────────────────────────

ACTION_EMOJI = {
    TradeAction.STRONG_BUY:  "🟢🟢",
    TradeAction.BUY:         "🟢",
    TradeAction.HOLD:        "🟡",
    TradeAction.SELL:        "🔴",
    TradeAction.STRONG_SELL: "🔴🔴",
}

BIAS_EMOJI = {
    MarketBias.BULLISH: "📈",
    MarketBias.BEARISH: "📉",
    MarketBias.NEUTRAL: "➡️",
}


def analyze_ticker(ticker: str, period: str = "6mo", interval: str = "1d"):
    """Run SMC analysis on a single ticker. Returns (strategy, setup) or None."""
    try:
        df = StockDataFetcher(ticker).fetch(period=period, interval=interval)
        strategy = SMCStrategy(df, ticker=ticker).run()
        return strategy, strategy.trade_setup
    except Exception as e:
        print(f"  [!] Error analyzing {ticker}: {e}")
        return None, None


MIN_RISK_REWARD = 2.0  # Minimum 1:2 R:R to send an alert


def is_actionable(setup) -> bool:
    """Only alert on buy/sell signals with at least 1:2 risk-reward."""
    if not setup:
        return False
    if setup.action not in (
        TradeAction.STRONG_BUY, TradeAction.BUY,
        TradeAction.SELL, TradeAction.STRONG_SELL,
    ):
        return False
    if setup.risk_reward < MIN_RISK_REWARD:
        return False
    return True


def format_alert(setup, strategy) -> str:
    """Format a single ticker alert as an HTML Telegram message."""
    action_em = ACTION_EMOJI.get(setup.action, "")
    bias_em = BIAS_EMOJI.get(setup.bias, "")

    # Price formatting
    price = setup.entry_price
    if price >= 1:
        pfmt = f"${price:,.2f}"
        sl_fmt = f"${setup.stop_loss:,.2f}"
        tp_fmt = f"${setup.take_profit:,.2f}"
    else:
        pfmt = f"${price:.4f}"
        sl_fmt = f"${setup.stop_loss:.4f}"
        tp_fmt = f"${setup.take_profit:.4f}"

    # Signal summary (show fakeout warnings inline)
    signal_lines = []
    fakeout_warnings = 0
    for sig in setup.signals[:6]:  # limit to top 6
        nice = sig.signal_type.value.replace("_", " ").title()
        icon = "▲" if sig.bias == MarketBias.BULLISH else ("▼" if sig.bias == MarketBias.BEARISH else "●")
        warn = ""
        if "⚠" in sig.details:
            fakeout_warnings += 1
            warn = " ⚠"
        signal_lines.append(f"  {icon} {nice} (+{sig.score}){warn}")
    signals_text = "\n".join(signal_lines) if signal_lines else "  No signals"

    confidence = "🟢 HIGH" if fakeout_warnings == 0 else (
        "🟡 MODERATE" if fakeout_warnings <= 2 else "🔴 LOW"
    )

    return (
        f"{action_em} <b>{setup.action.value}</b> — <b>{setup.ticker}</b>\n"
        f"{'─' * 28}\n"
        f"{bias_em} Bias: <b>{setup.bias.value.upper()}</b>  |  Score: <b>{setup.composite_score}</b>\n"
        f"🛡 Confidence: <b>{confidence}</b>\n"
        f"\n"
        f"💰 Entry:  <code>{pfmt}</code>\n"
        f"🛑 SL:     <code>{sl_fmt}</code>\n"
        f"🎯 TP:     <code>{tp_fmt}</code>\n"
        f"⚖️ R:R:    <code>1:{setup.risk_reward:.1f}</code>\n"
        f"📊 Size:   <code>{setup.position_size} units</code>\n"
        f"\n"
        f"<b>Signals:</b>\n"
        f"<code>{signals_text}</code>\n"
    )


def format_summary(results: list, label: str = "") -> str:
    """Format the scan summary header."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    actionable = [r for r in results if is_actionable(r["setup"])]
    buys = sum(1 for r in actionable if "BUY" in r["setup"].action.value)
    sells = sum(1 for r in actionable if "SELL" in r["setup"].action.value)

    title = f"📊 <b>{label}</b>" if label else "📊 <b>Stock Trader — Market Scan</b>"

    return (
        f"{title}\n"
        f"🕐 {now}\n"
        f"{'─' * 28}\n"
        f"Scanned: <b>{len(results)}</b> tickers\n"
        f"Filter: R:R ≥ 1:{MIN_RISK_REWARD:.0f} + fakeout checks\n"
        f"Actionable: <b>{len(actionable)}</b>  "
        f"(🟢 {buys} buy  |  🔴 {sells} sell)\n"
        f"{'─' * 28}\n"
    )


def format_no_signals(label: str = "") -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    title = f"📊 <b>{label}</b>" if label else "📊 <b>Stock Trader — Market Scan</b>"
    return (
        f"{title}\n"
        f"🕐 {now}\n"
        f"{'─' * 28}\n"
        f"🟡 No signals with R:R ≥ 1:{MIN_RISK_REWARD:.0f} found.\n"
        f"All positions: <b>HOLD</b>\n"
    )


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def get_watchlist(asset: str = None) -> list[str]:
    """Get the default watchlist for an asset class."""
    if asset and asset.lower() in ("crypto", "cryptocurrency"):
        return config.ASSET_CLASSES["Crypto"]["presets"]["Major"]
    elif asset and asset.lower() in ("forex", "fx"):
        return config.ASSET_CLASSES["Forex"]["presets"]["Majors"]
    elif asset and asset.lower() in ("commodities", "commodity"):
        return config.ASSET_CLASSES["Commodities"]["presets"]["Metals"] + \
               config.ASSET_CLASSES["Commodities"]["presets"]["Energy"][:2]
    else:
        # Default: top stocks
        return ["SPY", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA", "META"]


def get_all_watchlists() -> list[str]:
    """Get tickers from all asset classes."""
    tickers = []
    tickers += ["SPY", "AAPL", "MSFT", "TSLA", "GOOGL", "NVDA", "META"]
    tickers += config.ASSET_CLASSES["Crypto"]["presets"]["Major"]
    tickers += config.ASSET_CLASSES["Forex"]["presets"]["Majors"][:4]
    tickers += ["GC=F", "CL=F", "SI=F"]
    return tickers


def main():
    parser = argparse.ArgumentParser(description="Telegram Alert Bot")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to scan")
    parser.add_argument("--asset", type=str, help="Asset class: stocks, crypto, forex, commodities")
    parser.add_argument("--all", action="store_true", help="Scan all asset classes")
    parser.add_argument("--period", default="6mo", help="Data period")
    parser.add_argument("--interval", default="1d", help="Candle interval")
    parser.add_argument("--label", type=str, default="", help="Custom title for the alert header")
    parser.add_argument("--dry-run", action="store_true", help="Print messages without sending")
    args = parser.parse_args()

    # Get secrets
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        if not args.dry_run:
            print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
            print("Set them as environment variables or GitHub Actions secrets.")
            sys.exit(1)
        else:
            print("[dry-run] No Telegram credentials — will print only.\n")

    # Determine watchlist
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.all:
        tickers = get_all_watchlists()
    else:
        tickers = get_watchlist(args.asset)

    print(f"Scanning {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"Period: {args.period}, Interval: {args.interval}\n")

    # Run analysis
    results = []
    for t in tickers:
        print(f"  Analyzing {t}...", end=" ")
        strategy, setup = analyze_ticker(t, args.period, args.interval)
        if setup:
            results.append({"ticker": t, "setup": setup, "strategy": strategy})
            print(f"{setup.action.value} (score: {setup.composite_score})")
        else:
            print("SKIP")

    # Filter actionable
    actionable = [r for r in results if is_actionable(r["setup"])]

    # Build messages
    label = args.label
    messages = []
    if actionable:
        messages.append(format_summary(results, label=label))
        for r in actionable:
            messages.append(format_alert(r["setup"], r["strategy"]))
    else:
        messages.append(format_no_signals(label=label))

    # Send
    full_message = "\n".join(messages)

    # Telegram has a 4096 char limit per message — split if needed
    if len(full_message) > 4000:
        # Send summary first, then individual alerts
        if args.dry_run:
            print("\n" + "=" * 40)
            print(format_summary(results, label=label).replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""))
        else:
            send_telegram(token, chat_id, format_summary(results, label=label))

        for r in actionable:
            msg = format_alert(r["setup"], r["strategy"])
            if args.dry_run:
                print(msg.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""))
            else:
                send_telegram(token, chat_id, msg)
    else:
        if args.dry_run:
            clean = full_message
            for tag in ["<b>", "</b>", "<code>", "</code>"]:
                clean = clean.replace(tag, "")
            print("\n" + "=" * 40)
            print(clean)
        else:
            send_telegram(token, chat_id, full_message)

    print(f"\nDone. {len(actionable)} actionable signals out of {len(results)} scanned.")


if __name__ == "__main__":
    main()
