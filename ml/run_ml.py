"""
ML Portfolio Analytics — CLI Entry Point
==========================================
Runs the full ML analysis pipeline and outputs reports.

Usage:
    python -m ml.run_ml --system alpaca              # Alpaca stocks only
    python -m ml.run_ml --system hfm                 # HFM forex/crypto only
    python -m ml.run_ml --system both                # Both systems
    python -m ml.run_ml --system both --telegram     # Send via Telegram
    python -m ml.run_ml --system both --offline      # Use persisted data only
    python -m ml.run_ml --system both --days 180     # 6-month lookback
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
import urllib.parse

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.analyzer import PortfolioAnalyzer
from ml.report import generate_telegram_report, generate_cli_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  Telegram Sender
# ══════════════════════════════════════════════════════════

def send_telegram(text: str, chat_id: str = None):
    """Send report via Telegram Bot API."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("Telegram credentials not set — skipping notification")
        return

    # Split long messages (Telegram limit 4096 chars)
    chunks = _split_message(text, 4000)

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    for i, chunk in enumerate(chunks):
        payload = json.dumps({
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    logger.info(f"Telegram message sent ({i+1}/{len(chunks)})")
        except urllib.error.HTTPError as e:
            if e.code == 400:
                # Retry without HTML
                import re
                clean = re.sub(r"<[^>]+>", "", chunk)
                payload = json.dumps({
                    "chat_id": chat_id,
                    "text": clean,
                    "disable_web_page_preview": True,
                }).encode("utf-8")
                req2 = urllib.request.Request(
                    url, data=payload, headers={"Content-Type": "application/json"}
                )
                try:
                    with urllib.request.urlopen(req2, timeout=15):
                        pass
                except Exception:
                    pass
            else:
                logger.error(f"Telegram send failed: {e}")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")


def _split_message(text: str, max_len: int) -> list[str]:
    """Split a long message at newline boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line

    if current:
        chunks.append(current)

    return chunks


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ML Portfolio Analytics — Health Check & Forecasting"
    )
    parser.add_argument(
        "--system", default="both",
        choices=["alpaca", "hfm", "both"],
        help="Which trading system to analyze",
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Lookback period in days (default: 90)",
    )
    parser.add_argument(
        "--telegram", action="store_true",
        help="Send report via Telegram",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use persisted data only (no API calls)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )

    args = parser.parse_args()

    systems = []
    if args.system in ("alpaca", "both"):
        systems.append("alpaca")
    if args.system in ("hfm", "both"):
        systems.append("hfm")

    reports = []

    for system in systems:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Analyzing: {system.upper()}")
        logger.info(f"{'='*60}")

        analyzer = PortfolioAnalyzer(system=system, lookback_days=args.days)
        report = analyzer.analyze(skip_collection=args.offline)
        reports.append(report)

        # CLI output
        if args.json:
            from dataclasses import asdict
            print(json.dumps(asdict(report), indent=2, default=str))
        else:
            print(generate_cli_report(report))

        # Telegram
        if args.telegram:
            telegram_text = generate_telegram_report(report)
            send_telegram(telegram_text)
            logger.info(f"Telegram report sent for {system}")

    # Summary
    if len(reports) == 2 and not args.json:
        print("\n" + "=" * 60)
        print("  COMBINED SUMMARY")
        print("=" * 60)
        for r in reports:
            label = "Stocks" if r.system == "alpaca" else "Forex/Crypto"
            emoji = "\u2705" if r.health_score >= 60 else "\u26A0\uFE0F" if r.health_score >= 40 else "\u274C"
            print(f"  {emoji} {label:20s}  Health: {r.health_score:.0f}/100 ({r.health_grade})")
            print(f"     P&L: ${r.total_pnl:+,.0f} | WR: {r.win_rate:.0%} | Sharpe: {r.sharpe_ratio:.2f}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
