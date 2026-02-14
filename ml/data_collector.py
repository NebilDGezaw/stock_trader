"""
Data Collector — Gathers data from all available sources.
==========================================================
Sources:
  1. Alpaca API      → closed trade history, account snapshots
  2. MT5 API         → deal history, account snapshots
  3. Git history     → commits, file changes, correlation with P&L
  4. Market data     → SPY, BTC, VIX for regime context
  5. Persisted JSON  → accumulated historical snapshots

All data is normalized into a common schema and persisted to
ml/data/ for incremental accumulation across runs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════
#  Data Models
# ══════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """Unified trade record across both systems."""
    timestamp: str            # ISO format
    system: str               # "alpaca" or "hfm"
    ticker: str
    asset_type: str           # stock, leveraged, forex, crypto, commodity, metal
    action: str               # BUY or SELL
    entry_price: float = 0.0
    exit_price: float = 0.0
    volume: float = 0.0       # shares or lots
    pnl: float = 0.0          # dollar P&L
    pnl_pct: float = 0.0      # percentage return
    r_multiple: float = 0.0
    hold_duration_hours: float = 0.0
    strategy: str = ""
    composite_score: int = 0
    session: str = ""
    comment: str = ""


@dataclass
class AccountSnapshot:
    """Point-in-time account state."""
    timestamp: str
    system: str               # "alpaca" or "hfm"
    equity: float = 0.0
    balance: float = 0.0
    open_positions: int = 0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class GitCommit:
    """Parsed git commit with file-change context."""
    sha: str
    timestamp: str
    message: str
    files_changed: list = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0
    touches_strategy: bool = False
    touches_executor: bool = False
    touches_config: bool = False
    touches_risk: bool = False


# ══════════════════════════════════════════════════════════
#  Persistence helpers
# ══════════════════════════════════════════════════════════

def _load_json(filename: str) -> list[dict]:
    path = DATA_DIR / filename
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load {path}: {e}")
    return []


def _save_json(filename: str, data: list[dict]):
    path = DATA_DIR / filename
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Saved {len(data)} records to {path}")


def _deduplicate(records: list[dict], key_fields: list[str]) -> list[dict]:
    """Remove duplicates based on composite key."""
    seen = set()
    unique = []
    for r in records:
        key = tuple(r.get(f, "") for f in key_fields)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# ══════════════════════════════════════════════════════════
#  Alpaca Data Collection
# ══════════════════════════════════════════════════════════

def collect_alpaca_trades(days: int = 90) -> list[dict]:
    """Fetch closed trades from Alpaca API."""
    try:
        from trading.alpaca_client import AlpacaClient
        client = AlpacaClient()
        if not client.connect():
            logger.warning("Cannot connect to Alpaca — skipping trade collection")
            return []

        # Get account snapshot
        acct = client.get_account_info()
        if acct:
            snap = AccountSnapshot(
                timestamp=datetime.utcnow().isoformat(),
                system="alpaca",
                equity=acct.equity,
                balance=acct.balance,
                daily_pnl=acct.equity - acct.last_equity if acct.last_equity > 0 else 0,
                daily_pnl_pct=(acct.equity - acct.last_equity) / acct.last_equity * 100 if acct.last_equity > 0 else 0,
            )
            _append_snapshot(snap)

        # Get closed orders/activities for trade history
        trades = _fetch_alpaca_activities(client, days)
        return trades
    except Exception as e:
        logger.error(f"Alpaca data collection failed: {e}")
        return []


def _fetch_alpaca_activities(client, days: int) -> list[dict]:
    """Parse Alpaca account activities into TradeRecords."""
    records = []
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST()

        # Get portfolio history for equity curve
        history = api.get_portfolio_history(
            period=f"{days}D",
            timeframe="1D",
        )
        if history and hasattr(history, 'equity'):
            for i, (ts, eq) in enumerate(zip(history.timestamp, history.equity)):
                snap = AccountSnapshot(
                    timestamp=datetime.utcfromtimestamp(ts).isoformat(),
                    system="alpaca",
                    equity=float(eq),
                    balance=float(history.profit_loss[i]) if hasattr(history, 'profit_loss') else 0,
                    daily_pnl=float(history.profit_loss[i]) if hasattr(history, 'profit_loss') else 0,
                    daily_pnl_pct=float(history.profit_loss_pct[i] * 100) if hasattr(history, 'profit_loss_pct') else 0,
                )
                records.append(asdict(snap))

        # Get closed orders
        orders = api.list_orders(
            status="closed",
            limit=500,
            after=(datetime.utcnow() - timedelta(days=days)).isoformat(),
        )
        for order in orders:
            if order.filled_at and order.filled_avg_price:
                tr = TradeRecord(
                    timestamp=str(order.filled_at),
                    system="alpaca",
                    ticker=order.symbol,
                    asset_type=_classify_alpaca_ticker(order.symbol),
                    action=order.side.upper(),
                    entry_price=float(order.filled_avg_price),
                    volume=float(order.filled_qty or 0),
                    comment=order.client_order_id or "",
                )
                records.append(asdict(tr))

    except ImportError:
        logger.info("alpaca_trade_api not installed — using env-based collection")
        # Fallback: collect from open positions
        positions = client.get_positions()
        for pos in positions:
            tr = TradeRecord(
                timestamp=datetime.utcnow().isoformat(),
                system="alpaca",
                ticker=pos.symbol,
                asset_type=_classify_alpaca_ticker(pos.symbol),
                action="BUY" if pos.side == "long" else "SELL",
                entry_price=pos.avg_entry_price,
                volume=abs(pos.qty),
                pnl=pos.unrealized_pl,
                pnl_pct=pos.unrealized_plpc * 100,
            )
            records.append(asdict(tr))

    except Exception as e:
        logger.error(f"Alpaca activity fetch failed: {e}")

    return records


def _classify_alpaca_ticker(ticker: str) -> str:
    try:
        import config
        if ticker.upper() in getattr(config, "LEVERAGED_TICKERS", []):
            return "leveraged"
    except ImportError:
        pass
    return "stock"


# ══════════════════════════════════════════════════════════
#  HFM (MT5) Data Collection
# ══════════════════════════════════════════════════════════

def collect_hfm_trades(days: int = 90) -> list[dict]:
    """Fetch deal history from MT5."""
    try:
        from trading.mt5_client import MT5Client
        client = MT5Client()
        if not client.connect():
            logger.warning("Cannot connect to MT5 — skipping HFM collection")
            return []

        # Account snapshot
        acct = client.get_account_info()
        if acct:
            snap = AccountSnapshot(
                timestamp=datetime.utcnow().isoformat(),
                system="hfm",
                equity=acct.equity,
                balance=acct.balance,
                unrealized_pnl=acct.equity - acct.balance,
            )
            _append_snapshot(snap)

        return _fetch_mt5_deals(days)
    except Exception as e:
        logger.error(f"HFM data collection failed: {e}")
        return []


def _fetch_mt5_deals(days: int) -> list[dict]:
    """Parse MT5 deal history into TradeRecords."""
    records = []
    try:
        import MetaTrader5 as mt5
        from trading.symbols import get_asset_type_from_mt5

        from_date = datetime.utcnow() - timedelta(days=days)
        to_date = datetime.utcnow()

        deals = mt5.history_deals_get(from_date, to_date)
        if deals is None:
            logger.warning("No MT5 deals returned")
            return []

        # Group deals by position ticket to pair entries/exits
        position_deals: dict[int, list] = {}
        for deal in deals:
            if deal.position_id > 0:
                position_deals.setdefault(deal.position_id, []).append(deal)

        for pos_id, deal_list in position_deals.items():
            entries = [d for d in deal_list if d.entry == mt5.DEAL_ENTRY_IN]
            exits = [d for d in deal_list if d.entry == mt5.DEAL_ENTRY_OUT]

            if not entries:
                continue

            entry_deal = entries[0]
            symbol = entry_deal.symbol
            asset_type = get_asset_type_from_mt5(symbol)

            total_pnl = sum(d.profit for d in deal_list)
            total_commission = sum(d.commission for d in deal_list)
            total_swap = sum(d.swap for d in deal_list)
            net_pnl = total_pnl + total_commission + total_swap

            entry_time = datetime.utcfromtimestamp(entry_deal.time)
            exit_time = datetime.utcfromtimestamp(exits[-1].time) if exits else datetime.utcnow()
            hold_hours = (exit_time - entry_time).total_seconds() / 3600.0

            entry_price = entry_deal.price
            exit_price = exits[-1].price if exits else entry_price

            # Estimate R-multiple
            r_multiple = 0.0
            if entry_price > 0 and exit_price > 0:
                pnl_pips = abs(exit_price - entry_price)
                # Rough R: assume SL was ~1.5x ATR, TP was ~3x ATR
                r_multiple = net_pnl / max(abs(net_pnl) * 0.5, 50.0) if net_pnl != 0 else 0

            tr = TradeRecord(
                timestamp=entry_time.isoformat(),
                system="hfm",
                ticker=symbol,
                asset_type=asset_type,
                action="BUY" if entry_deal.type == mt5.DEAL_TYPE_BUY else "SELL",
                entry_price=entry_price,
                exit_price=exit_price,
                volume=entry_deal.volume,
                pnl=net_pnl,
                pnl_pct=(net_pnl / max(entry_price * entry_deal.volume, 1)) * 100,
                r_multiple=r_multiple,
                hold_duration_hours=hold_hours,
                comment=entry_deal.comment or "",
            )
            records.append(asdict(tr))

    except ImportError:
        logger.warning("MetaTrader5 not available — skipping MT5 deal history")
    except Exception as e:
        logger.error(f"MT5 deal fetch failed: {e}")

    return records


# ══════════════════════════════════════════════════════════
#  Git Commit History
# ══════════════════════════════════════════════════════════

STRATEGY_FILES = {"strategies/", "config.py"}
EXECUTOR_FILES = {"trading/executor.py", "trading/alpaca_executor.py"}
RISK_FILES = {"trading/position_manager.py", "trading/alpaca_position_manager.py"}

def collect_git_history(days: int = 90, repo_path: str = None) -> list[dict]:
    """Parse git log for strategy/config changes."""
    if repo_path is None:
        repo_path = str(Path(__file__).parent.parent)

    try:
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        result = subprocess.run(
            ["git", "log", f"--since={since}", "--pretty=format:%H|%aI|%s",
             "--stat", "--diff-filter=ACDMR"],
            capture_output=True, text=True, cwd=repo_path, timeout=30,
        )

        if result.returncode != 0:
            logger.warning(f"git log failed: {result.stderr}")
            return []

        commits = []
        current_commit = None

        for line in result.stdout.split("\n"):
            if "|" in line and line.count("|") >= 2 and not line.strip().startswith("-"):
                parts = line.split("|", 2)
                if len(parts) == 3 and len(parts[0]) == 40:
                    # Save previous commit
                    if current_commit:
                        commits.append(asdict(current_commit))

                    sha, ts, msg = parts
                    current_commit = GitCommit(
                        sha=sha.strip(),
                        timestamp=ts.strip(),
                        message=msg.strip(),
                    )
            elif current_commit and "|" in line:
                # This is a file stat line like: " strategies/forex_ict.py | 150 ++--"
                file_part = line.split("|")[0].strip()
                if file_part:
                    current_commit.files_changed.append(file_part)
                    if any(file_part.startswith(sf) for sf in STRATEGY_FILES):
                        current_commit.touches_strategy = True
                    if file_part in EXECUTOR_FILES:
                        current_commit.touches_executor = True
                    if file_part in RISK_FILES:
                        current_commit.touches_risk = True
                    if "config" in file_part.lower():
                        current_commit.touches_config = True
            elif current_commit and "insertion" in line or "deletion" in line:
                # Summary line: " 5 files changed, 150 insertions(+), 30 deletions(-)"
                parts = line.split(",")
                for p in parts:
                    p = p.strip()
                    if "insertion" in p:
                        current_commit.insertions = int(p.split()[0])
                    elif "deletion" in p:
                        current_commit.deletions = int(p.split()[0])

        if current_commit:
            commits.append(asdict(current_commit))

        logger.info(f"Collected {len(commits)} git commits from last {days} days")
        return commits

    except Exception as e:
        logger.error(f"Git history collection failed: {e}")
        return []


# ══════════════════════════════════════════════════════════
#  Market Context Data
# ══════════════════════════════════════════════════════════

def collect_market_context(days: int = 90) -> pd.DataFrame:
    """Fetch market regime context data (SPY, VIX, BTC, DXY)."""
    try:
        from data.fetcher import StockDataFetcher

        benchmarks = {
            "SPY": "SPY",          # US market
            "BTC": "BTC-USD",      # crypto sentiment
            "VIX": "^VIX",         # volatility
            "DXY": "DX-Y.NYB",    # dollar strength (affects forex)
            "GOLD": "GC=F",        # safe haven
        }

        frames = {}
        for label, ticker in benchmarks.items():
            try:
                df = StockDataFetcher(ticker).fetch(
                    period=f"{max(days, 60)}d", interval="1d"
                )
                if df is not None and len(df) > 5:
                    frames[f"{label}_close"] = df["Close"]
                    frames[f"{label}_return"] = df["Close"].pct_change()
                    frames[f"{label}_volatility"] = df["Close"].pct_change().rolling(20).std()
            except Exception as e:
                logger.warning(f"Failed to fetch {label} ({ticker}): {e}")

        if frames:
            context = pd.DataFrame(frames)
            logger.info(f"Market context: {len(context)} rows, {len(frames)} series")
            return context

    except Exception as e:
        logger.error(f"Market context collection failed: {e}")

    return pd.DataFrame()


# ══════════════════════════════════════════════════════════
#  Snapshot Accumulation
# ══════════════════════════════════════════════════════════

def _append_snapshot(snap: AccountSnapshot):
    """Append a snapshot to the accumulated history."""
    filename = f"snapshots_{snap.system}.json"
    existing = _load_json(filename)
    existing.append(asdict(snap))
    # Keep last 365 days
    cutoff = (datetime.utcnow() - timedelta(days=365)).isoformat()
    existing = [s for s in existing if s.get("timestamp", "") >= cutoff]
    _save_json(filename, existing)


# ══════════════════════════════════════════════════════════
#  Main Collection Pipeline
# ══════════════════════════════════════════════════════════

def collect_all(system: str = "both", days: int = 90) -> dict:
    """
    Run the full data collection pipeline.

    Returns dict with keys: alpaca_trades, hfm_trades, git_commits, market_context
    """
    result = {
        "alpaca_trades": [],
        "hfm_trades": [],
        "git_commits": [],
        "market_context": pd.DataFrame(),
    }

    if system in ("both", "alpaca"):
        logger.info("Collecting Alpaca data...")
        new_trades = collect_alpaca_trades(days)
        existing = _load_json("trades_alpaca.json")
        merged = _deduplicate(existing + new_trades, ["timestamp", "ticker", "action"])
        _save_json("trades_alpaca.json", merged)
        result["alpaca_trades"] = merged

    if system in ("both", "hfm"):
        logger.info("Collecting HFM data...")
        new_trades = collect_hfm_trades(days)
        existing = _load_json("trades_hfm.json")
        merged = _deduplicate(existing + new_trades, ["timestamp", "ticker", "action"])
        _save_json("trades_hfm.json", merged)
        result["hfm_trades"] = merged

    logger.info("Collecting git history...")
    result["git_commits"] = collect_git_history(days)
    _save_json("git_commits.json", result["git_commits"])

    logger.info("Collecting market context...")
    result["market_context"] = collect_market_context(days)

    return result
