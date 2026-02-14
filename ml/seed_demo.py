"""
Seed realistic demo data for ML analytics testing.
Based on actual portfolio characteristics:
  - Alpaca: ~$100K equity, ~50 trades, net +$6K, tech/leveraged, long-only
  - HFM: ~$76K equity, ~30 trades, early $24K loss then conservative
"""

import json
import random
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

random.seed(42)

# ═══════════════════════════════════════════════════════
#  Alpaca (Stocks) — Realistic Trade History
# ═══════════════════════════════════════════════════════

def generate_alpaca_data():
    tickers = {
        "stock": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSM", "AMD",
                   "LLY", "UNH", "ENPH", "FSLR", "TSLA", "COST", "WMT"],
        "leveraged": ["TQQQ", "SOXL", "UPRO", "TECL", "FNGU"],
    }

    trades = []
    snapshots = []
    base_equity = 100000.0
    equity = base_equity
    now = datetime.utcnow()

    # Generate ~60 trades over last 14 days (since system went live)
    for day_offset in range(14, 0, -1):
        day = now - timedelta(days=day_offset)
        if day.weekday() >= 5:  # skip weekends
            continue

        n_trades = random.randint(3, 7)
        daily_pnl = 0

        for _ in range(n_trades):
            asset_type = random.choice(["stock"] * 3 + ["leveraged"])
            ticker = random.choice(tickers[asset_type])
            hour = random.randint(10, 15)
            timestamp = day.replace(hour=hour, minute=random.randint(0, 59))

            # Realistic P&L distribution: slight positive skew
            if asset_type == "leveraged":
                pnl = random.gauss(80, 350)  # higher variance
                entry = random.uniform(20, 80)
                volume = random.randint(10, 50)
                hold_hours = random.uniform(1, 8)
            else:
                pnl = random.gauss(60, 200)  # moderate
                entry = random.uniform(100, 500)
                volume = random.randint(5, 30)
                hold_hours = random.uniform(4, 48)

            # Early days were more profitable (shorting era)
            if day_offset > 10:
                pnl += random.gauss(100, 100)

            # Recent days mixed (long-only halal)
            if day_offset <= 5:
                pnl -= random.gauss(30, 50)

            r_multiple = pnl / max(abs(pnl) * 0.5, 50) if pnl != 0 else 0
            score = random.randint(3, 8) if pnl > 0 else random.randint(2, 5)

            daily_pnl += pnl
            trades.append({
                "timestamp": timestamp.isoformat(),
                "system": "alpaca",
                "ticker": ticker,
                "asset_type": asset_type,
                "action": "BUY",
                "entry_price": round(entry, 2),
                "exit_price": round(entry * (1 + pnl / (entry * volume)), 2) if entry * volume != 0 else entry,
                "volume": volume,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / (entry * volume) * 100, 2) if entry * volume != 0 else 0,
                "r_multiple": round(r_multiple, 2),
                "hold_duration_hours": round(hold_hours, 1),
                "strategy": "smc" if asset_type == "stock" else "leveraged_momentum",
                "composite_score": score,
                "session": random.choice(["tech", "semis", "healthcare", "leveraged"]),
                "comment": "",
            })

        equity += daily_pnl
        snapshots.append({
            "timestamp": day.replace(hour=21, minute=15).isoformat(),
            "system": "alpaca",
            "equity": round(equity, 2),
            "balance": round(equity * 0.4, 2),  # ~40% cash
            "open_positions": random.randint(3, 8),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl / equity * 100, 2),
            "unrealized_pnl": round(daily_pnl * 0.5, 2),
        })

    DATA_DIR.joinpath("trades_alpaca.json").write_text(json.dumps(trades, indent=2))
    DATA_DIR.joinpath("snapshots_alpaca.json").write_text(json.dumps(snapshots, indent=2))
    print(f"Alpaca: {len(trades)} trades, {len(snapshots)} snapshots, final equity=${equity:,.0f}")


# ═══════════════════════════════════════════════════════
#  HFM (Forex/Crypto/Commodities) — Realistic Trade History
# ═══════════════════════════════════════════════════════

def generate_hfm_data():
    tickers = {
        "forex": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
        "crypto": ["#BTCUSD", "#ETHUSD"],
        "metal": ["XAUUSD", "XAGUSD"],
    }

    trades = []
    snapshots = []
    equity = 76000.0
    now = datetime.utcnow()

    # Phase 1: First 2 days — aggressive, big loss ($24K)
    for day_offset in range(14, 12, -1):
        day = now - timedelta(days=day_offset)
        n_trades = random.randint(4, 8)
        daily_pnl = 0

        for _ in range(n_trades):
            asset_type = random.choice(["forex", "crypto", "metal"])
            ticker = random.choice(tickers[asset_type])

            # Aggressive: big lots, tight stops → large losses
            pnl = random.gauss(-800, 600)  # heavily negative
            volume = random.uniform(0.5, 2.0)
            hold_hours = random.uniform(1, 12)
            entry = {"forex": 1.08, "crypto": 95000, "metal": 2050}[asset_type]
            r_multiple = round(pnl / max(abs(pnl) * 0.4, 100), 2)

            daily_pnl += pnl
            trades.append({
                "timestamp": day.replace(hour=random.randint(6, 20)).isoformat(),
                "system": "hfm",
                "ticker": ticker,
                "asset_type": asset_type,
                "action": random.choice(["BUY", "SELL"]),
                "entry_price": round(entry, 2),
                "exit_price": round(entry * (1 + pnl / 10000), 2),
                "volume": round(volume, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / equity * 100, 2),
                "r_multiple": r_multiple,
                "hold_duration_hours": round(hold_hours, 1),
                "strategy": f"{asset_type}_strategy",
                "composite_score": random.randint(2, 4),
                "session": random.choice(["london", "overlap", "ny", "crypto"]),
                "comment": "",
            })

        equity += daily_pnl
        snapshots.append({
            "timestamp": day.replace(hour=20, minute=45).isoformat(),
            "system": "hfm",
            "equity": round(equity, 2),
            "balance": round(equity, 2),
            "open_positions": random.randint(2, 6),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl / equity * 100, 2),
            "unrealized_pnl": round(daily_pnl * 0.3, 2),
        })

    # Phase 2: Conservative period (after fixes)
    for day_offset in range(12, 0, -1):
        day = now - timedelta(days=day_offset)
        if day.weekday() >= 5 and random.random() > 0.3:
            continue  # Some weekend crypto trades

        n_trades = random.randint(1, 3)  # Much fewer trades
        daily_pnl = 0

        for _ in range(n_trades):
            asset_type = random.choice(["forex", "crypto", "metal"])
            ticker = random.choice(tickers[asset_type])

            # Conservative: small lots, wider stops, modest results
            pnl = random.gauss(30, 150)
            volume = random.uniform(0.01, 0.10)
            hold_hours = random.uniform(4, 72)
            entry = {"forex": 1.08, "crypto": 96000, "metal": 2070}[asset_type]
            r_multiple = round(pnl / max(abs(pnl) * 0.5, 50), 2)

            daily_pnl += pnl
            trades.append({
                "timestamp": day.replace(hour=random.randint(6, 20)).isoformat(),
                "system": "hfm",
                "ticker": ticker,
                "asset_type": asset_type,
                "action": "BUY",
                "entry_price": round(entry, 2),
                "exit_price": round(entry * (1 + pnl / 10000), 2),
                "volume": round(volume, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / equity * 100, 2),
                "r_multiple": r_multiple,
                "hold_duration_hours": round(hold_hours, 1),
                "strategy": f"{asset_type}_strategy",
                "composite_score": random.randint(4, 7),
                "session": random.choice(["london", "overlap", "crypto"]),
                "comment": "",
            })

        equity += daily_pnl
        snapshots.append({
            "timestamp": day.replace(hour=20, minute=45).isoformat(),
            "system": "hfm",
            "equity": round(equity, 2),
            "balance": round(equity, 2),
            "open_positions": random.randint(0, 3),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl / equity * 100, 2),
            "unrealized_pnl": round(daily_pnl * 0.3, 2),
        })

    DATA_DIR.joinpath("trades_hfm.json").write_text(json.dumps(trades, indent=2))
    DATA_DIR.joinpath("snapshots_hfm.json").write_text(json.dumps(snapshots, indent=2))
    print(f"HFM: {len(trades)} trades, {len(snapshots)} snapshots, final equity=${equity:,.0f}")


if __name__ == "__main__":
    generate_alpaca_data()
    generate_hfm_data()
    print("Demo data seeded!")
