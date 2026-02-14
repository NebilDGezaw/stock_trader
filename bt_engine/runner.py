"""
CLI runner for the backtesting engine.

Usage:
    python -m bt_engine.runner --ticker AAPL --period 2y --interval 1d --stock-mode
    python -m bt_engine.runner --scan AAPL MSFT TSLA --period 2y
"""

import argparse
import sys

from bt_engine.engine import BacktestEngine, run_multi, comparison_table
from strategies.leveraged_momentum import LeveragedMomentumStrategy
from strategies.crypto_momentum import CryptoMomentumStrategy
from strategies.forex_ict import ForexICTStrategy
from strategies.commodity_strategy import CommodityStrategy


def main():
    parser = argparse.ArgumentParser(description="SMC Strategy Backtester")
    parser.add_argument("--ticker", type=str, help="Single ticker to backtest")
    parser.add_argument("--scan", nargs="+", help="Multiple tickers to compare")
    parser.add_argument("--period", type=str, default="2y", help="Data period (default: 2y)")
    parser.add_argument("--interval", type=str, default="1d", help="Candle interval (default: 1d)")
    parser.add_argument("--stock-mode", action="store_true", help="Enable stock mode")
    parser.add_argument("--window", type=int, default=120, help="Rolling window size (default: 120)")
    parser.add_argument("--csv", type=str, help="Export trades to CSV path")
    parser.add_argument("--leveraged", action="store_true",
                        help="Use LeveragedMomentumStrategy instead of SMC")
    parser.add_argument("--forex", action="store_true",
                        help="Use ForexICTStrategy with 4h candles")
    parser.add_argument("--crypto", action="store_true",
                        help="Use CryptoMomentumStrategy for crypto assets")
    parser.add_argument("--commodity", action="store_true",
                        help="Use CommodityStrategy for gold/silver")

    args = parser.parse_args()

    if not args.ticker and not args.scan:
        parser.print_help()
        sys.exit(1)

    if args.commodity:
        strategy_class = CommodityStrategy
        mode_label = "Commodity Mean-Reversion"
        if args.interval == "1d":
            args.interval = "4h"
        if args.window == 120:
            args.window = 60
    elif args.crypto:
        strategy_class = CryptoMomentumStrategy
        mode_label = "Crypto Momentum"
        if args.interval == "1d":
            args.interval = "4h"
        if args.window == 120:
            args.window = 60
    elif args.forex:
        strategy_class = ForexICTStrategy
        mode_label = "Forex Trend-Continuation"
        if args.interval == "1d":
            args.interval = "4h"
        if args.window == 120:
            args.window = 60
    elif args.leveraged:
        strategy_class = LeveragedMomentumStrategy
        mode_label = "Leveraged Momentum"
    else:
        strategy_class = None
        mode_label = "SMC"

    if args.scan:
        print(f"\nScanning {len(args.scan)} tickers: {', '.join(args.scan)}")
        print(f"Period: {args.period} | Interval: {args.interval} | Strategy: {mode_label} | Stock Mode: {args.stock_mode}\n")

        engines = run_multi(
            args.scan,
            period=args.period,
            interval=args.interval,
            stock_mode=args.stock_mode,
            window=args.window,
            strategy_class=strategy_class,
        )
        print(comparison_table(engines))

        # Print individual summaries
        for ticker, eng in engines.items():
            print(f"\n{eng.summary()}")

    else:
        ticker = args.ticker
        print(f"\nBacktesting {ticker}...")
        print(f"Period: {args.period} | Interval: {args.interval} | Strategy: {mode_label} | Stock Mode: {args.stock_mode}\n")

        engine = BacktestEngine(
            ticker,
            period=args.period,
            interval=args.interval,
            stock_mode=args.stock_mode,
            window=args.window,
            strategy_class=strategy_class,
        )
        engine.run()
        print(engine.summary())

        if args.csv:
            engine.to_csv(args.csv)
            print(f"\nTrades exported to {args.csv}")


if __name__ == "__main__":
    main()
