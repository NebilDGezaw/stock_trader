#!/usr/bin/env python3
"""
Stock Trader — ICT & Smart Money Concepts
==========================================
CLI entry point for analyzing stocks using ICT / Smart Money strategies.

Usage:
    python main.py --ticker AAPL
    python main.py --scan AAPL MSFT TSLA GOOGL AMZN
    python main.py --ticker SPY --interval 1h --period 1mo --verbose
"""

import argparse
import sys
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from data.fetcher import StockDataFetcher
from trader.decision_engine import DecisionEngine
from strategies.smc_strategy import SMCStrategy
from models.signals import TradeAction, MarketBias
import config

console = Console()


# ── CLI argument parsing ──────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stock Trader — ICT & Smart Money Concepts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --ticker AAPL\n"
            "  python main.py --scan AAPL MSFT TSLA GOOGL AMZN\n"
            "  python main.py --ticker SPY --interval 1h --period 1mo\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ticker", "-t",
        type=str,
        help="Single ticker to analyze",
    )
    group.add_argument(
        "--scan", "-s",
        nargs="+",
        type=str,
        help="Multiple tickers to scan",
    )
    parser.add_argument(
        "--period", "-p",
        type=str,
        default=config.DEFAULT_PERIOD,
        help=f"Data period (default: {config.DEFAULT_PERIOD})",
    )
    parser.add_argument(
        "--interval", "-i",
        type=str,
        default=config.DEFAULT_INTERVAL,
        help=f"Candle interval (default: {config.DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=config.INITIAL_CAPITAL,
        help=f"Starting capital (default: ${config.INITIAL_CAPITAL:,.0f})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all individual signals",
    )
    return parser


# ── Display helpers ───────────────────────────────────────

def action_style(action: TradeAction) -> str:
    """Return rich style string for a trade action."""
    styles = {
        TradeAction.STRONG_BUY:  "bold green",
        TradeAction.BUY:         "green",
        TradeAction.HOLD:        "yellow",
        TradeAction.SELL:        "red",
        TradeAction.STRONG_SELL: "bold red",
    }
    return styles.get(action, "white")


def bias_style(bias: MarketBias) -> str:
    styles = {
        MarketBias.BULLISH: "green",
        MarketBias.BEARISH: "red",
        MarketBias.NEUTRAL: "yellow",
    }
    return styles.get(bias, "white")


def print_header():
    console.print()
    console.print(
        Panel(
            "[bold cyan]Stock Trader[/bold cyan]  —  "
            "[dim]ICT & Smart Money Concepts[/dim]",
            box=box.DOUBLE,
            expand=False,
        )
    )
    console.print(f"  [dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    console.print()


def print_single_analysis(setup, engine, verbose: bool):
    """Rich-formatted output for a single ticker analysis."""
    strategy = engine.results[-1]["strategy"]
    df = engine.results[-1]["df"]

    # Header
    action_text = Text(setup.action.value, style=action_style(setup.action))
    bias_text = Text(setup.bias.value.upper(), style=bias_style(setup.bias))

    table = Table(
        title=f"Analysis: {setup.ticker}",
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Action", action_text)
    table.add_row("Market Bias", bias_text)
    table.add_row("Composite Score", f"{setup.composite_score}")
    table.add_row(
        "Score Breakdown",
        f"Bullish: {strategy.bullish_score}  |  Bearish: {strategy.bearish_score}",
    )
    table.add_row("", "")
    table.add_row("Entry Price", f"${setup.entry_price:.2f}")
    table.add_row("Stop Loss", f"${setup.stop_loss:.2f}")
    table.add_row("Take Profit", f"${setup.take_profit:.2f}")
    table.add_row("Risk : Reward", f"1 : {setup.risk_reward:.1f}")
    table.add_row("Position Size", f"{setup.position_size} shares")
    table.add_row("", "")
    table.add_row("Data Points", f"{len(df)} candles")
    table.add_row(
        "Date Range",
        f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}",
    )

    console.print(table)

    # Structure summary
    ms = strategy.structure
    console.print(f"\n  [bold]Market Structure[/bold]")
    console.print(f"    Swing Highs found : {len(ms.swing_highs)}")
    console.print(f"    Swing Lows found  : {len(ms.swing_lows)}")
    console.print(f"    Structure signals : {len(ms.signals)}")

    # Order blocks
    ob = strategy.ob_detector
    active_obs = ob.active_blocks()
    console.print(f"\n  [bold]Order Blocks[/bold]")
    console.print(f"    Total found : {len(ob.order_blocks)}")
    console.print(f"    Active      : {len(active_obs)}")

    # Fair Value Gaps
    fvg = strategy.fvg_detector
    active_fvgs = fvg.active_fvgs()
    console.print(f"\n  [bold]Fair Value Gaps[/bold]")
    console.print(f"    Total found : {len(fvg.fvgs)}")
    console.print(f"    Active      : {len(active_fvgs)}")

    # Liquidity
    liq = strategy.liq_analyzer
    console.print(f"\n  [bold]Liquidity[/bold]")
    console.print(f"    Levels found : {len(liq.levels)}")
    console.print(f"    Sweeps       : {sum(1 for l in liq.levels if l.swept)}")

    # Verbose: show all signals
    if verbose:
        console.print(f"\n  [bold]All Signals ({len(setup.signals)})[/bold]")
        for sig in setup.signals:
            if sig.bias == MarketBias.BULLISH:
                icon, style = "▲", "green"
            elif sig.bias == MarketBias.BEARISH:
                icon, style = "▼", "red"
            else:
                icon, style = "●", "yellow"
            console.print(
                f"    [{style}]{icon}[/{style}] "
                f"[dim]{sig.signal_type.value}[/dim]  "
                f"score=+{sig.score}  {sig.details}"
            )

    console.print()


def print_scan_results(setups, verbose: bool):
    """Rich-formatted table for multi-ticker scan."""
    table = Table(
        title="Multi-Ticker Scan Results",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    table.add_column("Ticker", style="bold cyan")
    table.add_column("Action", justify="center")
    table.add_column("Bias", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Stop Loss", justify="right")
    table.add_column("Take Profit", justify="right")
    table.add_column("R:R", justify="right")
    table.add_column("Shares", justify="right")

    for s in setups:
        table.add_row(
            s.ticker,
            Text(s.action.value, style=action_style(s.action)),
            Text(s.bias.value.upper(), style=bias_style(s.bias)),
            str(s.composite_score),
            f"${s.entry_price:.2f}",
            f"${s.stop_loss:.2f}",
            f"${s.take_profit:.2f}",
            f"1:{s.risk_reward:.1f}",
            str(s.position_size),
        )

    console.print(table)

    # Show actionable tickers
    actionable = [
        s for s in setups
        if s.action in (TradeAction.STRONG_BUY, TradeAction.BUY,
                        TradeAction.SELL, TradeAction.STRONG_SELL)
    ]
    if actionable:
        console.print(f"\n  [bold]Actionable Signals ({len(actionable)}):[/bold]")
        for s in actionable:
            style = action_style(s.action)
            console.print(
                f"    [{style}]{s.action.value}[/{style}] "
                f"{s.ticker} @ ${s.entry_price:.2f} "
                f"(score: {s.composite_score})"
            )
    else:
        console.print("\n  [yellow]No actionable signals at this time.[/yellow]")

    console.print()


# ── Main ──────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    print_header()

    engine = DecisionEngine(capital=args.capital)

    if args.ticker:
        console.print(
            f"  Analyzing [bold cyan]{args.ticker}[/bold cyan] "
            f"({args.period}, {args.interval}) ...\n"
        )
        try:
            setup = engine.analyze_ticker(
                args.ticker,
                period=args.period,
                interval=args.interval,
            )
            print_single_analysis(setup, engine, args.verbose)
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            sys.exit(1)

    elif args.scan:
        console.print(
            f"  Scanning [bold cyan]{len(args.scan)}[/bold cyan] tickers "
            f"({args.period}, {args.interval}) ...\n"
        )
        setups = engine.scan_tickers(
            args.scan,
            period=args.period,
            interval=args.interval,
        )
        if setups:
            print_scan_results(setups, args.verbose)
        else:
            console.print("  [red]No data returned for any ticker.[/red]")
            sys.exit(1)


if __name__ == "__main__":
    main()
