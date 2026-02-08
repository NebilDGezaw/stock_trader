"""
Decision Engine — orchestrates the full analysis pipeline for one or more
tickers and presents formatted results.
"""
from __future__ import annotations

import pandas as pd
from typing import List
from datetime import datetime

import config
from data.fetcher import StockDataFetcher
from strategies.smc_strategy import SMCStrategy
from models.signals import TradeSetup, TradeAction, MarketBias


class DecisionEngine:
    """
    High-level entry point: fetch data → run strategy → produce decisions.
    """

    def __init__(self, capital: float = None):
        self.capital = capital or config.INITIAL_CAPITAL
        self._results: list[dict] = []

    # ── public API ────────────────────────────────────────

    def analyze_ticker(
        self,
        ticker: str,
        period: str = None,
        interval: str = None,
    ) -> TradeSetup:
        """
        Full pipeline for a single ticker.
        Returns the TradeSetup recommendation.
        """
        fetcher = StockDataFetcher(ticker)
        df = fetcher.fetch(period=period, interval=interval)

        strategy = SMCStrategy(df, ticker=ticker).run()
        setup = strategy.trade_setup

        self._results.append({
            "ticker": ticker,
            "setup": setup,
            "strategy": strategy,
            "df": df,
        })

        return setup

    def scan_tickers(
        self,
        tickers: list[str],
        period: str = None,
        interval: str = None,
    ) -> list[TradeSetup]:
        """
        Scan multiple tickers and return setups sorted by absolute score.
        """
        setups = []
        for t in tickers:
            try:
                setup = self.analyze_ticker(t, period=period, interval=interval)
                setups.append(setup)
            except Exception as e:
                print(f"  [!] Skipping {t}: {e}")
        setups.sort(key=lambda s: abs(s.composite_score), reverse=True)
        return setups

    @property
    def results(self) -> list[dict]:
        return self._results

    # ── formatting helpers ────────────────────────────────

    @staticmethod
    def format_setup(setup: TradeSetup) -> str:
        """Pretty-print a single trade setup."""
        return setup.summary()

    @staticmethod
    def format_scan(setups: list[TradeSetup]) -> str:
        """Format a scan of multiple tickers into a compact table."""
        lines = [
            "",
            f"{'═' * 72}",
            f"  {'TICKER':<8} {'ACTION':<14} {'BIAS':<10} "
            f"{'SCORE':>6}  {'ENTRY':>10}  {'SL':>10}  {'TP':>10}  {'R:R':>5}",
            f"{'─' * 72}",
        ]
        for s in setups:
            lines.append(
                f"  {s.ticker:<8} {s.action.value:<14} {s.bias.value:<10} "
                f"{s.composite_score:>6}  "
                f"${s.entry_price:>9.2f}  "
                f"${s.stop_loss:>9.2f}  "
                f"${s.take_profit:>9.2f}  "
                f"1:{s.risk_reward:.1f}"
            )
        lines.append(f"{'═' * 72}")
        return "\n".join(lines)

    @staticmethod
    def format_signals(setup: TradeSetup) -> str:
        """List all signals that contributed to the decision."""
        lines = [f"\n  Signals for {setup.ticker}:", f"  {'─' * 40}"]
        for sig in setup.signals:
            icon = "▲" if sig.bias == MarketBias.BULLISH else (
                "▼" if sig.bias == MarketBias.BEARISH else "●"
            )
            lines.append(
                f"  {icon} [{sig.signal_type.value}] "
                f"score={sig.score}  {sig.details}"
            )
        lines.append(f"  {'─' * 40}")
        return "\n".join(lines)
