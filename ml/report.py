"""
Report Generator — Produces formatted reports for Telegram and CLI.
====================================================================
Generates beautiful, data-rich reports from PortfolioReport objects.
Two output modes:
  1. Telegram HTML (with emoji, HTML tags)
  2. CLI text (with box-drawing characters)
"""

from __future__ import annotations

import logging
from datetime import datetime

from ml.analyzer import PortfolioReport

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  Telegram HTML Report
# ══════════════════════════════════════════════════════════

def generate_telegram_report(report: PortfolioReport) -> str:
    """Generate an HTML-formatted report for Telegram."""
    system_label = "STOCKS (Alpaca)" if report.system == "alpaca" else "FOREX/CRYPTO/COMMODITIES (HFM)"
    system_emoji = "\U0001F4C8" if report.system == "alpaca" else "\U0001F30D"

    lines = []

    # ── Header ──
    lines.append(f"<b>{system_emoji} PORTFOLIO HEALTH REPORT</b>")
    lines.append(f"<b>{system_label}</b>")
    lines.append(f"<i>{datetime.utcnow().strftime('%A, %b %d %Y %H:%M UTC')}</i>")
    lines.append("")

    # ── Health Score ──
    score = report.health_score
    grade = report.health_grade
    grade_emoji = _grade_emoji(grade)
    lines.append(f"{grade_emoji} <b>HEALTH SCORE: {score:.0f}/100 ({grade})</b>")

    # ── Goal tracking ──
    if report.weekly_target_pct > 0:
        target_emoji = "\u2705" if report.on_track else "\u274C"
        lines.append(f"{target_emoji} <b>Weekly Target: {report.weekly_target_pct:.1f}%</b> | Actual: <b>{report.weekly_actual_pct:+.1f}%</b>")
        if report.weeks_analyzed > 0:
            lines.append(f"  Weeks on target: {report.weeks_on_target}/{report.weeks_analyzed}")
    lines.append("")

    # ── Week-in-Review ──
    attr = report.week_attribution
    if attr and attr.get("verdict"):
        verdict = attr["verdict"]
        verdict_emojis = {
            "ON_TRACK": "\U0001F7E2", "MARKET_DRIVEN": "\U0001F30A",
            "STRATEGY_FAIL": "\U0001F534", "STRATEGY_WIN": "\U0001F3C6",
            "BAD_LUCK": "\U0001F3B2", "POOR_ENTRIES": "\u26A0\uFE0F",
            "RISK_MANAGEMENT": "\u2696\uFE0F", "CHURNING": "\U0001F504",
            "LOW_ACTIVITY": "\U0001F6AB", "NO_TRADES": "\u2796",
        }
        ve = verdict_emojis.get(verdict, "\u2753")
        lines.append(f"{ve} <b>WEEK IN REVIEW: {verdict.replace('_', ' ')}</b>")
        lines.append(f"  {attr.get('explanation', '')}")
        if attr.get("best_trade"):
            lines.append(
                f"  Best: {attr['best_trade']['ticker']} (${attr['best_trade']['pnl']:+,.0f}) | "
                f"Worst: {attr['worst_trade']['ticker']} (${attr['worst_trade']['pnl']:+,.0f})"
            )
        lines.append("")

    # ── Fear & Greed (HFM only) ──
    if report.fear_greed_index > 0:
        fg = report.fear_greed_index
        fg_bar = "\U0001F7E2" if fg < 30 else "\U0001F7E1" if fg < 55 else "\U0001F7E0" if fg < 75 else "\U0001F534"
        lines.append(f"{fg_bar} <b>Crypto Fear & Greed: {fg}/100 ({report.fear_greed_label})</b>")
        lines.append("")

    # ── Performance Summary ──
    lines.append("\U0001F4CA <b>Performance Summary</b>")
    lines.append(f"  Total P&L:     <b>${report.total_pnl:+,.0f}</b> ({report.total_pnl_pct:+.1f}%)")
    lines.append(f"  Week P&L:      ${report.week_pnl:+,.0f}")
    lines.append(f"  Month P&L:     ${report.month_pnl:+,.0f}")
    lines.append(f"  Win Rate:      {report.win_rate:.0%} ({report.total_trades} trades)")
    lines.append(f"  Avg R:         {report.avg_r_multiple:+.2f}R")
    lines.append(f"  Profit Factor: {report.profit_factor:.2f}")
    lines.append(f"  Sharpe Ratio:  {report.sharpe_ratio:.2f}")
    lines.append(f"  Max Drawdown:  {report.max_drawdown_pct:.1f}%")
    if report.avg_hold_hours > 0:
        lines.append(f"  Avg Hold:      {report.avg_hold_hours:.1f}h")
    lines.append("")

    # ── Bayesian Estimate ──
    if report.bayesian_summary:
        bs = report.bayesian_summary
        exp_emoji = "\u2705" if bs.get("expectancy", 0) > 0 else "\u274C"
        lines.append(f"\U0001F9E0 <b>Bayesian Model</b> ({bs.get('confidence', 'low')} confidence)")
        lines.append(f"  Win Rate:    {bs.get('win_rate', 0):.1%}")
        lines.append(f"  Expected R:  {bs.get('expected_r', 0):+.2f}")
        lines.append(f"  {exp_emoji} Expectancy: ${bs.get('expectancy', 0):+.2f}/trade")
        lines.append("")

    # ── Forecasts ──
    lines.append("\U0001F52E <b>ML Forecasts</b>")
    for fc_dict, label in [
        (report.forecast_1w, "Next Week"),
        (report.forecast_1m, "Next Month"),
        (report.forecast_3m, "Next Quarter"),
        (report.forecast_eoy, "End of Year"),
    ]:
        if fc_dict.get("available"):
            low = fc_dict.get("lower_change_pct", 0)
            high = fc_dict.get("upper_change_pct", 0)
            method = fc_dict.get("method", "?")
            lines.append(f"  {label}: <b>{low:+.1f}% to {high:+.1f}%</b> [{method}]")

    # Monte Carlo
    if report.monte_carlo_1m and report.monte_carlo_1m.get("n_simulations", 0) > 0:
        mc = report.monte_carlo_1m
        equity = report.current_equity or 100000
        lines.append("")
        lines.append(f"\U0001F3B2 <b>Monte Carlo (10K sims)</b>")
        lines.append(f"  1 Month range: {mc['percentile_10']*100:+.1f}% to {mc['percentile_90']*100:+.1f}%")
        lines.append(f"  Prob positive: {mc['prob_positive']:.0%}")
        lines.append(f"  Prob >1%:      {mc['prob_above_1pct']:.0%}")
        lines.append(f"  Prob <-5%:     {mc['prob_below_neg5pct']:.0%}")

    if report.monte_carlo_eoy and report.monte_carlo_eoy.get("n_simulations", 0) > 0:
        mc = report.monte_carlo_eoy
        lines.append(f"  EOY range:     {mc['percentile_10']*100:+.1f}% to {mc['percentile_90']*100:+.1f}%")
    lines.append("")

    # ── Benchmark ("Is this worth it?") ──
    bm = report.benchmarks
    if bm and bm.get("verdict"):
        worth_emoji = "\U0001F3C6" if bm.get("worth_it") else "\u274C"
        lines.append(f"{worth_emoji} <b>VS PASSIVE BENCHMARK</b>")
        primary = "SPY" if report.system == "alpaca" else "BTC"
        secondary = "QQQ" if report.system == "alpaca" else "GLD"
        period = bm.get("comparison_period", "")

        # Show returns side by side
        our_ret = bm.get("portfolio", {})
        primary_ret = bm.get(primary, {})
        secondary_ret = bm.get(secondary, {})
        for p_label in ["1w", "1m", "ytd"]:
            ours = our_ret.get(p_label)
            theirs = primary_ret.get(p_label)
            theirs2 = secondary_ret.get(p_label)
            if ours is not None and theirs is not None:
                alpha = ours - theirs
                alpha_emoji = "\u2705" if alpha > 0 else "\u274C"
                line = f"  {p_label.upper():4s} {alpha_emoji} Us: <b>{ours:+.1f}%</b> vs {primary}: {theirs:+.1f}%"
                if theirs2 is not None:
                    line += f" vs {secondary}: {theirs2:+.1f}%"
                line += f"  (alpha: {alpha:+.1f}%)"
                lines.append(line)

        lines.append(f"  <i>{bm['verdict']}</i>")
        if bm.get("sharpe_verdict"):
            lines.append(f"  <i>{bm['sharpe_verdict']}</i>")
        lines.append("")

    # ── Weaknesses ──
    if report.weaknesses:
        lines.append("\u26A0\uFE0F <b>Weaknesses Detected</b>")
        for i, w in enumerate(report.weaknesses[:7], 1):
            severity_emoji = "\U0001F534" if w["severity"] == "high" else "\U0001F7E1"
            metric_label = "win rate" if w["metric"] == "win_rate" else "avg P&L"
            lines.append(
                f"  {severity_emoji} {i}. {w['dimension']}=<b>{w['value']}</b>: "
                f"{metric_label}={w['this_value']} "
                f"(portfolio: {w['portfolio_avg']}, {w['trade_count']} trades)"
            )
        lines.append("")

    # ── Code Change Impact ──
    if report.code_changes:
        lines.append("\U0001F504 <b>Code Change Impact</b>")
        for change in report.code_changes[:5]:
            verdict_emoji = "\u2705" if change["verdict"] == "POSITIVE" else "\u274C" if change["verdict"] == "NEGATIVE" else "\u2796"
            lines.append(
                f"  {verdict_emoji} <code>{change['sha']}</code> {change['message'][:50]}"
            )
            lines.append(
                f"     Win rate: {change['before_win_rate']:.0%} \u2192 {change['after_win_rate']:.0%} | "
                f"Avg P&L: ${change['before_avg_pnl']:.0f} \u2192 ${change['after_avg_pnl']:.0f}"
            )
        lines.append("")

    # ── Top Suggestions ──
    if report.suggestions:
        lines.append("\U0001F4A1 <b>Suggestions</b>")
        for i, suggestion in enumerate(report.suggestions[:5], 1):
            lines.append(f"  {i}. {suggestion}")
        lines.append("")

    # ── Model Info ──
    lines.append(f"\U0001F916 <i>ML: classifier={report.classifier_accuracy:.0%} acc, "
                 f"{report.anomalous_periods} anomalies, "
                 f"{len(report.changepoints)} changepoints</i>")

    if report.market_regime:
        lines.append(f"\U0001F30E <i>Market: {report.market_regime} regime, VIX={report.vix_level:.1f}</i>")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
#  CLI Text Report
# ══════════════════════════════════════════════════════════

def generate_cli_report(report: PortfolioReport) -> str:
    """Generate a box-drawing formatted report for terminal output."""
    system_label = "STOCKS (Alpaca)" if report.system == "alpaca" else "FOREX/CRYPTO/COMMODITIES (HFM)"

    w = 60  # width
    lines = []

    lines.append("\u2554" + "\u2550" * w + "\u2557")
    lines.append("\u2551" + f"  PORTFOLIO HEALTH REPORT".center(w) + "\u2551")
    lines.append("\u2551" + f"  {system_label}".center(w) + "\u2551")
    lines.append("\u2551" + f"  {datetime.utcnow().strftime('%A, %b %d %Y')}".center(w) + "\u2551")
    lines.append("\u255A" + "\u2550" * w + "\u255D")
    lines.append("")

    # Health Score
    score = report.health_score
    grade = report.health_grade
    bar = _progress_bar(score, 100, 30)
    lines.append(f"  HEALTH: [{bar}] {score:.0f}/100 ({grade})")

    # Goal tracking
    if report.weekly_target_pct > 0:
        target_mark = "[HIT]" if report.on_track else "[MISS]"
        lines.append(f"  TARGET: {report.weekly_target_pct:.1f}%/week | This week: {report.weekly_actual_pct:+.1f}% {target_mark}")
        if report.weeks_analyzed > 0:
            lines.append(f"  TRACK RECORD: {report.weeks_on_target}/{report.weeks_analyzed} weeks on target")
    lines.append("")

    # Week-in-Review
    attr = report.week_attribution
    if attr and attr.get("verdict"):
        verdict = attr["verdict"]
        lines.append(f"  WEEK IN REVIEW: {verdict.replace('_', ' ')}")
        explanation = attr.get("explanation", "")
        for wrapped_line in _wrap(explanation, w - 4):
            lines.append(f"    {wrapped_line}")
        if attr.get("best_trade"):
            lines.append(
                f"    Best: {attr['best_trade']['ticker']} "
                f"(${attr['best_trade']['pnl']:+,.0f}) | "
                f"Worst: {attr['worst_trade']['ticker']} "
                f"(${attr['worst_trade']['pnl']:+,.0f})"
            )
        lines.append("")

    # Fear & Greed
    if report.fear_greed_index > 0:
        fg_bar = _progress_bar(report.fear_greed_index, 100, 20)
        lines.append(f"  CRYPTO FEAR & GREED: [{fg_bar}] {report.fear_greed_index}/100 ({report.fear_greed_label})")
        lines.append("")

    # Component scores
    if report.health_components:
        lines.append("  Component Scores:")
        for comp, score_val in report.health_components.items():
            bar = _progress_bar(score_val, 100, 20)
            lines.append(f"    {comp:20s} [{bar}] {score_val:.0f}")
        lines.append("")

    # Performance
    lines.append("\u250C" + "\u2500" * w + "\u2510")
    lines.append("\u2502" + "  PERFORMANCE SUMMARY".ljust(w) + "\u2502")
    lines.append("\u251C" + "\u2500" * w + "\u2524")
    lines.append("\u2502" + f"  Total P&L:     ${report.total_pnl:+,.0f} ({report.total_pnl_pct:+.1f}%)".ljust(w) + "\u2502")
    lines.append("\u2502" + f"  Week P&L:      ${report.week_pnl:+,.0f}".ljust(w) + "\u2502")
    lines.append("\u2502" + f"  Month P&L:     ${report.month_pnl:+,.0f}".ljust(w) + "\u2502")
    lines.append("\u2502" + f"  Win Rate:      {report.win_rate:.0%} ({report.total_trades} trades)".ljust(w) + "\u2502")
    lines.append("\u2502" + f"  Avg R:         {report.avg_r_multiple:+.2f}R".ljust(w) + "\u2502")
    lines.append("\u2502" + f"  Profit Factor: {report.profit_factor:.2f}".ljust(w) + "\u2502")
    lines.append("\u2502" + f"  Sharpe Ratio:  {report.sharpe_ratio:.2f}".ljust(w) + "\u2502")
    lines.append("\u2502" + f"  Max Drawdown:  {report.max_drawdown_pct:.1f}%".ljust(w) + "\u2502")
    lines.append("\u2514" + "\u2500" * w + "\u2518")
    lines.append("")

    # Forecasts
    lines.append("  FORECASTS (80% confidence interval)")
    for fc_dict, label in [
        (report.forecast_1w, "Next Week   "),
        (report.forecast_1m, "Next Month  "),
        (report.forecast_3m, "Next Quarter"),
        (report.forecast_eoy, "End of Year "),
    ]:
        if fc_dict.get("available"):
            low = fc_dict.get("lower_change_pct", 0)
            high = fc_dict.get("upper_change_pct", 0)
            method = fc_dict.get("method", "?")
            lines.append(f"    {label}: {low:+6.1f}% to {high:+6.1f}%  [{method}]")
    lines.append("")

    # Monte Carlo
    if report.monte_carlo_1m and report.monte_carlo_1m.get("n_simulations", 0) > 0:
        mc = report.monte_carlo_1m
        lines.append(f"  MONTE CARLO (10,000 simulations)")
        lines.append(f"    1-Month range (10th-90th):  {mc['percentile_10']*100:+.1f}% to {mc['percentile_90']*100:+.1f}%")
        lines.append(f"    Prob positive return:        {mc['prob_positive']:.0%}")
        lines.append(f"    Prob > 1% return:            {mc['prob_above_1pct']:.0%}")
        lines.append(f"    Prob > 5% loss:              {mc['prob_below_neg5pct']:.0%}")
        lines.append("")

    # Benchmark
    bm = report.benchmarks
    if bm and bm.get("verdict"):
        worth = "[WORTH IT]" if bm.get("worth_it") else "[NOT WORTH IT]"
        lines.append(f"  VS PASSIVE BENCHMARK {worth}")
        primary = "SPY" if report.system == "alpaca" else "BTC"
        secondary = "QQQ" if report.system == "alpaca" else "GLD"
        our_ret = bm.get("portfolio", {})
        primary_ret = bm.get(primary, {})
        secondary_ret = bm.get(secondary, {})
        for p_label in ["1w", "1m", "ytd"]:
            ours = our_ret.get(p_label)
            theirs = primary_ret.get(p_label)
            theirs2 = secondary_ret.get(p_label)
            if ours is not None and theirs is not None:
                alpha = ours - theirs
                mark = "+" if alpha > 0 else "-"
                line = f"    {p_label.upper():4s} Us: {ours:+.1f}%  {primary}: {theirs:+.1f}%"
                if theirs2 is not None:
                    line += f"  {secondary}: {theirs2:+.1f}%"
                line += f"  alpha: {alpha:+.1f}%"
                lines.append(line)
        for vline in _wrap(bm["verdict"], w - 4):
            lines.append(f"    {vline}")
        lines.append("")

    # Weaknesses
    if report.weaknesses:
        lines.append("  WEAKNESSES")
        for i, wk in enumerate(report.weaknesses[:7], 1):
            sev = "[HIGH]" if wk["severity"] == "high" else "[MED] "
            metric_label = "WR" if wk["metric"] == "win_rate" else "PnL"
            lines.append(
                f"    {sev} {wk['dimension']}={wk['value']}: "
                f"{metric_label}={wk['this_value']} "
                f"(avg: {wk['portfolio_avg']}, n={wk['trade_count']})"
            )
        lines.append("")

    # Suggestions
    if report.suggestions:
        lines.append("  SUGGESTIONS")
        for i, s in enumerate(report.suggestions[:5], 1):
            # Word wrap at 56 chars
            wrapped = _wrap(s, w - 6)
            lines.append(f"    {i}. {wrapped[0]}")
            for wl in wrapped[1:]:
                lines.append(f"       {wl}")
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════

def _grade_emoji(grade: str) -> str:
    return {
        "Excellent": "\U0001F7E2",   # green circle
        "Good": "\U0001F7E2",
        "Fair": "\U0001F7E1",        # yellow circle
        "Poor": "\U0001F7E0",        # orange circle
        "Critical": "\U0001F534",    # red circle
        "No Data": "\u26AA",         # white circle
    }.get(grade, "\u26AA")


def _progress_bar(value: float, max_val: float, width: int = 20) -> str:
    """Generate a text progress bar."""
    filled = int(value / max_val * width) if max_val > 0 else 0
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _wrap(text: str, width: int) -> list[str]:
    """Simple word wrap."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]
