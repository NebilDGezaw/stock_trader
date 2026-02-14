"""
ML Portfolio Analytics Module
==============================
Sophisticated machine learning system for portfolio health monitoring,
performance forecasting, weakness detection, and code-change impact analysis.

Separate pipelines for:
  - Alpaca (US stocks & leveraged ETFs)
  - HFM (Forex, Crypto, Commodities via MT5)

Components:
  data_collector  — Gathers data from APIs, git, logs, market
  features        — Feature engineering pipeline
  models          — Ensemble ML models (RF, XGBoost, ARIMA, Monte Carlo)
  analyzer        — Orchestrates health scoring, forecasting, weakness detection
  report          — Generates Telegram-ready reports
  run_ml          — CLI entry point
"""
