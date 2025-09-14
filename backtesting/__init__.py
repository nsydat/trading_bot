"""
Backtesting Module
==================

Complete standalone backtesting system for trading strategies.
Works offline with historical data, no API keys required.

Features:
- Historical data backtesting
- Performance metrics calculation
- Risk analysis and trade statistics
- Sample data included for immediate testing
- Support for long/short positions
- Commission and slippage simulation

Author: Trading Bot System
Version: 1.0.0
"""

from .backtest_engine import BacktestEngine, BacktestResult, BacktestConfig
from .performance_metrics import PerformanceCalculator, PerformanceReport
from .data_loader import DataLoader

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'BacktestConfig',
    'PerformanceCalculator',
    'PerformanceReport',
    'DataLoader'
]

__version__ = "1.0.0"