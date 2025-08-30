"""
Data Handling Module
===================

This module provides comprehensive data handling capabilities for the trading bot.
It includes data fetching, processing, validation, and caching functionality.

Key Components:
- DataFetcher: Fetches market data from exchanges with proper rate limiting
- DataProcessor: Processes and cleans raw market data
- Data validation and error handling utilities
- Caching mechanisms for improved performance

Usage:
    from core.data import DataFetcher, DataProcessor
    from config.settings import get_settings
    
    settings = get_settings()
    fetcher = DataFetcher(settings)
    processor = DataProcessor()
    
    # Fetch and process data
    raw_data = await fetcher.fetch_ohlcv('BTCUSDT', '1h', limit=100)
    processed_data = await processor.process_ohlcv(raw_data)
"""

from .data_fetcher import DataFetcher
from .data_processor import DataProcessor

__all__ = [
    'DataFetcher',
    'DataProcessor'
]

__version__ = "1.0.0"
__author__ = "dat-ns"