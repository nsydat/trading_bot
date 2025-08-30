"""
Exchange Integration Module
===========================

This module provides exchange integration functionality for the trading bot.
It follows the abstract base class pattern to support multiple exchanges.

Key Components:
- BaseExchange: Abstract interface for all exchange implementations
- BinanceExchange: Binance-specific implementation
- Exchange utilities and helpers

Usage:
    from core.exchange import BinanceExchange
    from config.settings import get_settings
    
    settings = get_settings()
    exchange = BinanceExchange(settings)
    await exchange.initialize()
"""

from .base_exchange import BaseExchange
from .binance_exchange import BinanceExchange

__all__ = [
    'BaseExchange',
    'BinanceExchange'
]