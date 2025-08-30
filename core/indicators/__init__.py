"""
Technical Indicators Module
===========================

This module provides technical analysis indicators for trading strategies.
All indicators are optimized for performance and include comprehensive error handling.

Available Indicators:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)  
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

Usage:
    from core.indicators import TechnicalIndicators
    
    indicators = TechnicalIndicators()
    sma = indicators.sma(data['close'], period=20)
    rsi = indicators.rsi(data['close'], period=14)
"""

from .technical_indicators import TechnicalIndicators

# Available indicator functions
__all__ = [
    'TechnicalIndicators'
]

# Version info
__version__ = "1.0.0"
__author__ = "dat-ns"

# Quick access to commonly used indicators
def sma(data, period=20):
    """Quick access to Simple Moving Average."""
    return TechnicalIndicators.sma(data, period)

def ema(data, period=20):
    """Quick access to Exponential Moving Average."""
    return TechnicalIndicators.ema(data, period)

def rsi(data, period=14):
    """Quick access to Relative Strength Index."""
    return TechnicalIndicators.rsi(data, period)

def macd(data, fast=12, slow=26, signal=9):
    """Quick access to MACD indicator."""
    return TechnicalIndicators.macd(data, fast, slow, signal)

def bollinger_bands(data, period=20, std_dev=2):
    """Quick access to Bollinger Bands."""
    return TechnicalIndicators.bollinger_bands(data, period, std_dev)