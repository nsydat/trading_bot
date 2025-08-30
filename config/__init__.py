"""
Configuration Module
===================

This module handles all configuration-related functionality for the trading bot.
"""

from .settings import Settings, get_settings

__all__ = [
    'Settings',
    'get_settings'
]