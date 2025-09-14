"""
Risk Management Module
======================

Comprehensive risk management system for the trading bot.
Handles position sizing, risk calculations, and risk metrics.

This module provides:
- Position sizing based on account risk
- Risk-reward ratio calculations
- Stop loss and take profit calculations
- Risk metrics and validation
- Support for different asset types and leverage

Author: dat-ns
Version: 1.0.0
"""

from .risk_calculator import (
    RiskCalculator,
    RiskMetrics,
    RiskParameters,
    PositionRisk,
    TradeRisk
)
from .position_sizing import (
    PositionSizer,
    PositionSize,
    SizingMethod,
    AccountInfo
)

__all__ = [
    # Risk Calculator
    'RiskCalculator',
    'RiskMetrics',
    'RiskParameters', 
    'PositionRisk',
    'TradeRisk',
    
    # Position Sizing
    'PositionSizer',
    'PositionSize',
    'SizingMethod',
    'AccountInfo'
]

__version__ = "1.0.0"
__author__ = "dat-ns"