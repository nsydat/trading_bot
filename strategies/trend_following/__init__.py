"""
Trend Following Strategies
==========================

This module contains trend-following trading strategies that aim to profit
from sustained price movements in a particular direction.

Trend following strategies typically:
- Identify the direction of the prevailing trend
- Enter positions in the direction of the trend
- Use momentum indicators and moving averages
- Cut losses quickly and let profits run

Available Strategies:
- EMA Crossover: Uses exponential moving average crossovers to identify trends
- MACD Strategy: Uses MACD indicator for trend detection (to be implemented)
- Momentum Strategy: Uses momentum oscillators (to be implemented)

Author: dat-ns
Version: 1.0.0
"""

import logging
from typing import List, Dict, Type

from ..base_strategy import BaseStrategy
from .ema_crossover import EMACrossoverStrategy

# Logger for this module
logger = logging.getLogger(__name__)

# Registry of trend following strategies
TREND_FOLLOWING_STRATEGIES: Dict[str, Type[BaseStrategy]] = {
    'ema_crossover': EMACrossoverStrategy,
    # More strategies will be added here:
    # 'macd': MACDStrategy,
    # 'momentum': MomentumStrategy,
    # 'breakout': BreakoutStrategy,
}


def get_trend_following_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
    """
    Factory function to create trend following strategy instances.
    
    Args:
        strategy_name (str): Name of the strategy to create
        **kwargs: Additional arguments to pass to strategy constructor
        
    Returns:
        BaseStrategy: Instance of the requested strategy
        
    Raises:
        ValueError: If strategy name is not found
    """
    if strategy_name not in TREND_FOLLOWING_STRATEGIES:
        available = list(TREND_FOLLOWING_STRATEGIES.keys())
        raise ValueError(
            f"Trend following strategy '{strategy_name}' not found. "
            f"Available strategies: {available}"
        )
    
    strategy_class = TREND_FOLLOWING_STRATEGIES[strategy_name]
    logger.info(f"📈 Creating trend following strategy: {strategy_name}")
    
    return strategy_class(**kwargs)


def list_trend_following_strategies() -> List[str]:
    """
    Get list of all available trend following strategy names.
    
    Returns:
        List[str]: List of strategy names
    """
    return list(TREND_FOLLOWING_STRATEGIES.keys())


def get_strategy_description(strategy_name: str) -> str:
    """
    Get description of a specific trend following strategy.
    
    Args:
        strategy_name (str): Name of the strategy
        
    Returns:
        str: Strategy description
        
    Raises:
        ValueError: If strategy name is not found
    """
    if strategy_name not in TREND_FOLLOWING_STRATEGIES:
        raise ValueError(f"Strategy '{strategy_name}' not found")
    
    strategy_class = TREND_FOLLOWING_STRATEGIES[strategy_name]
    return strategy_class.description or "No description available"


# Export main classes and functions
__all__ = [
    'EMACrossoverStrategy',
    'TREND_FOLLOWING_STRATEGIES',
    'get_trend_following_strategy',
    'list_trend_following_strategies',
    'get_strategy_description'
]

# Log available strategies
logger.info(f"📈 Loaded {len(TREND_FOLLOWING_STRATEGIES)} trend following strategies: {list(TREND_FOLLOWING_STRATEGIES.keys())}")