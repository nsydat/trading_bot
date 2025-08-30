"""
Trading Strategies Module
========================

This module contains all trading strategies for the automated trading bot.
Strategies are organized by category and implement a common base interface.

Strategy Categories:
- Trend Following: EMA crossover, MACD, etc.
- Mean Reversion: RSI oversold/overbought, Bollinger Bands, etc.
- Multi-timeframe: Strategies that analyze multiple timeframes
- ML-based: Machine learning powered strategies

Author: dat-ns
Version: 1.0.0
"""

from typing import Dict, Type, List
import logging

from .base_strategy import BaseStrategy, StrategyType, Signal
from .trend_following.ema_crossover import EMACrossoverStrategy

# Strategy registry for dynamic loading
STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    'ema_crossover': EMACrossoverStrategy,
    # More strategies will be added here
}

logger = logging.getLogger(__name__)


def get_strategy(strategy_name: str, **kwargs) -> BaseStrategy:
    """
    Factory function to create strategy instances.
    
    Args:
        strategy_name (str): Name of the strategy to create
        **kwargs: Additional arguments to pass to strategy constructor
        
    Returns:
        BaseStrategy: Instance of the requested strategy
        
    Raises:
        ValueError: If strategy name is not found in registry
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available_strategies = list(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Strategy '{strategy_name}' not found. "
            f"Available strategies: {available_strategies}"
        )
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    logger.info(f"📊 Creating strategy instance: {strategy_name}")
    
    return strategy_class(**kwargs)


def list_available_strategies() -> List[str]:
    """
    Get list of all available strategy names.
    
    Returns:
        List[str]: List of strategy names
    """
    return list(STRATEGY_REGISTRY.keys())


def get_strategy_info(strategy_name: str) -> Dict[str, any]:
    """
    Get information about a specific strategy.
    
    Args:
        strategy_name (str): Name of the strategy
        
    Returns:
        Dict[str, any]: Strategy information including type, description, etc.
        
    Raises:
        ValueError: If strategy name is not found
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{strategy_name}' not found")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    
    return {
        'name': strategy_name,
        'class_name': strategy_class.__name__,
        'type': strategy_class.strategy_type.value if hasattr(strategy_class, 'strategy_type') else 'unknown',
        'description': strategy_class.__doc__ or 'No description available',
        'module': strategy_class.__module__
    }


def register_strategy(name: str, strategy_class: Type[BaseStrategy]):
    """
    Register a new strategy in the registry.
    
    Args:
        name (str): Name to register the strategy under
        strategy_class (Type[BaseStrategy]): Strategy class to register
    """
    if not issubclass(strategy_class, BaseStrategy):
        raise ValueError("Strategy class must inherit from BaseStrategy")
    
    STRATEGY_REGISTRY[name] = strategy_class
    logger.info(f"📊 Registered strategy: {name}")


# Export main classes and functions
__all__ = [
    'BaseStrategy',
    'StrategyType', 
    'Signal',
    'EMACrossoverStrategy',
    'get_strategy',
    'list_available_strategies',
    'get_strategy_info',
    'register_strategy',
    'STRATEGY_REGISTRY'
]