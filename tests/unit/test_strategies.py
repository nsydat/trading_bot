"""
Unit Tests for Trading Strategies
================================

Comprehensive test suite for trading strategies including:
- Strategy initialization and configuration
- Signal generation accuracy
- Backtesting functionality
- Edge cases and error handling
- Strategy performance metrics

Author: dat-ns
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import asyncio
from datetime import datetime, timedelta

# Import strategy classes
from strategies.base_strategy import BaseStrategy, StrategyType, Signal, SignalType
from strategies.trend_following.ema_crossover import EMACrossoverStrategy
from strategies import get_strategy, list_available_strategies, STRATEGY_REGISTRY

# Import supporting classes
from core.models.market_data import OHLCV, PriceData
from indicators.base_indicator import IndicatorConfig


class TestDataGenerator:
    """Generate test market data for strategy testing."""
    
    @staticmethod
    def create_trending_market(length: int = 100, 
                             trend_strength: float = 0.02,
                             base_price: float = 100.0) -> List[OHLCV]:
        """
        Create market data with strong trend for testing trend-following strategies.
        
        Args:
            length: Number of candles
            trend_strength: Trend strength (positive for uptrend)
            base_price: Starting price
            
        Returns:
            List[OHLCV]: Market data with trend
        """
        np.random.seed(42)  # Reproducible results
        data = []
        price = base_price
        
        for i in range(length):
            # Add trend component
            price += trend_strength * price
            # Add some noise
            price += np.random.normal(0, 0.005 * price)
            
            # Generate OHLC
            high = price * (1 + abs(np.random.normal(0, 0.002)))
            low = price * (1 - abs(np.random.normal(0, 0.002)))
            open_price = price + np.random.normal(0, 0.001 * price)
            volume = np.random.uniform(1000, 5000)
            
            data.append(OHLCV(
                timestamp=1640995200 + i * 300,
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high, 2))),
                low=Decimal(str(round(low, 2))),
                close=Decimal(str(round(price, 2))),
                volume=Decimal(str(round(volume, 2)))
            ))
        
        return data
    
    @staticmethod
    def create_sideways_market(length: int = 100,
                              base_price: float = 100.0,
                              volatility: float = 0.01) -> List[OHLCV]:
        """Create sideways/ranging market data."""
        np.random.seed(42)
        data = []
        
        for i in range(length):
            # Price oscillates around base_price
            price = base_price + base_price * volatility * np.sin(i * 0.1) + \
                   np.random.normal(0, volatility * base_price * 0.5)
            
            high = price * (1 + abs(np.random.normal(0, 0.001)))
            low = price * (1 - abs(np.random.normal(0, 0.001)))
            open_price = price + np.random.normal(0, 0.0005 * price)
            volume = np.random.uniform(1000, 5000)
            
            data.append(OHLCV(
                timestamp=1640995200 + i * 300,
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high, 2))),
                low=Decimal(str(round(low, 2))),
                close=Decimal(str(round(price, 2))),
                volume=Decimal(str(round(volume, 2)))
            ))
        
        return data
    
    @staticmethod
    def create_volatile_market(length: int = 100,
                              base_price: float = 100.0,
                              volatility: float = 0.05) -> List[OHLCV]:
        """Create highly volatile market data."""
        np.random.seed(42)
        data = []
        price = base_price
        
        for i in range(length):
            # High volatility random walk
            price *= (1 + np.random.normal(0, volatility))
            price = max(price, base_price * 0.5)  # Floor price
            
            high = price * (1 + abs(np.random.normal(0, volatility * 0.3)))
            low = price * (1 - abs(np.random.normal(0, volatility * 0.3)))
            open_price = price + np.random.normal(0, volatility * 0.2 * price)
            volume = np.random.uniform(5000, 20000)  # Higher volume in volatile markets
            
            data.append(OHLCV(
                timestamp=1640995200 + i * 300,
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high, 2))),
                low=Decimal(str(round(low, 2))),
                close=Decimal(str(round(price, 2))),
                volume=Decimal(str(round(volume, 2)))
            ))
        
        return data


class MockIndicator:
    """Mock indicator for strategy testing."""
    
    def __init__(self, values: List[float]):
        self.values = values
        self.current_index = 0
        self.name = "MockIndicator"
    
    def update(self, candle: OHLCV):
        """Mock update method."""
        pass
    
    def is_ready(self) -> bool:
        return self.current_index < len(self.values)
    
    @property
    def current_value(self) -> Decimal:
        if self.is_ready():
            value = self.values[min(self.current_index, len(self.values) - 1)]
            return Decimal(str(value))
        return None
    
    def get_value(self, index: int = 0) -> Decimal:
        """Get value at specific index (0 = current, 1 = previous, etc.)"""
        idx = max(0, self.current_index - index)
        return Decimal(str(self.values[idx])) if idx < len(self.values) else None
    
    def advance(self):
        """Advance to next value (for testing)."""
        self.current_index += 1


class TestBaseStrategy:
    """Test cases for BaseStrategy abstract class."""
    
    def test_base_strategy_interface(self):
        """Test BaseStrategy interface requirements."""
        # BaseStrategy should not be instantiable directly
        with pytest.raises(TypeError):
            BaseStrategy("test_config")
    
    def test_signal_types(self):
        """Test Signal and SignalType enums."""
        # Test SignalType enum
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.HOLD.value == "HOLD"
        
        # Test Signal creation
        signal = Signal(
            type=SignalType.BUY,
            strength=0.8,
            price=Decimal("100.50"),
            timestamp=1640995200
        )
        
        assert signal.type == SignalType.BUY
        assert signal.strength == 0.8
        assert signal.price == Decimal("100.50")
        assert signal.timestamp == 1640995200
    
    def test_strategy_type_enum(self):
        """Test StrategyType enum."""
        assert StrategyType.TREND_FOLLOWING.value == "trend_following"
        assert StrategyType.MEAN_REVERSION.value == "mean_reversion"
        assert StrategyType.MOMENTUM.value == "momentum"
        assert StrategyType.ARBITRAGE.value == "arbitrage"


class TestEMACrossoverStrategy:
    """Test cases for EMA Crossover Strategy."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'fast_period': 12,
            'slow_period': 26,
            'symbol': 'BTCUSDT',
            'timeframe': '5m'
        }
        self.strategy = EMACrossoverStrategy(self.config)
    
    def test_ema_crossover_initialization(self):
        """Test EMA crossover strategy initialization."""
        assert self.strategy.name == "EMA Crossover"
        assert self.strategy.strategy_type == StrategyType.TREND_FOLLOWING
        assert self.strategy.fast_period == 12
        assert self.strategy.slow_period == 26
        assert not self.strategy.is_ready()
    
    def test_ema_crossover_signal_generation(self):
        """Test EMA crossover signal generation."""
        # Create trending data that should generate signals
        trending_data = TestDataGenerator.create_trending_market(50, 0.01)
        
        signals = []
        for candle in trending_data:
            signal = self.strategy.analyze(candle)
            if signal and signal.type != SignalType.HOLD:
                signals.append(signal)
        
        # Should have generated some signals in trending market
        assert len(signals) > 0
        
        # In uptrending market, first signal should likely be BUY
        if signals:
            # Not all first signals will be BUY due to randomness, but most should be
            buy_signals = [s for s in signals if s.type == SignalType.BUY]
            assert len(buy_signals) > 0
    
    def test_ema_crossover_bullish_crossover(self):
        """Test detection of bullish crossover (fast EMA crosses above slow EMA)."""
        # Create data where fast EMA will cross above slow EMA
        # Start with downtrend, then strong uptrend
        data = []
        price = 100.0
        
        # Downtrend for 30 periods
        for i in range(30):
            price *= 0.99
            data.append(OHLCV(
                timestamp=1640995200 + i * 300,
                open=Decimal(str(price * 1.001)),
                high=Decimal(str(price * 1.002)),
                low=Decimal(str(price * 0.999)),
                close=Decimal(str(price)),
                volume=Decimal("1000.0")
            ))
        
        # Strong uptrend for 20 periods
        for i in range(30, 50):
            price *= 1.02  # 2% increase per period
            data.append(OHLCV(
                timestamp=1640995200 + i * 300,
                open=Decimal(str(price * 0.999)),
                high=Decimal(str(price * 1.001)),
                low=Decimal(str(price * 0.998)),
                close=Decimal(str(price)),
                volume=Decimal("1000.0")
            ))
        
        # Process data and look for crossover
        signals = []
        for candle in data:
            signal = self.strategy.analyze(candle)
            if signal and signal.type == SignalType.BUY:
                signals.append(signal)
        
        # Should detect bullish crossover
        assert len(signals) > 0
    
    def test_ema_crossover_bearish_crossover(self):
        """Test detection of bearish crossover (fast EMA crosses below slow EMA)."""
        # Create opposite scenario - uptrend then downtrend
        data = []
        price = 100.0
        
        # Uptrend for 30 periods
        for i in range(30):
            price *= 1.01
            data.append(OHLCV(
                timestamp=1640995200 + i * 300,
                open=Decimal(str(price * 0.999)),
                high=Decimal(str(price * 1.001)),
                low=Decimal(str(price * 0.998)),
                close=Decimal(str(price)),
                volume=Decimal("1000.0")
            ))
        
        # Strong downtrend for 20 periods
        for i in range(30, 50):
            price *= 0.98  # 2% decrease per period
            data.append(OHLCV(
                timestamp=1640995200 + i * 300,
                open=Decimal(str(price * 1.001)),
                high=Decimal(str(price * 1.002)),
                low=Decimal(str(price * 0.999)),
                close=Decimal(str(price)),
                volume=Decimal("1000.0")
            ))
        
        # Process data and look for crossover
        signals = []
        for candle in data:
            signal = self.strategy.analyze(candle)
            if signal and signal.type == SignalType.SELL:
                signals.append(signal)
        
        # Should detect bearish crossover
        assert len(signals) > 0
    
    def test_ema_crossover_no_signals_sideways(self):
        """Test that few signals are generated in sideways market."""
        sideways_data = TestDataGenerator.create_sideways_market(100)
        
        signals = []
        for candle in sideways_data:
            signal = self.strategy.analyze(candle)
            if signal and signal.type != SignalType.HOLD:
                signals.append(signal)
        
        # Sideways market should generate fewer signals
        # Allow some signals due to noise, but should be limited
        assert len(signals) < 10  # Reasonable threshold for 100 candles
    
    def test_ema_crossover_signal_strength(self):
        """Test signal strength calculation."""
        trending_data = TestDataGenerator.create_trending_market(50, 0.02)
        
        for candle in trending_data:
            signal = self.strategy.analyze(candle)
            if signal and signal.type != SignalType.HOLD:
                # Signal strength should be between 0 and 1
                assert 0.0 <= signal.strength <= 1.0
                # In strong trend, signal strength should be meaningful
                assert signal.strength > 0.1  # Not too weak
    
    def test_ema_crossover_insufficient_data(self):
        """Test strategy behavior with insufficient data."""
        # Add less data than required for slow EMA
        short_data = TestDataGenerator.create_trending_market(10)
        
        signals = []
        for candle in short_data:
            signal = self.strategy.analyze(candle)
            if signal:
                signals.append(signal)
        
        # Should not generate trading signals with insufficient data
        # May generate HOLD signals
        trading_signals = [s for s in signals if s.type != SignalType.HOLD]
        assert len(trading_signals) == 0
    
    def test_ema_crossover_reset(self):
        """Test strategy reset functionality."""
        # Add some data
        data = TestDataGenerator.create_trending_market(30)
        for candle in data[:20]:
            self.strategy.analyze(candle)
        
        # Strategy should be ready
        assert self.strategy.is_ready()
        
        # Reset strategy
        self.strategy.reset()
        
        # Should not be ready after reset
        assert not self.strategy.is_ready()
    
    def test_ema_crossover_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid fast/slow period combination
        invalid_config = {
            'fast_period': 26,  # Fast period >= slow period
            'slow_period': 12,
            'symbol': 'BTCUSDT',
            'timeframe': '5m'
        }
        
        with pytest.raises(ValueError):
            EMACrossoverStrategy(invalid_config)
        
        # Test missing required parameters
        incomplete_config = {
            'fast_period': 12
            # Missing slow_period
        }
        
        with pytest.raises((ValueError, KeyError)):
            EMACrossoverStrategy(incomplete_config)


class TestStrategyFactory:
    """Test strategy factory functions."""
    
    def test_get_strategy_valid(self):
        """Test getting valid strategy from factory."""
        strategy = get_strategy('ema_crossover', 
                               fast_period=12, 
                               slow_period=26,
                               symbol='BTCUSDT',
                               timeframe='5m')
        
        assert isinstance(strategy, EMACrossoverStrategy)
        assert strategy.name == "EMA Crossover"
    
    def test_get_strategy_invalid(self):
        """Test getting invalid strategy from factory."""
        with pytest.raises(ValueError) as exc_info:
            get_strategy('nonexistent_strategy')
        
        assert "not found" in str(exc_info.value)
        assert "Available strategies" in str(exc_info.value)
    
    def test_list_available_strategies(self):
        """Test listing available strategies."""
        strategies = list_available_strategies()
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert 'ema_crossover' in strategies
    
    def test_strategy_registry(self):
        """Test strategy registry."""
        assert 'ema_crossover' in STRATEGY_REGISTRY
        assert STRATEGY_REGISTRY['ema_crossover'] == EMACrossoverStrategy


class TestStrategyPerformance:
    """Performance tests for strategies."""
    
    def test_strategy_update_performance(self):
        """Test strategy performance with large datasets."""
        import time
        
        config = {
            'fast_period': 12,
            'slow_period': 26,
            'symbol': 'BTCUSDT',
            'timeframe': '5m'
        }
        strategy = EMACrossoverStrategy(config)
        
        # Generate large dataset
        large_data = TestDataGenerator.create_trending_market(5000)
        
        start_time = time.time()
        
        for candle in large_data:
            signal = strategy.analyze(candle)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 5000 candles quickly (< 2 seconds)
        assert processing_time < 2.0
        
        # Calculate analysis rate
        rate = len(large_data) / processing_time
        print(f"Strategy analysis rate: {rate:.0f} candles/second")
        
        # Should achieve reasonable throughput
        assert rate > 1000  # At least 1000 candles/second


class TestStrategyBacktesting:
    """Test strategy backtesting functionality."""
    
    def setup_method(self):
        """Setup backtesting environment."""
        self.config = {
            'fast_period': 12,
            'slow_period': 26,
            'symbol': 'BTCUSDT',
            'timeframe': '5m'
        }
        self.strategy = EMACrossoverStrategy(self.config)
    
    def test_simple_backtest(self):
        """Test simple backtesting functionality."""
        # Generate test data with known trend
        test_data = TestDataGenerator.create_trending_market(100, 0.015)  # 1.5% trend
        
        signals = []
        for candle in test_data:
            signal = self.strategy.analyze(candle)
            if signal and signal.type != SignalType.HOLD:
                signals.append((candle.timestamp, signal))
        
        # Basic backtest metrics
        if len(signals) >= 2:
            # Calculate simple return from first buy to last sell (or vice versa)
            first_signal_time, first_signal = signals[0]
            last_signal_time, last_signal = signals[-1]
            
            first_price = next(c.close for c in test_data if c.timestamp == first_signal_time)
            last_price = next(c.close for c in test_data if c.timestamp == last_signal_time)
            
            # In uptrending data, strategy should be profitable
            if first_signal.type == SignalType.BUY and last_signal.type == SignalType.SELL:
                profit_pct = (float(last_price) - float(first_price)) / float(first_price)
                assert profit_pct > 0  # Should be profitable in uptrend
    
    def test_backtest_with_transaction_costs(self):
        """Test backtesting considering transaction costs."""
        test_data = TestDataGenerator.create_trending_market(50, 0.01)
        
        signals = []
        for candle in test_data:
            signal = self.strategy.analyze(candle)
            if signal and signal.type != SignalType.HOLD:
                signals.append(signal)
        
        # Count number of trades (signal changes)
        trade_count = len(signals)
        
        # With transaction costs, frequent trading reduces profitability
        # Strategy should not generate excessive signals
        assert trade_count < len(test_data) * 0.2  # Less than 20% of candles generate signals
    
    def test_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # This would be implemented if strategy includes performance metrics
        test_data = TestDataGenerator.create_volatile_market(100, volatility=0.03)
        
        equity_curve = []
        portfolio_value = 10000.0  # Starting capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0.0
        
        for candle in test_data:
            signal = self.strategy.analyze(candle)
            current_price = float(candle.close)
            
            # Simple position management
            if signal and signal.type == SignalType.BUY and position <= 0:
                if position == -1:  # Close short position
                    profit = (entry_price - current_price) / entry_price
                    portfolio_value *= (1 + profit)
                position = 1
                entry_price = current_price
            elif signal and signal.type == SignalType.SELL and position >= 0:
                if position == 1:  # Close long position
                    profit = (current_price - entry_price) / entry_price
                    portfolio_value *= (1 + profit)
                position = -1
                entry_price = current_price
            
            equity_curve.append(portfolio_value)
        
        # Calculate maximum drawdown
        if equity_curve:
            peak = equity_curve[0]
            max_drawdown = 0.0
            
            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Drawdown should be reasonable (< 50% for this test)
            assert max_drawdown < 0.5


class TestStrategyRiskManagement:
    """Test strategy risk management integration."""
    
    def setup_method(self):
        """Setup risk management testing."""
        self.config = {
            'fast_period': 12,
            'slow_period': 26,
            'symbol': 'BTCUSDT',
            'timeframe': '5m',
            'risk_per_trade': 0.02,  # 2% risk per trade
            'max_drawdown': 0.10     # 10% max drawdown
        }
        self.strategy = EMACrossoverStrategy(self.config)
    
    def test_position_sizing_integration(self):
        """Test position sizing integration with signals."""
        if hasattr(self.strategy, 'calculate_position_size'):
            signal = Signal(
                type=SignalType.BUY,
                strength=0.8,
                price=Decimal("100.00"),
                timestamp=1640995200
            )
            
            account_balance = 10000.0
            stop_loss_distance = 2.0  # $2 stop loss
            
            position_size = self.strategy.calculate_position_size(
                signal, account_balance, stop_loss_distance
            )
            
            # Position size should be reasonable
            assert position_size > 0
            # Risk should not exceed configured limit
            risk_amount = position_size * stop_loss_distance
            max_risk = account_balance * 0.02
            assert risk_amount <= max_risk * 1.01  # Small tolerance for rounding


class TestStrategyOptimization:
    """Test strategy parameter optimization."""
    
    def test_parameter_optimization_grid_search(self):
        """Test grid search parameter optimization."""
        # Test different parameter combinations
        test_data = TestDataGenerator.create_trending_market(200, 0.01)
        
        best_config = None
        best_performance = -float('inf')
        
        # Grid search over different EMA periods
        for fast_period in [8, 12, 16]:
            for slow_period in [21, 26, 30]:
                if fast_period >= slow_period:
                    continue
                
                config = {
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'symbol': 'BTCUSDT',
                    'timeframe': '5m'
                }
                
                strategy = EMACrossoverStrategy(config)
                
                # Simple performance metric: total return
                signals = []
                for candle in test_data:
                    signal = strategy.analyze(candle)
                    if signal and signal.type != SignalType.HOLD:
                        signals.append((candle.close, signal))
                
                # Calculate performance
                if len(signals) >= 2:
                    returns = []
                    for i in range(1, len(signals)):
                        prev_price, prev_signal = signals[i-1]
                        curr_price, curr_signal = signals[i]
                        
                        if prev_signal.type == SignalType.BUY:
                            ret = (float(curr_price) - float(prev_price)) / float(prev_price)
                        else:
                            ret = (float(prev_price) - float(curr_price)) / float(prev_price)
                        
                        returns.append(ret)
                    
                    total_return = sum(returns)
                    if total_return > best_performance:
                        best_performance = total_return
                        best_config = config
        
        # Should find some configuration
        assert best_config is not None
        assert best_performance != -float('inf')


# Pytest fixtures for strategy testing
@pytest.fixture
def trending_up_data():
    """Fixture for uptrending market data."""
    return TestDataGenerator.create_trending_market(50, 0.015)

@pytest.fixture  
def trending_down_data():
    """Fixture for downtrending market data."""
    return TestDataGenerator.create_trending_market(50, -0.015)

@pytest.fixture
def sideways_data():
    """Fixture for sideways market data."""
    return TestDataGenerator.create_sideways_market(50)

@pytest.fixture
def volatile_data():
    """Fixture for volatile market data."""
    return TestDataGenerator.create_volatile_market(50)

@pytest.fixture
def ema_strategy():
    """Fixture for EMA crossover strategy."""
    config = {
        'fast_period': 12,
        'slow_period': 26,
        'symbol': 'BTCUSDT',
        'timeframe': '5m'
    }
    return EMACrossoverStrategy(config)


# Parametrized tests
@pytest.mark.parametrize("fast,slow", [(8, 21), (12, 26), (16, 30)])
def test_ema_strategy_different_periods(fast, slow):
    """Test EMA strategy with different period combinations."""
    config = {
        'fast_period': fast,
        'slow_period': slow,
        'symbol': 'BTCUSDT',
        'timeframe': '5m'
    }
    
    strategy = EMACrossoverStrategy(config)
    assert strategy.fast_period == fast
    assert strategy.slow_period == slow

@pytest.mark.parametrize("trend_strength", [0.005, 0.01, 0.02, 0.03])
def test_strategy_performance_different_trends(trend_strength):
    """Test strategy performance across different trend strengths."""
    config = {
        'fast_period': 12,
        'slow_period': 26,
        'symbol': 'BTCUSDT',
        'timeframe': '5m'
    }
    
    strategy = EMACrossoverStrategy(config)
    test_data = TestDataGenerator.create_trending_market(100, trend_strength)
    
    signals = []
    for candle in test_data:
        signal = strategy.analyze(candle)
        if signal and signal.type != SignalType.HOLD:
            signals.append(signal)
    
    # Stronger trends should generate more confident signals
    if signals:
        avg_strength = sum(s.strength for s in signals) / len(signals)
        # This is a general expectation - stronger trends might yield higher strength
        assert avg_strength > 0.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])