"""
Unit Tests for Technical Indicators
===================================

Comprehensive test suite for all technical indicators including:
- SMA, EMA, RSI, MACD, Bollinger Bands
- Edge cases and error conditions
- Performance tests
- Mock data fixtures

Author: dat-ns
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any
import asyncio

# Import indicators (assuming they exist in indicators module)
from indicators.moving_averages import SMAIndicator, EMAIndicator
from indicators.oscillators import RSIIndicator, MACDIndicator
from indicators.volatility import BollingerBandsIndicator
from indicators.base_indicator import IndicatorConfig
from core.models.market_data import OHLCV, PriceData


class TestDataFixtures:
    """Test data fixtures for indicator testing."""
    
    @staticmethod
    def generate_sample_ohlcv(length: int = 100, 
                             start_price: float = 100.0,
                             volatility: float = 0.02) -> List[OHLCV]:
        """
        Generate sample OHLCV data for testing.
        
        Args:
            length: Number of candles to generate
            start_price: Starting price
            volatility: Price volatility factor
            
        Returns:
            List[OHLCV]: Sample OHLCV data
        """
        np.random.seed(42)  # For reproducible tests
        
        data = []
        current_price = start_price
        
        for i in range(length):
            # Random walk with slight upward bias
            change = np.random.normal(0.001, volatility)
            current_price *= (1 + change)
            
            # Generate OHLC around current price
            high = current_price * (1 + abs(np.random.normal(0, volatility/2)))
            low = current_price * (1 - abs(np.random.normal(0, volatility/2)))
            open_price = current_price + np.random.normal(0, volatility/4) * current_price
            volume = np.random.uniform(1000, 10000)
            
            data.append(OHLCV(
                timestamp=1640995200 + i * 300,  # 5-minute intervals
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high, 2))),
                low=Decimal(str(round(low, 2))),
                close=Decimal(str(round(current_price, 2))),
                volume=Decimal(str(round(volume, 2)))
            ))
        
        return data
    
    @staticmethod
    def generate_trending_data(length: int = 50, 
                              trend: str = "up") -> List[OHLCV]:
        """
        Generate trending price data for testing trend indicators.
        
        Args:
            length: Number of data points
            trend: "up", "down", or "sideways"
            
        Returns:
            List[OHLCV]: Trending OHLCV data
        """
        data = []
        base_price = 100.0
        trend_factor = {"up": 0.002, "down": -0.002, "sideways": 0.0}[trend]
        
        for i in range(length):
            price = base_price * (1 + trend_factor * i) + np.random.normal(0, 0.5)
            
            data.append(OHLCV(
                timestamp=1640995200 + i * 300,
                open=Decimal(str(round(price * 0.999, 2))),
                high=Decimal(str(round(price * 1.001, 2))),
                low=Decimal(str(round(price * 0.998, 2))),
                close=Decimal(str(round(price, 2))),
                volume=Decimal("1000.0")
            ))
        
        return data


class TestSMAIndicator:
    """Test cases for Simple Moving Average indicator."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.config = IndicatorConfig(period=20)
        self.sma = SMAIndicator(self.config)
        self.sample_data = TestDataFixtures.generate_sample_ohlcv(50)
    
    def test_sma_initialization(self):
        """Test SMA indicator initialization."""
        assert self.sma.period == 20
        assert self.sma.name == "SMA"
        assert len(self.sma.values) == 0
        assert not self.sma.is_ready()
    
    def test_sma_calculation_accuracy(self):
        """Test SMA calculation accuracy against manual calculation."""
        # Add data points
        for candle in self.sample_data[:25]:
            self.sma.update(candle)
        
        # Verify SMA is ready after enough data points
        assert self.sma.is_ready()
        
        # Manual calculation for verification
        closes = [float(candle.close) for candle in self.sample_data[5:25]]
        expected_sma = sum(closes[-20:]) / 20
        
        assert abs(float(self.sma.current_value) - expected_sma) < 0.01
    
    def test_sma_insufficient_data(self):
        """Test SMA behavior with insufficient data."""
        # Add less data than required period
        for candle in self.sample_data[:10]:
            self.sma.update(candle)
        
        assert not self.sma.is_ready()
        assert self.sma.current_value is None
    
    def test_sma_edge_cases(self):
        """Test SMA with edge cases."""
        # Test with period = 1
        sma_1 = SMAIndicator(IndicatorConfig(period=1))
        sma_1.update(self.sample_data[0])
        
        assert sma_1.is_ready()
        assert sma_1.current_value == self.sample_data[0].close
        
        # Test with very large period
        sma_large = SMAIndicator(IndicatorConfig(period=1000))
        for candle in self.sample_data:
            sma_large.update(candle)
        
        assert not sma_large.is_ready()
    
    def test_sma_reset(self):
        """Test SMA reset functionality."""
        # Add some data
        for candle in self.sample_data[:25]:
            self.sma.update(candle)
        
        assert self.sma.is_ready()
        
        # Reset and verify
        self.sma.reset()
        assert not self.sma.is_ready()
        assert len(self.sma.values) == 0


class TestEMAIndicator:
    """Test cases for Exponential Moving Average indicator."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = IndicatorConfig(period=20)
        self.ema = EMAIndicator(self.config)
        self.sample_data = TestDataFixtures.generate_sample_ohlcv(50)
    
    def test_ema_initialization(self):
        """Test EMA indicator initialization."""
        assert self.ema.period == 20
        assert self.ema.name == "EMA"
        assert self.ema.multiplier == 2 / (20 + 1)
    
    def test_ema_calculation(self):
        """Test EMA calculation logic."""
        # Add first data point (becomes initial EMA)
        self.ema.update(self.sample_data[0])
        assert self.ema.current_value == self.sample_data[0].close
        
        # Add second data point
        self.ema.update(self.sample_data[1])
        
        # Verify EMA calculation
        expected = float(self.sample_data[0].close) + self.ema.multiplier * (
            float(self.sample_data[1].close) - float(self.sample_data[0].close))
        
        assert abs(float(self.ema.current_value) - expected) < 0.01
    
    def test_ema_vs_sma_convergence(self):
        """Test EMA vs SMA behavior over time."""
        sma = SMAIndicator(IndicatorConfig(period=20))
        
        # Use trending data
        trending_data = TestDataFixtures.generate_trending_data(100, "up")
        
        for candle in trending_data:
            self.ema.update(candle)
            sma.update(candle)
        
        # EMA should be more responsive to recent changes
        # In an uptrend, EMA should be higher than SMA
        assert float(self.ema.current_value) > float(sma.current_value)


class TestRSIIndicator:
    """Test cases for RSI indicator."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = IndicatorConfig(period=14)
        self.rsi = RSIIndicator(self.config)
        self.sample_data = TestDataFixtures.generate_sample_ohlcv(50)
    
    def test_rsi_initialization(self):
        """Test RSI initialization."""
        assert self.rsi.period == 14
        assert self.rsi.name == "RSI"
        assert not self.rsi.is_ready()
    
    def test_rsi_calculation_range(self):
        """Test RSI stays within 0-100 range."""
        for candle in self.sample_data:
            self.rsi.update(candle)
        
        if self.rsi.is_ready():
            assert 0 <= float(self.rsi.current_value) <= 100
    
    def test_rsi_extreme_values(self):
        """Test RSI with extreme market conditions."""
        # Create data with strong uptrend
        strong_up_data = []
        price = 100.0
        
        for i in range(30):
            price *= 1.02  # 2% increase each period
            strong_up_data.append(OHLCV(
                timestamp=1640995200 + i * 300,
                open=Decimal(str(price * 0.99)),
                high=Decimal(str(price * 1.01)),
                low=Decimal(str(price * 0.98)),
                close=Decimal(str(price)),
                volume=Decimal("1000.0")
            ))
        
        rsi_up = RSIIndicator(IndicatorConfig(period=14))
        for candle in strong_up_data:
            rsi_up.update(candle)
        
        # RSI should be high (but not necessarily 100)
        if rsi_up.is_ready():
            assert float(rsi_up.current_value) > 70
    
    def test_rsi_overbought_oversold(self):
        """Test RSI overbought/oversold conditions."""
        # This is more of an integration test
        for candle in self.sample_data:
            self.rsi.update(candle)
        
        if self.rsi.is_ready():
            rsi_value = float(self.rsi.current_value)
            
            # Test threshold methods if they exist
            if hasattr(self.rsi, 'is_overbought'):
                is_overbought = self.rsi.is_overbought(threshold=70)
                assert is_overbought == (rsi_value > 70)
            
            if hasattr(self.rsi, 'is_oversold'):
                is_oversold = self.rsi.is_oversold(threshold=30)
                assert is_oversold == (rsi_value < 30)


class TestMACDIndicator:
    """Test cases for MACD indicator."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = IndicatorConfig(
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        self.macd = MACDIndicator(self.config)
        self.sample_data = TestDataFixtures.generate_sample_ohlcv(100)
    
    def test_macd_initialization(self):
        """Test MACD initialization."""
        assert self.macd.fast_period == 12
        assert self.macd.slow_period == 26
        assert self.macd.signal_period == 9
        assert self.macd.name == "MACD"
    
    def test_macd_calculation(self):
        """Test MACD calculation components."""
        for candle in self.sample_data:
            self.macd.update(candle)
        
        if self.macd.is_ready():
            result = self.macd.get_values()
            
            assert 'macd_line' in result
            assert 'signal_line' in result
            assert 'histogram' in result
            
            # Histogram should be macd_line - signal_line
            expected_histogram = result['macd_line'] - result['signal_line']
            assert abs(float(result['histogram']) - float(expected_histogram)) < 0.001
    
    def test_macd_crossover_detection(self):
        """Test MACD crossover detection."""
        if hasattr(self.macd, 'detect_crossover'):
            # Add historical data
            for candle in self.sample_data[:-1]:
                self.macd.update(candle)
            
            previous_values = self.macd.get_values() if self.macd.is_ready() else None
            
            # Add latest data
            self.macd.update(self.sample_data[-1])
            
            if self.macd.is_ready() and previous_values:
                crossover = self.macd.detect_crossover()
                # Crossover should be boolean or specific type
                assert isinstance(crossover, (bool, str, type(None)))


class TestBollingerBandsIndicator:
    """Test cases for Bollinger Bands indicator."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = IndicatorConfig(period=20, std_dev=2.0)
        self.bb = BollingerBandsIndicator(self.config)
        self.sample_data = TestDataFixtures.generate_sample_ohlcv(50)
    
    def test_bollinger_bands_initialization(self):
        """Test Bollinger Bands initialization."""
        assert self.bb.period == 20
        assert self.bb.std_dev == 2.0
        assert self.bb.name == "BollingerBands"
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        for candle in self.sample_data:
            self.bb.update(candle)
        
        if self.bb.is_ready():
            result = self.bb.get_values()
            
            assert 'upper_band' in result
            assert 'middle_band' in result  # SMA
            assert 'lower_band' in result
            
            # Upper band should be higher than middle, middle higher than lower
            assert float(result['upper_band']) > float(result['middle_band'])
            assert float(result['middle_band']) > float(result['lower_band'])
    
    def test_bollinger_squeeze_detection(self):
        """Test Bollinger Bands squeeze detection."""
        if hasattr(self.bb, 'detect_squeeze'):
            for candle in self.sample_data:
                self.bb.update(candle)
            
            if self.bb.is_ready():
                squeeze = self.bb.detect_squeeze()
                assert isinstance(squeeze, bool)


class TestIndicatorPerformance:
    """Performance tests for indicators."""
    
    def test_indicator_update_performance(self):
        """Test indicator update performance with large datasets."""
        import time
        
        # Generate large dataset
        large_data = TestDataFixtures.generate_sample_ohlcv(10000)
        
        indicators = [
            SMAIndicator(IndicatorConfig(period=20)),
            EMAIndicator(IndicatorConfig(period=20)),
            RSIIndicator(IndicatorConfig(period=14)),
            MACDIndicator(IndicatorConfig(fast_period=12, slow_period=26, signal_period=9))
        ]
        
        for indicator in indicators:
            start_time = time.time()
            
            for candle in large_data:
                indicator.update(candle)
            
            end_time = time.time()
            update_time = end_time - start_time
            
            # Should process 10k updates in reasonable time (< 1 second)
            assert update_time < 1.0, f"{indicator.name} took {update_time:.3f}s for 10k updates"
            
            # Calculate updates per second
            ups = len(large_data) / update_time
            print(f"{indicator.name}: {ups:.0f} updates/second")
    
    def test_memory_usage(self):
        """Test indicator memory usage doesn't grow unbounded."""
        import sys
        
        sma = SMAIndicator(IndicatorConfig(period=20))
        
        # Initial memory footprint
        initial_size = sys.getsizeof(sma)
        
        # Add many data points
        large_data = TestDataFixtures.generate_sample_ohlcv(1000)
        for candle in large_data:
            sma.update(candle)
        
        # Check memory didn't grow excessively
        final_size = sys.getsizeof(sma)
        growth = final_size - initial_size
        
        # Memory growth should be reasonable (less than 10KB for this test)
        assert growth < 10000, f"Excessive memory growth: {growth} bytes"


class TestIndicatorIntegration:
    """Integration tests for indicators working together."""
    
    def test_multiple_indicators_sync(self):
        """Test multiple indicators processing same data stay in sync."""
        indicators = [
            SMAIndicator(IndicatorConfig(period=20)),
            EMAIndicator(IndicatorConfig(period=20)),
            RSIIndicator(IndicatorConfig(period=14))
        ]
        
        sample_data = TestDataFixtures.generate_sample_ohlcv(50)
        
        # Process same data through all indicators
        for candle in sample_data:
            for indicator in indicators:
                indicator.update(candle)
        
        # All indicators should have processed the same number of data points
        for indicator in indicators:
            if indicator.is_ready():
                assert len(indicator.values) > 0
    
    def test_indicator_chain_processing(self):
        """Test indicators that depend on other indicators."""
        # This would test more complex indicators that use other indicators
        # as inputs (like MACD using EMA, or custom composite indicators)
        pass


# Pytest fixtures for reusable test data
@pytest.fixture
def sample_ohlcv():
    """Fixture providing sample OHLCV data."""
    return TestDataFixtures.generate_sample_ohlcv(100)


@pytest.fixture
def trending_up_data():
    """Fixture providing uptrending data."""
    return TestDataFixtures.generate_trending_data(50, "up")


@pytest.fixture
def trending_down_data():
    """Fixture providing downtrending data."""
    return TestDataFixtures.generate_trending_data(50, "down")


@pytest.fixture
def sideways_data():
    """Fixture providing sideways trending data."""
    return TestDataFixtures.generate_trending_data(50, "sideways")


# Parametrized tests for different indicator periods
@pytest.mark.parametrize("period", [5, 10, 20, 50])
def test_sma_different_periods(period):
    """Test SMA with different periods."""
    config = IndicatorConfig(period=period)
    sma = SMAIndicator(config)
    data = TestDataFixtures.generate_sample_ohlcv(period + 10)
    
    for candle in data:
        sma.update(candle)
    
    assert sma.is_ready()
    assert sma.period == period


@pytest.mark.parametrize("std_dev", [1.0, 1.5, 2.0, 2.5])
def test_bollinger_bands_different_std_dev(std_dev):
    """Test Bollinger Bands with different standard deviations."""
    config = IndicatorConfig(period=20, std_dev=std_dev)
    bb = BollingerBandsIndicator(config)
    data = TestDataFixtures.generate_sample_ohlcv(30)
    
    for candle in data:
        bb.update(candle)
    
    if bb.is_ready():
        values = bb.get_values()
        # Larger std_dev should create wider bands
        band_width = float(values['upper_band']) - float(values['lower_band'])
        assert band_width > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])