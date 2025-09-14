"""
Tests Module
============

Test suite for the automated trading bot system.
Contains unit tests, integration tests, and test utilities.

Test Organization:
- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions
- fixtures/: Test data fixtures and utilities
- mocks/: Mock objects and API responses

Author: dat-ns
Version: 1.0.0
"""

import os
import sys
import pytest
import logging
from pathlib import Path
from typing import Dict, Any, List
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    'log_level': 'DEBUG',
    'test_data_dir': 'tests/data',
    'mock_api_responses': True,
    'test_timeout': 30,  # seconds
}

# Setup test logging
def setup_test_logging():
    """Setup logging configuration for tests."""
    logging.basicConfig(
        level=getattr(logging, TEST_CONFIG['log_level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tests/test.log', mode='w')
        ]
    )

class TestEnvironment:
    """Test environment management utilities."""
    
    @staticmethod
    def setup_test_environment():
        """Setup test environment variables and configuration."""
        os.environ['TRADING_BOT_ENV'] = 'test'
        os.environ['BINANCE_TESTNET'] = 'true'
        os.environ['LOG_LEVEL'] = 'DEBUG'
        
        # Create test directories
        test_dirs = ['tests/logs', 'tests/data', 'tests/reports']
        for test_dir in test_dirs:
            Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def cleanup_test_environment():
        """Cleanup test environment after tests."""
        # Clean up test files if needed
        pass

class MockBinanceAPI:
    """Mock Binance API for testing."""
    
    def __init__(self):
        self.account_info_response = {
            'accountType': 'SPOT',
            'balances': [
                {'asset': 'USDT', 'free': '10000.00000000', 'locked': '0.00000000'},
                {'asset': 'BTC', 'free': '0.50000000', 'locked': '0.00000000'}
            ],
            'canTrade': True,
            'canWithdraw': True,
            'canDeposit': True
        }
        
        self.order_response = {
            'symbol': 'BTCUSDT',
            'orderId': 12345,
            'orderListId': -1,
            'clientOrderId': 'test_order_1',
            'transactTime': 1640995200000,
            'price': '50000.00000000',
            'origQty': '0.01000000',
            'executedQty': '0.01000000',
            'cummulativeQuoteQty': '500.00000000',
            'status': 'FILLED',
            'timeInForce': 'GTC',
            'type': 'MARKET',
            'side': 'BUY'
        }
        
        self.klines_response = self._generate_sample_klines()
    
    def _generate_sample_klines(self) -> List[List]:
        """Generate sample kline data for testing."""
        import time
        import random
        
        klines = []
        base_price = 50000.0
        timestamp = int(time.time() * 1000) - (100 * 300000)  # 100 5-minute candles ago
        
        for i in range(100):
            # Generate realistic OHLCV data
            open_price = base_price * (1 + random.uniform(-0.01, 0.01))
            close_price = open_price * (1 + random.uniform(-0.02, 0.02))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.005))
            volume = random.uniform(10, 100)
            
            kline = [
                timestamp + (i * 300000),  # Open time
                f"{open_price:.8f}",       # Open
                f"{high_price:.8f}",       # High  
                f"{low_price:.8f}",        # Low
                f"{close_price:.8f}",      # Close
                f"{volume:.8f}",           # Volume
                timestamp + (i * 300000) + 299999,  # Close time
                f"{volume * close_price:.8f}",      # Quote asset volume
                random.randint(50, 200),   # Number of trades
                f"{volume * 0.6:.8f}",     # Taker buy base asset volume
                f"{volume * close_price * 0.6:.8f}",  # Taker buy quote asset volume
                "0"                        # Ignore
            ]
            klines.append(kline)
            base_price = close_price
        
        return klines
    
    async def get_account(self):
        """Mock get account info."""
        await asyncio.sleep(0.01)  # Simulate API delay
        return self.account_info_response
    
    async def create_order(self, **kwargs):
        """Mock create order."""
        await asyncio.sleep(0.01)
        # Modify response based on order parameters
        response = self.order_response.copy()
        response.update(kwargs)
        return response
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 100):
        """Mock get klines."""
        await asyncio.sleep(0.01)
        return self.klines_response[-limit:]

class TestDataBuilder:
    """Builder for test data and scenarios."""
    
    @staticmethod
    def create_market_scenario(scenario_type: str, length: int = 100) -> Dict[str, Any]:
        """
        Create different market scenarios for testing.
        
        Args:
            scenario_type: Type of scenario ('bull', 'bear', 'sideways', 'volatile')
            length: Number of data points
            
        Returns:
            Dict containing scenario data and metadata
        """
        scenarios = {
            'bull': {'trend': 0.02, 'volatility': 0.01, 'description': 'Bull market'},
            'bear': {'trend': -0.02, 'volatility': 0.015, 'description': 'Bear market'},  
            'sideways': {'trend': 0.0, 'volatility': 0.008, 'description': 'Sideways market'},
            'volatile': {'trend': 0.005, 'volatility': 0.05, 'description': 'High volatility market'}
        }
        
        if scenario_type not in scenarios:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        params = scenarios[scenario_type]
        
        return {
            'type': scenario_type,
            'parameters': params,
            'length': length,
            'data': TestDataBuilder._generate_ohlcv_data(length, **params)
        }
    
    @staticmethod
    def _generate_ohlcv_data(length: int, trend: float, volatility: float, **kwargs) -> List[Dict]:
        """Generate OHLCV data based on parameters."""
        import random
        import time
        
        data = []
        price = 50000.0  # Starting price
        timestamp = int(time.time() * 1000) - (length * 300000)
        
        for i in range(length):
            # Apply trend and volatility
            price_change = trend + random.gauss(0, volatility)
            price *= (1 + price_change)
            
            # Generate OHLC
            open_price = price * (1 + random.uniform(-0.001, 0.001))
            close_price = price
            high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, volatility * 0.3)))
            low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, volatility * 0.3)))
            volume = random.uniform(10, 1000)
            
            data.append({
                'timestamp': timestamp + (i * 300000),
                'open': round(open_price, 8),
                'high': round(high_price, 8), 
                'low': round(low_price, 8),
                'close': round(close_price, 8),
                'volume': round(volume, 8)
            })
        
        return data

class AsyncTestCase:
    """Base class for async test cases."""
    
    @staticmethod
    def run_async_test(coro):
        """Run async test function."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

# Pytest fixtures for common test objects
@pytest.fixture(scope="session")
def test_environment():
    """Setup test environment for entire test session."""
    TestEnvironment.setup_test_environment()
    setup_test_logging()
    yield
    TestEnvironment.cleanup_test_environment()

@pytest.fixture
def mock_binance_api():
    """Provide mock Binance API for tests."""
    return MockBinanceAPI()

@pytest.fixture
def sample_market_data():
    """Provide sample market data for tests."""
    return TestDataBuilder.create_market_scenario('bull', 50)

@pytest.fixture(params=['bull', 'bear', 'sideways', 'volatile'])
def market_scenarios(request):
    """Parametrized fixture for different market scenarios."""
    return TestDataBuilder.create_market_scenario(request.param, 100)

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Test markers
pytest_plugins = []

# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API connection"
    )

# Test utilities
class TestAssertions:
    """Custom assertion helpers for trading bot tests."""
    
    @staticmethod
    def assert_price_in_range(price: float, min_price: float, max_price: float, tolerance: float = 0.01):
        """Assert price is within expected range."""
        assert min_price * (1 - tolerance) <= price <= max_price * (1 + tolerance), \
            f"Price {price} not in range [{min_price}, {max_price}] with tolerance {tolerance}"
    
    @staticmethod
    def assert_signal_valid(signal, expected_types: List[str] = None):
        """Assert signal object is valid."""
        assert hasattr(signal, 'type'), "Signal must have 'type' attribute"
        assert hasattr(signal, 'strength'), "Signal must have 'strength' attribute"
        assert hasattr(signal, 'price'), "Signal must have 'price' attribute"
        assert hasattr(signal, 'timestamp'), "Signal must have 'timestamp' attribute"
        
        assert 0.0 <= signal.strength <= 1.0, f"Signal strength {signal.strength} must be between 0 and 1"
        
        if expected_types:
            assert signal.type in expected_types, f"Signal type {signal.type} not in expected types {expected_types}"
    
    @staticmethod
    def assert_indicator_ready(indicator, min_values: int = 1):
        """Assert indicator is ready and has minimum values."""
        assert indicator.is_ready(), "Indicator should be ready"
        assert len(indicator.values) >= min_values, f"Indicator should have at least {min_values} values"
        assert indicator.current_value is not None, "Indicator should have current value"

# Export test utilities
__all__ = [
    'TestEnvironment',
    'MockBinanceAPI', 
    'TestDataBuilder',
    'AsyncTestCase',
    'TestAssertions',
    'TEST_CONFIG'
]

# Version info
__version__ = "1.0.0"
__author__ = "dat-ns"