"""
Base Exchange Interface
======================

Abstract base class defining the interface for all exchange integrations.
This ensures consistency across different exchange implementations and makes
the system easily extensible to support additional exchanges.

All exchange implementations must inherit from this class and implement
all abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from core.utils.exceptions import ExchangeConnectionError, OrderExecutionError


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class PositionSide(Enum):
    """Position side for futures trading."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # For hedge mode


@dataclass
class OrderInfo:
    """Order information structure."""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: str
    filled_quantity: float
    remaining_quantity: float
    average_price: Optional[float]
    timestamp: datetime
    update_time: datetime


@dataclass
class Position:
    """Position information structure."""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    percentage: float
    timestamp: datetime


@dataclass
class Balance:
    """Balance information structure."""
    asset: str
    free: float
    locked: float
    total: float


@dataclass
class MarketData:
    """Market data structure for OHLCV data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Ticker:
    """Ticker information structure."""
    symbol: str
    price: float
    change_24h: float
    change_24h_percent: float
    high_24h: float
    low_24h: float
    volume_24h: float
    timestamp: datetime


class BaseExchange(ABC):
    """
    Abstract base class for all exchange integrations.
    
    This class defines the interface that all exchange implementations must follow.
    It provides a consistent API for interacting with different cryptocurrency exchanges.
    
    Key Responsibilities:
    - Market data retrieval
    - Account information access
    - Order management (place, cancel, query)
    - Position management (for futures exchanges)
    - Balance information
    - Connection management
    """
    
    def __init__(self, settings):
        """
        Initialize the exchange with configuration settings.
        
        Args:
            settings: Configuration settings containing API credentials and options
        """
        self.settings = settings
        self.is_connected = False
        self.is_testnet = getattr(settings, 'BINANCE_TESTNET', True)
        self.client = None
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the exchange connection and validate credentials.
        
        Returns:
            bool: True if initialization successful, False otherwise
            
        Raises:
            ExchangeConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the exchange and cleanup resources.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the exchange connection is healthy.
        
        Returns:
            bool: True if connection is healthy, False otherwise
        """
        pass
    
    # Market Data Methods
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get current ticker information for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            Ticker: Current ticker information
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    @abstractmethod
    async def get_klines(self, symbol: str, interval: str, 
                        limit: int = 500, start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[MarketData]:
        """
        Get candlestick/kline data for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            interval (str): Time interval (e.g., '1m', '5m', '1h', '1d')
            limit (int): Number of klines to retrieve (max 1000)
            start_time (Optional[datetime]): Start time for data
            end_time (Optional[datetime]): End time for data
            
        Returns:
            List[MarketData]: List of OHLCV data
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book data for a symbol.
        
        Args:
            symbol (str): Trading symbol
            limit (int): Depth of order book
            
        Returns:
            Dict[str, Any]: Order book data with bids and asks
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    # Account Information Methods
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information including balances and permissions.
        
        Returns:
            Dict[str, Any]: Account information
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    @abstractmethod
    async def get_balances(self) -> List[Balance]:
        """
        Get account balances for all assets.
        
        Returns:
            List[Balance]: List of balance information
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    @abstractmethod
    async def get_balance(self, asset: str) -> Optional[Balance]:
        """
        Get balance for a specific asset.
        
        Args:
            asset (str): Asset symbol (e.g., 'USDT', 'BTC')
            
        Returns:
            Optional[Balance]: Balance information or None if not found
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    # Position Management Methods (for futures exchanges)
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get current positions.
        
        Args:
            symbol (Optional[str]): Specific symbol to get position for
            
        Returns:
            List[Position]: List of current positions
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Optional[Position]: Position information or None if no position
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    # Order Management Methods
    
    @abstractmethod
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: float, price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: str = "GTC",
                         reduce_only: bool = False,
                         client_order_id: Optional[str] = None) -> OrderInfo:
        """
        Place a new order.
        
        Args:
            symbol (str): Trading symbol
            side (OrderSide): Buy or sell
            order_type (OrderType): Type of order
            quantity (float): Order quantity
            price (Optional[float]): Order price for limit orders
            stop_price (Optional[float]): Stop price for stop orders
            time_in_force (str): Time in force (GTC, IOC, FOK)
            reduce_only (bool): Reduce only flag for futures
            client_order_id (Optional[str]): Client order ID
            
        Returns:
            OrderInfo: Information about the placed order
            
        Raises:
            OrderExecutionError: If order placement fails
        """
        pass
    
    @abstractmethod
    async def place_test_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                              quantity: float, price: Optional[float] = None,
                              stop_price: Optional[float] = None) -> bool:
        """
        Place a test order to validate parameters without actual execution.
        
        Args:
            symbol (str): Trading symbol
            side (OrderSide): Buy or sell
            order_type (OrderType): Type of order
            quantity (float): Order quantity
            price (Optional[float]): Order price for limit orders
            stop_price (Optional[float]): Stop price for stop orders
            
        Returns:
            bool: True if test order is valid
            
        Raises:
            OrderExecutionError: If test order validation fails
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            symbol (str): Trading symbol
            order_id (str): Order ID to cancel
            
        Returns:
            bool: True if order was cancelled successfully
            
        Raises:
            OrderExecutionError: If order cancellation fails
        """
        pass
    
    @abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.
        
        Args:
            symbol (Optional[str]): Symbol to cancel orders for, or None for all symbols
            
        Returns:
            int: Number of orders cancelled
            
        Raises:
            OrderExecutionError: If order cancellation fails
        """
        pass
    
    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> Optional[OrderInfo]:
        """
        Get information about a specific order.
        
        Args:
            symbol (str): Trading symbol
            order_id (str): Order ID
            
        Returns:
            Optional[OrderInfo]: Order information or None if not found
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        """
        Get all open orders.
        
        Args:
            symbol (Optional[str]): Symbol to get orders for, or None for all symbols
            
        Returns:
            List[OrderInfo]: List of open orders
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    @abstractmethod
    async def get_order_history(self, symbol: str, limit: int = 100) -> List[OrderInfo]:
        """
        Get order history for a symbol.
        
        Args:
            symbol (str): Trading symbol
            limit (int): Maximum number of orders to retrieve
            
        Returns:
            List[OrderInfo]: List of historical orders
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    # Trading Rules and Limits
    
    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information including trading rules and limits.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Dict[str, Any]: Symbol information and trading rules
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    @abstractmethod
    async def get_trading_fees(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get trading fees information.
        
        Args:
            symbol (Optional[str]): Symbol to get fees for
            
        Returns:
            Dict[str, Any]: Trading fees information
            
        Raises:
            ExchangeConnectionError: If API call fails
        """
        pass
    
    # Utility Methods
    
    def format_symbol(self, symbol: str) -> str:
        """
        Format symbol according to exchange requirements.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            str: Formatted symbol
        """
        return symbol.upper()
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported by the exchange.
        
        Args:
            symbol (str): Trading symbol to validate
            
        Returns:
            bool: True if symbol is valid
        """
        # Default implementation - should be overridden by specific exchanges
        return bool(symbol and len(symbol) >= 3)
    
    def calculate_quantity_precision(self, symbol: str, quantity: float) -> float:
        """
        Calculate quantity with proper precision for the symbol.
        
        Args:
            symbol (str): Trading symbol
            quantity (float): Raw quantity
            
        Returns:
            float: Quantity with proper precision
        """
        # Default implementation - should be overridden by specific exchanges
        return round(quantity, 6)
    
    def calculate_price_precision(self, symbol: str, price: float) -> float:
        """
        Calculate price with proper precision for the symbol.
        
        Args:
            symbol (str): Trading symbol
            price (float): Raw price
            
        Returns:
            float: Price with proper precision
        """
        # Default implementation - should be overridden by specific exchanges
        return round(price, 2)