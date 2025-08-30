"""
Binance Exchange Implementation
===============================

Concrete implementation of the BaseExchange interface for Binance Futures.
Provides full integration with Binance API including market data, account
management, and order execution.

Features:
- Futures trading support
- Real-time market data
- Order management with proper error handling
- Account and position information
- Testnet support for development
- Rate limiting and connection management
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN

from binance import AsyncClient, BinanceSocketManager
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.enums import *

from .base_exchange import (
    BaseExchange, OrderSide, OrderType, PositionSide,
    OrderInfo, Position, Balance, MarketData, Ticker
)
from core.utils.exceptions import (
    ExchangeConnectionError, OrderExecutionError, DataError
)


class BinanceExchange(BaseExchange):
    """
    Binance Futures exchange implementation.
    
    This class provides complete integration with Binance Futures API,
    supporting both testnet and mainnet operations.
    
    Key Features:
    - Async/await support for better performance
    - Comprehensive error handling and logging
    - Position management for futures trading
    - Market data streaming capabilities
    - Order management with validation
    - Account information and balance tracking
    """
    
    def __init__(self, settings):
        """
        Initialize Binance exchange with settings.
        
        Args:
            settings: Configuration settings containing API credentials
        """
        super().__init__(settings)
        self.logger = logging.getLogger(__name__)
        
        # Binance specific settings
        self.api_key = settings.BINANCE_API_KEY
        self.secret_key = settings.BINANCE_SECRET_KEY
        self.testnet = settings.BINANCE_TESTNET
        
        # Client instances
        self.client: Optional[AsyncClient] = None
        self.socket_manager: Optional[BinanceSocketManager] = None
        
        # Exchange info cache
        self.exchange_info: Optional[Dict] = None
        self.symbol_info_cache: Dict[str, Dict] = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.request_delay = 0.1  # 100ms between requests
        
        self.logger.info(f"🏗️ Binance Exchange initialized (testnet: {self.testnet})")
    
    async def initialize(self) -> bool:
        """
        Initialize Binance client and validate connection.
        
        Returns:
            bool: True if initialization successful
            
        Raises:
            ExchangeConnectionError: If connection fails
        """
        try:
            self.logger.info("🔌 Connecting to Binance API...")
            
            # Validate API credentials
            if not self.api_key or not self.secret_key:
                if not self.testnet:
                    raise ExchangeConnectionError(
                        "API key and secret are required for live trading",
                        exchange="binance"
                    )
                else:
                    self.logger.warning("⚠️ No API credentials provided, using public endpoints only")
            
            # Create async client
            self.client = await AsyncClient.create(
                api_key=self.api_key,
                api_secret=self.secret_key,
                testnet=self.testnet
            )
            
            # Test connection
            await self._test_connection()
            
            # Load exchange info
            await self._load_exchange_info()
            
            # Initialize socket manager for streaming
            if self.client:
                self.socket_manager = BinanceSocketManager(self.client)
            
            self.is_connected = True
            self.logger.info("✅ Binance connection established successfully")
            return True
            
        except BinanceAPIException as e:
            error_msg = f"Binance API error: {e}"
            self.logger.error(f"❌ {error_msg}")
            raise ExchangeConnectionError(error_msg, exchange="binance", 
                                        api_error_code=str(e.code)) from e
        except Exception as e:
            error_msg = f"Failed to initialize Binance connection: {e}"
            self.logger.error(f"❌ {error_msg}")
            raise ExchangeConnectionError(error_msg, exchange="binance") from e
    
    async def disconnect(self) -> None:
        """Disconnect from Binance and cleanup resources."""
        try:
            self.logger.info("🔌 Disconnecting from Binance...")
            
            if self.socket_manager:
                # Close any active streams
                # await self.socket_manager.close()
                self.socket_manager = None
            
            if self.client:
                await self.client.close_connection()
                self.client = None
            
            self.is_connected = False
            self.logger.info("✅ Binance disconnection complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during Binance disconnection: {e}")
    
    async def health_check(self) -> bool:
        """
        Check Binance connection health.
        
        Returns:
            bool: True if connection is healthy
        """
        try:
            if not self.client:
                return False
            
            # Simple ping to test connection
            await self.client.ping()
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Binance health check failed: {e}")
            return False
    
    # Market Data Methods
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get current ticker information for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            Ticker: Current ticker information
        """
        try:
            await self._rate_limit()
            symbol = self.format_symbol(symbol)
            
            ticker_data = await self.client.get_ticker(symbol=symbol)
            
            return Ticker(
                symbol=symbol,
                price=float(ticker_data['price']),
                change_24h=float(ticker_data['priceChange']),
                change_24h_percent=float(ticker_data['priceChangePercent']),
                high_24h=float(ticker_data['highPrice']),
                low_24h=float(ticker_data['lowPrice']),
                volume_24h=float(ticker_data['volume']),
                timestamp=datetime.now(timezone.utc)
            )
            
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Failed to get ticker for {symbol}: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
        except Exception as e:
            raise DataError(f"Error getting ticker for {symbol}: {e}",
                          data_source="binance", symbol=symbol) from e
    
    async def get_klines(self, symbol: str, interval: str, 
                        limit: int = 500, start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[MarketData]:
        """
        Get candlestick data for a symbol.
        
        Args:
            symbol (str): Trading symbol
            interval (str): Time interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit (int): Number of klines to retrieve
            start_time (Optional[datetime]): Start time
            end_time (Optional[datetime]): End time
            
        Returns:
            List[MarketData]: List of OHLCV data
        """
        try:
            await self._rate_limit()
            symbol = self.format_symbol(symbol)
            
            # Convert datetime to timestamp if provided
            start_str = None
            end_str = None
            if start_time:
                start_str = str(int(start_time.timestamp() * 1000))
            if end_time:
                end_str = str(int(end_time.timestamp() * 1000))
            
            klines = await self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                startTime=start_str,
                endTime=end_str
            )
            
            market_data = []
            for kline in klines:
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5])
                ))
            
            return market_data
            
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Failed to get klines for {symbol}: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
        except Exception as e:
            raise DataError(f"Error getting klines for {symbol}: {e}",
                          data_source="binance", symbol=symbol) from e
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book data for a symbol.
        
        Args:
            symbol (str): Trading symbol
            limit (int): Depth of order book
            
        Returns:
            Dict[str, Any]: Order book with bids and asks
        """
        try:
            await self._rate_limit()
            symbol = self.format_symbol(symbol)
            
            order_book = await self.client.get_order_book(symbol=symbol, limit=limit)
            
            return {
                'symbol': symbol,
                'bids': [[float(bid[0]), float(bid[1])] for bid in order_book['bids']],
                'asks': [[float(ask[0]), float(ask[1])] for ask in order_book['asks']],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Failed to get order book for {symbol}: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
    
    # Account Information Methods
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict[str, Any]: Account information
        """
        try:
            await self._rate_limit()
            account_info = await self.client.futures_account()
            
            return {
                'account_alias': account_info.get('accountAlias', ''),
                'asset': account_info.get('asset', ''),
                'balance': float(account_info.get('balance', 0)),
                'cross_wallet_balance': float(account_info.get('crossWalletBalance', 0)),
                'cross_unrealized_pnl': float(account_info.get('crossUnrealizedPnl', 0)),
                'available_balance': float(account_info.get('availableBalance', 0)),
                'max_withdraw_amount': float(account_info.get('maxWithdrawAmount', 0)),
                'margin_available': account_info.get('marginAvailable', True),
                'update_time': datetime.fromtimestamp(account_info.get('updateTime', 0) / 1000, tz=timezone.utc)
            }
            
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Failed to get account info: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
    
    async def get_balances(self) -> List[Balance]:
        """
        Get account balances for all assets.
        
        Returns:
            List[Balance]: List of balance information
        """
        try:
            await self._rate_limit()
            account_info = await self.client.futures_account()
            
            balances = []
            for asset_info in account_info.get('assets', []):
                if float(asset_info['walletBalance']) > 0:  # Only include non-zero balances
                    balances.append(Balance(
                        asset=asset_info['asset'],
                        free=float(asset_info['availableBalance']),
                        locked=float(asset_info['initialMargin']),
                        total=float(asset_info['walletBalance'])
                    ))
            
            return balances
            
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Failed to get balances: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
    
    async def get_balance(self, asset: str) -> Optional[Balance]:
        """
        Get balance for a specific asset.
        
        Args:
            asset (str): Asset symbol (e.g., 'USDT')
            
        Returns:
            Optional[Balance]: Balance information or None
        """
        balances = await self.get_balances()
        for balance in balances:
            if balance.asset == asset.upper():
                return balance
        return None
    
    # Position Management Methods
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get current positions.
        
        Args:
            symbol (Optional[str]): Specific symbol to get position for
            
        Returns:
            List[Position]: List of current positions
        """
        try:
            await self._rate_limit()
            
            if symbol:
                symbol = self.format_symbol(symbol)
                positions_data = await self.client.futures_position_information(symbol=symbol)
            else:
                positions_data = await self.client.futures_position_information()
            
            positions = []
            for pos_data in positions_data:
                position_amt = float(pos_data['positionAmt'])
                if abs(position_amt) > 0:  # Only include non-zero positions
                    positions.append(Position(
                        symbol=pos_data['symbol'],
                        side=PositionSide.LONG if position_amt > 0 else PositionSide.SHORT,
                        size=abs(position_amt),
                        entry_price=float(pos_data['entryPrice']),
                        mark_price=float(pos_data['markPrice']),
                        unrealized_pnl=float(pos_data['unRealizedPnl']),
                        percentage=float(pos_data['percentage']),
                        timestamp=datetime.fromtimestamp(int(pos_data['updateTime']) / 1000, tz=timezone.utc)
                    ))
            
            return positions
            
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Failed to get positions: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Optional[Position]: Position information or None
        """
        positions = await self.get_positions(symbol)
        return positions[0] if positions else None
    
    # Order Management Methods
    
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: float, price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: str = "GTC",
                         reduce_only: bool = False,
                         client_order_id: Optional[str] = None) -> OrderInfo:
        """
        Place a new order on Binance Futures.
        
        Args:
            symbol (str): Trading symbol
            side (OrderSide): Buy or sell
            order_type (OrderType): Type of order
            quantity (float): Order quantity
            price (Optional[float]): Order price for limit orders
            stop_price (Optional[float]): Stop price for stop orders
            time_in_force (str): Time in force (GTC, IOC, FOK)
            reduce_only (bool): Reduce only flag
            client_order_id (Optional[str]): Client order ID
            
        Returns:
            OrderInfo: Information about the placed order
        """
        try:
            await self._rate_limit()
            symbol = self.format_symbol(symbol)
            
            # Validate and format parameters
            quantity = self.calculate_quantity_precision(symbol, quantity)
            if price:
                price = self.calculate_price_precision(symbol, price)
            if stop_price:
                stop_price = self.calculate_price_precision(symbol, stop_price)
            
            # Convert enums to Binance API format
            binance_side = SIDE_BUY if side == OrderSide.BUY else SIDE_SELL
            binance_type = self._convert_order_type_to_binance(order_type)
            
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'side': binance_side,
                'type': binance_type,
                'quantity': quantity,
                'timeInForce': time_in_force,
                'reduceOnly': reduce_only
            }
            
            # Add price for limit orders
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
                if price is None:
                    raise OrderExecutionError("Price is required for limit orders")
                order_params['price'] = price
            
            # Add stop price for stop orders
            if order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
                if stop_price is None:
                    raise OrderExecutionError("Stop price is required for stop orders")
                order_params['stopPrice'] = stop_price
            
            # Add client order ID if provided
            if client_order_id:
                order_params['newClientOrderId'] = client_order_id
            
            # Place the order
            order_result = await self.client.futures_create_order(**order_params)
            
            # Convert response to OrderInfo
            return self._convert_binance_order_to_order_info(order_result)
            
        except BinanceOrderException as e:
            raise OrderExecutionError(f"Binance order error: {e}", 
                                    order_type=order_type.value, symbol=symbol, 
                                    side=side.value, quantity=quantity, price=price) from e
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Binance API error during order placement: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
        except Exception as e:
            raise OrderExecutionError(f"Unexpected error placing order: {e}",
                                    symbol=symbol, side=side.value) from e
    
    async def place_test_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                              quantity: float, price: Optional[float] = None,
                              stop_price: Optional[float] = None) -> bool:
        """
        Place a test order to validate parameters.
        
        Args:
            symbol (str): Trading symbol
            side (OrderSide): Buy or sell
            order_type (OrderType): Type of order
            quantity (float): Order quantity
            price (Optional[float]): Order price for limit orders
            stop_price (Optional[float]): Stop price for stop orders
            
        Returns:
            bool: True if test order is valid
        """
        try:
            await self._rate_limit()
            symbol = self.format_symbol(symbol)
            
            # Validate and format parameters
            quantity = self.calculate_quantity_precision(symbol, quantity)
            if price:
                price = self.calculate_price_precision(symbol, price)
            if stop_price:
                stop_price = self.calculate_price_precision(symbol, stop_price)
            
            # Convert enums to Binance API format
            binance_side = SIDE_BUY if side == OrderSide.BUY else SIDE_SELL
            binance_type = self._convert_order_type_to_binance(order_type)
            
            # Prepare test order parameters
            test_params = {
                'symbol': symbol,
                'side': binance_side,
                'type': binance_type,
                'quantity': quantity,
                'timeInForce': TIME_IN_FORCE_GTC
            }
            
            # Add price for limit orders
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
                if price is None:
                    raise OrderExecutionError("Price is required for limit orders")
                test_params['price'] = price
            
            # Add stop price for stop orders
            if order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
                if stop_price is None:
                    raise OrderExecutionError("Stop price is required for stop orders")
                test_params['stopPrice'] = stop_price
            
            # Place test order
            await self.client.futures_create_test_order(**test_params)
            
            self.logger.info(f"✅ Test order validation successful for {symbol}")
            return True
            
        except BinanceOrderException as e:
            self.logger.warning(f"❌ Test order validation failed: {e}")
            raise OrderExecutionError(f"Test order validation failed: {e}",
                                    order_type=order_type.value, symbol=symbol) from e
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Binance API error during test order: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            symbol (str): Trading symbol
            order_id (str): Order ID to cancel
            
        Returns:
            bool: True if order was cancelled successfully
        """
        try:
            await self._rate_limit()
            symbol = self.format_symbol(symbol)
            
            await self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            
            self.logger.info(f"✅ Order {order_id} cancelled successfully for {symbol}")
            return True
            
        except BinanceAPIException as e:
            if e.code == -2011:  # Order does not exist
                self.logger.warning(f"⚠️ Order {order_id} does not exist or already cancelled")
                return True
            raise ExchangeConnectionError(f"Failed to cancel order {order_id}: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
        except Exception as e:
            raise OrderExecutionError(f"Error cancelling order {order_id}: {e}") from e
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.
        
        Args:
            symbol (Optional[str]): Symbol to cancel orders for
            
        Returns:
            int: Number of orders cancelled
        """
        try:
            await self._rate_limit()
            
            if symbol:
                symbol = self.format_symbol(symbol)
                result = await self.client.futures_cancel_all_open_orders(symbol=symbol)
                cancelled_count = result.get('msg', '').count('SUCCESS')
            else:
                # Get all open orders first
                open_orders = await self.get_open_orders()
                cancelled_count = 0
                
                # Cancel orders for each symbol
                symbols_with_orders = set(order.symbol for order in open_orders)
                for sym in symbols_with_orders:
                    try:
                        await self.client.futures_cancel_all_open_orders(symbol=sym)
                        cancelled_count += len([o for o in open_orders if o.symbol == sym])
                    except BinanceAPIException as e:
                        self.logger.warning(f"⚠️ Failed to cancel orders for {sym}: {e}")
            
            self.logger.info(f"✅ Cancelled {cancelled_count} orders")
            return cancelled_count
            
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Failed to cancel all orders: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
        except Exception as e:
            raise OrderExecutionError(f"Error cancelling all orders: {e}") from e
    
    async def get_order(self, symbol: str, order_id: str) -> Optional[OrderInfo]:
        """
        Get information about a specific order.
        
        Args:
            symbol (str): Trading symbol
            order_id (str): Order ID
            
        Returns:
            Optional[OrderInfo]: Order information or None if not found
        """
        try:
            await self._rate_limit()
            symbol = self.format_symbol(symbol)
            
            order_data = await self.client.futures_get_order(symbol=symbol, orderId=order_id)
            return self._convert_binance_order_to_order_info(order_data)
            
        except BinanceAPIException as e:
            if e.code == -2013:  # Order does not exist
                return None
            raise ExchangeConnectionError(f"Failed to get order {order_id}: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        """
        Get all open orders.
        
        Args:
            symbol (Optional[str]): Symbol to get orders for
            
        Returns:
            List[OrderInfo]: List of open orders
        """
        try:
            await self._rate_limit()
            
            if symbol:
                symbol = self.format_symbol(symbol)
                orders_data = await self.client.futures_get_open_orders(symbol=symbol)
            else:
                orders_data = await self.client.futures_get_open_orders()
            
            orders = []
            for order_data in orders_data:
                orders.append(self._convert_binance_order_to_order_info(order_data))
            
            return orders
            
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Failed to get open orders: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
    
    async def get_order_history(self, symbol: str, limit: int = 100) -> List[OrderInfo]:
        """
        Get order history for a symbol.
        
        Args:
            symbol (str): Trading symbol
            limit (int): Maximum number of orders to retrieve
            
        Returns:
            List[OrderInfo]: List of historical orders
        """
        try:
            await self._rate_limit()
            symbol = self.format_symbol(symbol)
            
            orders_data = await self.client.futures_get_all_orders(symbol=symbol, limit=limit)
            
            orders = []
            for order_data in orders_data:
                orders.append(self._convert_binance_order_to_order_info(order_data))
            
            return orders
            
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Failed to get order history for {symbol}: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
    
    # Trading Rules and Symbol Info
    
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information including trading rules.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Dict[str, Any]: Symbol information and trading rules
        """
        try:
            symbol = self.format_symbol(symbol)
            
            # Use cached exchange info if available
            if not self.exchange_info:
                await self._load_exchange_info()
            
            # Find symbol in exchange info
            for sym_info in self.exchange_info['symbols']:
                if sym_info['symbol'] == symbol:
                    return sym_info
            
            raise DataError(f"Symbol {symbol} not found in exchange info")
            
        except Exception as e:
            raise DataError(f"Error getting symbol info for {symbol}: {e}",
                          data_source="binance", symbol=symbol) from e
    
    async def get_trading_fees(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get trading fees information.
        
        Args:
            symbol (Optional[str]): Symbol to get fees for
            
        Returns:
            Dict[str, Any]: Trading fees information
        """
        try:
            await self._rate_limit()
            
            if symbol:
                symbol = self.format_symbol(symbol)
                trading_status = await self.client.futures_commission_rate(symbol=symbol)
                return {
                    'symbol': symbol,
                    'maker_commission_rate': float(trading_status['makerCommissionRate']),
                    'taker_commission_rate': float(trading_status['takerCommissionRate'])
                }
            else:
                account_info = await self.client.futures_account()
                return {
                    'maker_commission_rate': float(account_info.get('makerCommission', 0)) / 10000,
                    'taker_commission_rate': float(account_info.get('takerCommission', 0)) / 10000
                }
            
        except BinanceAPIException as e:
            raise ExchangeConnectionError(f"Failed to get trading fees: {e}",
                                        exchange="binance", api_error_code=str(e.code)) from e
    
    # Utility Methods
    
    def calculate_quantity_precision(self, symbol: str, quantity: float) -> float:
        """Calculate quantity with proper precision for the symbol."""
        try:
            symbol_info = self.symbol_info_cache.get(symbol)
            if symbol_info:
                # Find LOT_SIZE filter
                for filter_info in symbol_info.get('filters', []):
                    if filter_info['filterType'] == 'LOT_SIZE':
                        step_size = float(filter_info['stepSize'])
                        precision = len(str(step_size).rstrip('0').split('.')[-1])
                        return float(Decimal(str(quantity)).quantize(
                            Decimal(str(step_size)), rounding=ROUND_DOWN))
            
            # Default precision if no symbol info
            return round(quantity, 6)
            
        except Exception:
            return round(quantity, 6)
    
    def calculate_price_precision(self, symbol: str, price: float) -> float:
        """Calculate price with proper precision for the symbol."""
        try:
            symbol_info = self.symbol_info_cache.get(symbol)
            if symbol_info:
                # Find PRICE_FILTER
                for filter_info in symbol_info.get('filters', []):
                    if filter_info['filterType'] == 'PRICE_FILTER':
                        tick_size = float(filter_info['tickSize'])
                        precision = len(str(tick_size).rstrip('0').split('.')[-1])
                        return float(Decimal(str(price)).quantize(
                            Decimal(str(tick_size)), rounding=ROUND_DOWN))
            
            # Default precision if no symbol info
            return round(price, 2)
            
        except Exception:
            return round(price, 2)
    
    # Private Helper Methods
    
    async def _test_connection(self) -> None:
        """Test the connection to Binance API."""
        try:
            # Test public endpoint
            await self.client.ping()
            
            # Test private endpoint if credentials provided
            if self.api_key and self.secret_key:
                await self.client.futures_account()
                self.logger.info("🔐 API credentials validated successfully")
            else:
                self.logger.info("🌐 Public API connection successful")
                
        except BinanceAPIException as e:
            if e.code == -1021:  # Timestamp error
                raise ExchangeConnectionError(
                    "Timestamp synchronization error. Please check system time.",
                    exchange="binance", api_error_code=str(e.code)
                ) from e
            elif e.code == -2015:  # Invalid API key
                raise ExchangeConnectionError(
                    "Invalid API key or secret",
                    exchange="binance", api_error_code=str(e.code)
                ) from e
            else:
                raise ExchangeConnectionError(
                    f"Binance API connection test failed: {e}",
                    exchange="binance", api_error_code=str(e.code)
                ) from e
    
    async def _load_exchange_info(self) -> None:
        """Load exchange information and cache symbol data."""
        try:
            self.logger.info("📋 Loading Binance exchange information...")
            self.exchange_info = await self.client.futures_exchange_info()
            
            # Cache symbol information for quick access
            for symbol_info in self.exchange_info['symbols']:
                symbol = symbol_info['symbol']
                self.symbol_info_cache[symbol] = symbol_info
            
            self.logger.info(f"✅ Loaded info for {len(self.symbol_info_cache)} symbols")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load exchange info: {e}")
            raise ExchangeConnectionError(f"Failed to load exchange info: {e}") from e
    
    async def _rate_limit(self) -> None:
        """Simple rate limiting to avoid hitting API limits."""
        import time
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last_request)
        
        self.last_request_time = time.time()
    
    def _convert_order_type_to_binance(self, order_type: OrderType) -> str:
        """Convert OrderType enum to Binance API format."""
        type_mapping = {
            OrderType.MARKET: ORDER_TYPE_MARKET,
            OrderType.LIMIT: ORDER_TYPE_LIMIT,
            OrderType.STOP_MARKET: ORDER_TYPE_STOP_MARKET,
            OrderType.STOP_LIMIT: ORDER_TYPE_STOP,
            OrderType.TAKE_PROFIT_MARKET: ORDER_TYPE_TAKE_PROFIT_MARKET,
            OrderType.TAKE_PROFIT_LIMIT: ORDER_TYPE_TAKE_PROFIT
        }
        return type_mapping.get(order_type, ORDER_TYPE_MARKET)
    
    def _convert_binance_order_to_order_info(self, order_data: Dict) -> OrderInfo:
        """Convert Binance order data to OrderInfo object."""
        # Determine order side
        side = OrderSide.BUY if order_data['side'] == SIDE_BUY else OrderSide.SELL
        
        # Determine order type
        order_type_mapping = {
            ORDER_TYPE_MARKET: OrderType.MARKET,
            ORDER_TYPE_LIMIT: OrderType.LIMIT,
            ORDER_TYPE_STOP_MARKET: OrderType.STOP_MARKET,
            ORDER_TYPE_STOP: OrderType.STOP_LIMIT,
            ORDER_TYPE_TAKE_PROFIT_MARKET: OrderType.TAKE_PROFIT_MARKET,
            ORDER_TYPE_TAKE_PROFIT: OrderType.TAKE_PROFIT_LIMIT
        }
        order_type = order_type_mapping.get(order_data['type'], OrderType.MARKET)
        
        # Calculate remaining quantity
        orig_qty = float(order_data['origQty'])
        executed_qty = float(order_data['executedQty'])
        remaining_qty = orig_qty - executed_qty
        
        return OrderInfo(
            order_id=str(order_data['orderId']),
            client_order_id=order_data.get('clientOrderId', ''),
            symbol=order_data['symbol'],
            side=side,
            order_type=order_type,
            quantity=orig_qty,
            price=float(order_data['price']) if order_data['price'] != '0' else None,
            status=order_data['status'],
            filled_quantity=executed_qty,
            remaining_quantity=remaining_qty,
            average_price=float(order_data['avgPrice']) if float(order_data['avgPrice']) > 0 else None,
            timestamp=datetime.fromtimestamp(int(order_data['time']) / 1000, tz=timezone.utc),
            update_time=datetime.fromtimestamp(int(order_data['updateTime']) / 1000, tz=timezone.utc)
        )