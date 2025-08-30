"""
Data Fetcher Module
==================

Handles fetching market data from exchanges with proper rate limiting,
error handling, and retry mechanisms. Supports both real-time and 
historical data retrieval.

Features:
- OHLCV data fetching from Binance
- Automatic rate limiting and request throttling  
- Retry mechanisms with exponential backoff
- Data validation and error handling
- Caching for improved performance
- Support for multiple timeframes
- Network error recovery
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import pandas as pd

from core.exchange.binance_exchange import BinanceExchange
from core.exchange.base_exchange import MarketData
from core.utils.exceptions import (
    DataError, ExchangeConnectionError, NetworkError,
    ValidationError, TradingBotException
)


@dataclass
class DataFetchRequest:
    """Data fetch request structure."""
    symbol: str
    interval: str
    limit: int = 500
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    priority: int = 1  # 1 = high, 2 = medium, 3 = low


@dataclass
class DataFetchResult:
    """Data fetch result structure."""
    symbol: str
    interval: str
    data: pd.DataFrame
    metadata: Dict[str, Any]
    fetch_time: datetime
    is_cached: bool = False


class RateLimiter:
    """
    Rate limiter for API requests with configurable limits and windows.
    """
    
    def __init__(self, max_requests: int = 1200, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests (int): Maximum requests per time window
            time_window (int): Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = asyncio.Lock()
        
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            # Check if we can make a request
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time + 0.1)  # Add small buffer
            
            # Record this request
            self.requests.append(now)


class DataCache:
    """
    Simple in-memory cache for market data with TTL support.
    """
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        """
        Initialize data cache.
        
        Args:
            default_ttl (int): Default TTL in seconds
        """
        self.cache: Dict[str, Tuple[pd.DataFrame, datetime, int]] = {}
        self.default_ttl = default_ttl
        
    def _generate_key(self, symbol: str, interval: str, limit: int, 
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> str:
        """Generate cache key for request."""
        key_parts = [symbol, interval, str(limit)]
        if start_time:
            key_parts.append(start_time.isoformat())
        if end_time:
            key_parts.append(end_time.isoformat())
        return "|".join(key_parts)
    
    def get(self, symbol: str, interval: str, limit: int,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Get cached data if available and not expired."""
        key = self._generate_key(symbol, interval, limit, start_time, end_time)
        
        if key in self.cache:
            data, cached_time, ttl = self.cache[key]
            if (datetime.now() - cached_time).total_seconds() < ttl:
                return data.copy()
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, symbol: str, interval: str, limit: int, data: pd.DataFrame,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None, ttl: Optional[int] = None) -> None:
        """Cache data with TTL."""
        key = self._generate_key(symbol, interval, limit, start_time, end_time)
        ttl = ttl or self.default_ttl
        self.cache[key] = (data.copy(), datetime.now(), ttl)
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        expired_keys = []
        now = datetime.now()
        
        for key, (data, cached_time, ttl) in self.cache.items():
            if (now - cached_time).total_seconds() >= ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)


class DataFetcher:
    """
    Main data fetcher class that handles market data retrieval from exchanges.
    
    Features:
    - Rate-limited API requests
    - Automatic retry with exponential backoff
    - Data validation and error handling
    - Caching for improved performance
    - Support for multiple data types and timeframes
    """
    
    def __init__(self, settings, exchange: Optional[BinanceExchange] = None):
        """
        Initialize the data fetcher.
        
        Args:
            settings: Configuration settings
            exchange: Optional exchange instance (will create if not provided)
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Exchange connection
        self.exchange = exchange or BinanceExchange(settings)
        self.is_exchange_initialized = exchange is not None
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            max_requests=1200,  # Binance limit: 1200 requests per minute
            time_window=60
        )
        
        # Caching
        self.cache = DataCache(default_ttl=300)  # 5 minutes
        
        # Request queue for priority handling
        self.request_queue: asyncio.Queue = asyncio.Queue()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0
        }
        
        self.logger.info("📊 Data Fetcher initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the data fetcher and its dependencies.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("🔧 Initializing Data Fetcher...")
            
            # Initialize exchange if not already done
            if not self.is_exchange_initialized:
                success = await self.exchange.initialize()
                if not success:
                    raise ExchangeConnectionError("Failed to initialize exchange connection")
                self.is_exchange_initialized = True
            
            # Test data fetch
            await self._test_data_fetch()
            
            self.logger.info("✅ Data Fetcher initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data Fetcher initialization failed: {e}")
            return False
    
    async def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 500,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         use_cache: bool = True,
                         priority: int = 1) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            interval (str): Time interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit (int): Number of candles to fetch (max 1000)
            start_time (Optional[datetime]): Start time for historical data
            end_time (Optional[datetime]): End time for historical data
            use_cache (bool): Whether to use cached data
            priority (int): Request priority (1=high, 2=medium, 3=low)
            
        Returns:
            pd.DataFrame: OHLCV data with columns [timestamp, open, high, low, close, volume]
            
        Raises:
            DataError: If data fetch fails
            ValidationError: If parameters are invalid
        """
        # Validate parameters
        self._validate_fetch_params(symbol, interval, limit, start_time, end_time)
        
        # Check cache first
        if use_cache:
            cached_data = self.cache.get(symbol, interval, limit, start_time, end_time)
            if cached_data is not None:
                self.stats['cache_hits'] += 1
                self.logger.debug(f"📦 Cache hit for {symbol} {interval}")
                return cached_data
            else:
                self.stats['cache_misses'] += 1
        
        # Fetch data from exchange
        try:
            start_fetch_time = time.time()
            
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Fetch market data
            market_data = await self._fetch_with_retry(
                symbol, interval, limit, start_time, end_time, max_retries=3
            )
            
            # Convert to DataFrame
            df = self._convert_market_data_to_dataframe(market_data)
            
            # Update statistics
            fetch_time = time.time() - start_fetch_time
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['successful_requests'] - 1) + fetch_time)
                / self.stats['successful_requests']
            )
            
            # Cache the result
            if use_cache:
                cache_ttl = self._calculate_cache_ttl(interval)
                self.cache.set(symbol, interval, limit, df, start_time, end_time, cache_ttl)
            
            self.logger.debug(f"✅ Fetched {len(df)} candles for {symbol} {interval} in {fetch_time:.2f}s")
            return df
            
        except Exception as e:
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            self.logger.error(f"❌ Failed to fetch OHLCV data for {symbol} {interval}: {e}")
            raise DataError(
                f"Failed to fetch OHLCV data for {symbol}",
                data_source="binance",
                symbol=symbol,
                timeframe=interval
            ) from e
    
    async def fetch_multiple_ohlcv(self, requests: List[DataFetchRequest],
                                  use_cache: bool = True,
                                  max_concurrent: int = 5) -> List[DataFetchResult]:
        """
        Fetch OHLCV data for multiple symbols/intervals concurrently.
        
        Args:
            requests (List[DataFetchRequest]): List of fetch requests
            use_cache (bool): Whether to use cached data
            max_concurrent (int): Maximum concurrent requests
            
        Returns:
            List[DataFetchResult]: List of fetch results
        """
        # Sort requests by priority
        sorted_requests = sorted(requests, key=lambda x: x.priority)
        
        # Create semaphore for concurrent limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_single(request: DataFetchRequest) -> DataFetchResult:
            async with semaphore:
                try:
                    fetch_start = datetime.now()
                    df = await self.fetch_ohlcv(
                        request.symbol,
                        request.interval,
                        request.limit,
                        request.start_time,
                        request.end_time,
                        use_cache
                    )
                    
                    return DataFetchResult(
                        symbol=request.symbol,
                        interval=request.interval,
                        data=df,
                        metadata={
                            'limit': request.limit,
                            'start_time': request.start_time,
                            'end_time': request.end_time,
                            'rows_count': len(df)
                        },
                        fetch_time=fetch_start,
                        is_cached=False  # TODO: Track cache hits
                    )
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to fetch {request.symbol} {request.interval}: {e}")
                    # Return empty result instead of failing completely
                    return DataFetchResult(
                        symbol=request.symbol,
                        interval=request.interval,
                        data=pd.DataFrame(),
                        metadata={'error': str(e)},
                        fetch_time=datetime.now(),
                        is_cached=False
                    )
        
        # Execute all requests concurrently
        tasks = [fetch_single(request) for request in sorted_requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = [result for result in results if isinstance(result, DataFetchResult)]
        
        self.logger.info(f"✅ Completed {len(valid_results)}/{len(requests)} data fetch requests")
        return valid_results
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker information for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Dict[str, Any]: Ticker information
        """
        try:
            await self.rate_limiter.acquire()
            ticker = await self.exchange.get_ticker(symbol)
            
            return {
                'symbol': ticker.symbol,
                'price': ticker.price,
                'change_24h': ticker.change_24h,
                'change_24h_percent': ticker.change_24h_percent,
                'high_24h': ticker.high_24h,
                'low_24h': ticker.low_24h,
                'volume_24h': ticker.volume_24h,
                'timestamp': ticker.timestamp
            }
            
        except Exception as e:
            raise DataError(f"Failed to fetch ticker for {symbol}", 
                          data_source="binance", symbol=symbol) from e
    
    async def fetch_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Fetch order book data for a symbol.
        
        Args:
            symbol (str): Trading symbol
            limit (int): Depth of order book
            
        Returns:
            Dict[str, Any]: Order book data
        """
        try:
            await self.rate_limiter.acquire()
            return await self.exchange.get_order_book(symbol, limit)
            
        except Exception as e:
            raise DataError(f"Failed to fetch order book for {symbol}",
                          data_source="binance", symbol=symbol) from e
    
    async def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is available for trading.
        
        Args:
            symbol (str): Trading symbol to validate
            
        Returns:
            bool: True if symbol is valid
        """
        try:
            await self.rate_limiter.acquire()
            symbol_info = await self.exchange.get_symbol_info(symbol)
            return symbol_info.get('status') == 'TRADING'
            
        except Exception as e:
            self.logger.warning(f"⚠️ Symbol validation failed for {symbol}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get data fetcher statistics.
        
        Returns:
            Dict[str, Any]: Statistics information
        """
        return {
            **self.stats,
            'cache_size': len(self.cache.cache),
            'success_rate': (
                self.stats['successful_requests'] / max(self.stats['total_requests'], 1) * 100
            ),
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1) * 100
            )
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources and disconnect from exchange."""
        try:
            self.logger.info("🧹 Cleaning up Data Fetcher...")
            
            # Clear cache
            self.cache.clear()
            
            # Disconnect exchange if we initialized it
            if self.is_exchange_initialized and self.exchange:
                await self.exchange.disconnect()
            
            self.logger.info("✅ Data Fetcher cleanup complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during Data Fetcher cleanup: {e}")
    
    # Private helper methods
    
    def _validate_fetch_params(self, symbol: str, interval: str, limit: int,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> None:
        """Validate fetch parameters."""
        if not symbol or len(symbol) < 3:
            raise ValidationError("Invalid symbol", field_name="symbol", field_value=symbol)
        
        valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', 
                          '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        if interval not in valid_intervals:
            raise ValidationError(
                f"Invalid interval. Must be one of: {valid_intervals}",
                field_name="interval", 
                field_value=interval
            )
        
        if limit <= 0 or limit > 1000:
            raise ValidationError(
                "Limit must be between 1 and 1000",
                field_name="limit",
                field_value=limit
            )
        
        if start_time and end_time and start_time >= end_time:
            raise ValidationError("Start time must be before end time")
    
    async def _test_data_fetch(self) -> None:
        """Test data fetch functionality."""
        try:
            self.logger.info("🧪 Testing data fetch functionality...")
            
            # Test with a simple request
            test_symbol = self.settings.DEFAULT_SYMBOL or 'BTCUSDT'
            test_data = await self.exchange.get_klines(test_symbol, '1h', limit=1)
            
            if not test_data:
                raise DataError("Test data fetch returned empty result")
            
            self.logger.info("✅ Data fetch test successful")
            
        except Exception as e:
            raise DataError(f"Data fetch test failed: {e}") from e
    
    async def _fetch_with_retry(self, symbol: str, interval: str, limit: int,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               max_retries: int = 3) -> List[MarketData]:
        """
        Fetch data with retry mechanism and exponential backoff.
        
        Args:
            symbol (str): Trading symbol
            interval (str): Time interval
            limit (int): Number of candles
            start_time (Optional[datetime]): Start time
            end_time (Optional[datetime]): End time
            max_retries (int): Maximum number of retries
            
        Returns:
            List[MarketData]: Market data from exchange
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await self.exchange.get_klines(
                    symbol, interval, limit, start_time, end_time
                )
                
            except ExchangeConnectionError as e:
                last_exception = e
                if attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s, 8s...
                    wait_time = 2 ** attempt
                    self.logger.warning(
                        f"⚠️ Fetch attempt {attempt + 1} failed for {symbol}, "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"❌ All {max_retries + 1} fetch attempts failed for {symbol}")
                    
            except Exception as e:
                last_exception = e
                self.logger.error(f"❌ Unexpected error fetching {symbol}: {e}")
                break
        
        # If we get here, all retries failed
        if isinstance(last_exception, ExchangeConnectionError):
            raise last_exception
        else:
            raise DataError(f"Failed to fetch data after {max_retries + 1} attempts") from last_exception
    
    def _convert_market_data_to_dataframe(self, market_data: List[MarketData]) -> pd.DataFrame:
        """
        Convert MarketData list to pandas DataFrame.
        
        Args:
            market_data (List[MarketData]): Market data from exchange
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        if not market_data:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        data_list = []
        for candle in market_data:
            data_list.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        df = pd.DataFrame(data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _calculate_cache_ttl(self, interval: str) -> int:
        """Calculate appropriate cache TTL based on interval."""
        interval_seconds = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600,
            '8h': 28800, '12h': 43200, '1d': 86400, '3d': 259200,
            '1w': 604800, '1M': 2592000
        }
        
        base_seconds = interval_seconds.get(interval, 300)  # Default 5 minutes
        
        # Cache for 1/10th of the interval, minimum 30 seconds, maximum 5 minutes
        ttl = max(30, min(300, base_seconds // 10))
        return ttl