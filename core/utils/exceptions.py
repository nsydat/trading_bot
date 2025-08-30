"""
Custom Exception Classes
========================

This module defines all custom exception classes used throughout the trading bot.
These exceptions provide specific error handling for different components and scenarios.
"""

from typing import Optional, Any, Dict


class TradingBotException(Exception):
    """
    Base exception class for all trading bot related errors.
    
    All other custom exceptions should inherit from this class.
    This provides a common interface for handling bot-specific errors.
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize the trading bot exception.
        
        Args:
            message (str): Human-readable error message
            error_code (Optional[str]): Machine-readable error code
            details (Optional[Dict[str, Any]]): Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary format.
        
        Returns:
            Dict[str, Any]: Exception data as dictionary
        """
        return {
            'exception_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }


class ConfigurationError(TradingBotException):
    """
    Exception raised for configuration-related errors.
    
    This includes:
    - Missing required configuration parameters
    - Invalid configuration values
    - Environment variable issues
    - Settings validation failures
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 config_value: Optional[Any] = None):
        """
        Initialize configuration error.
        
        Args:
            message (str): Error message
            config_key (Optional[str]): Configuration key that caused the error
            config_value (Optional[Any]): Invalid configuration value
        """
        details = {}
        if config_key:
            details['config_key'] = config_key
        if config_value is not None:
            details['config_value'] = str(config_value)
        
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key
        self.config_value = config_value


class InitializationError(TradingBotException):
    """
    Exception raised when bot components fail to initialize properly.
    
    This includes:
    - Component initialization failures
    - Dependency injection issues
    - Service startup problems
    - Resource allocation failures
    """
    
    def __init__(self, message: str, component: Optional[str] = None, 
                 underlying_error: Optional[Exception] = None):
        """
        Initialize initialization error.
        
        Args:
            message (str): Error message
            component (Optional[str]): Component that failed to initialize
            underlying_error (Optional[Exception]): Original exception that caused the failure
        """
        details = {}
        if component:
            details['component'] = component
        if underlying_error:
            details['underlying_error'] = str(underlying_error)
            details['underlying_error_type'] = type(underlying_error).__name__
        
        super().__init__(message, "INIT_ERROR", details)
        self.component = component
        self.underlying_error = underlying_error


class ExchangeConnectionError(TradingBotException):
    """
    Exception raised for exchange connection and API-related errors.
    
    This includes:
    - API connection failures
    - Authentication errors
    - Rate limiting issues
    - Network timeouts
    - Invalid API responses
    """
    
    def __init__(self, message: str, exchange: str = "binance", 
                 status_code: Optional[int] = None, 
                 api_error_code: Optional[str] = None):
        """
        Initialize exchange connection error.
        
        Args:
            message (str): Error message
            exchange (str): Name of the exchange
            status_code (Optional[int]): HTTP status code if applicable
            api_error_code (Optional[str]): Exchange-specific error code
        """
        details = {
            'exchange': exchange
        }
        if status_code:
            details['status_code'] = status_code
        if api_error_code:
            details['api_error_code'] = api_error_code
        
        super().__init__(message, "EXCHANGE_ERROR", details)
        self.exchange = exchange
        self.status_code = status_code
        self.api_error_code = api_error_code


class StrategyError(TradingBotException):
    """
    Exception raised for trading strategy-related errors.
    
    This includes:
    - Strategy execution failures
    - Invalid strategy parameters
    - Signal generation errors
    - Strategy logic issues
    """
    
    def __init__(self, message: str, strategy_name: Optional[str] = None, 
                 symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """
        Initialize strategy error.
        
        Args:
            message (str): Error message
            strategy_name (Optional[str]): Name of the strategy
            symbol (Optional[str]): Trading symbol
            timeframe (Optional[str]): Timeframe being analyzed
        """
        details = {}
        if strategy_name:
            details['strategy_name'] = strategy_name
        if symbol:
            details['symbol'] = symbol
        if timeframe:
            details['timeframe'] = timeframe
        
        super().__init__(message, "STRATEGY_ERROR", details)
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.timeframe = timeframe


class RiskManagementError(TradingBotException):
    """
    Exception raised for risk management violations and errors.
    
    This includes:
    - Position size violations
    - Risk limit breaches
    - Drawdown threshold exceeded
    - Invalid risk parameters
    """
    
    def __init__(self, message: str, risk_type: Optional[str] = None, 
                 current_value: Optional[float] = None, 
                 threshold_value: Optional[float] = None):
        """
        Initialize risk management error.
        
        Args:
            message (str): Error message
            risk_type (Optional[str]): Type of risk violation
            current_value (Optional[float]): Current risk value
            threshold_value (Optional[float]): Risk threshold that was exceeded
        """
        details = {}
        if risk_type:
            details['risk_type'] = risk_type
        if current_value is not None:
            details['current_value'] = current_value
        if threshold_value is not None:
            details['threshold_value'] = threshold_value
        
        super().__init__(message, "RISK_ERROR", details)
        self.risk_type = risk_type
        self.current_value = current_value
        self.threshold_value = threshold_value


class OrderExecutionError(TradingBotException):
    """
    Exception raised for order execution failures.
    
    This includes:
    - Order placement failures
    - Order cancellation issues
    - Insufficient balance errors
    - Invalid order parameters
    - Market condition issues
    """
    
    def __init__(self, message: str, order_type: Optional[str] = None, 
                 symbol: Optional[str] = None, side: Optional[str] = None,
                 quantity: Optional[float] = None, price: Optional[float] = None):
        """
        Initialize order execution error.
        
        Args:
            message (str): Error message
            order_type (Optional[str]): Type of order (market, limit, etc.)
            symbol (Optional[str]): Trading symbol
            side (Optional[str]): Order side (buy/sell)
            quantity (Optional[float]): Order quantity
            price (Optional[float]): Order price
        """
        details = {}
        if order_type:
            details['order_type'] = order_type
        if symbol:
            details['symbol'] = symbol
        if side:
            details['side'] = side
        if quantity is not None:
            details['quantity'] = quantity
        if price is not None:
            details['price'] = price
        
        super().__init__(message, "ORDER_ERROR", details)
        self.order_type = order_type
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price


class DataError(TradingBotException):
    """
    Exception raised for data-related errors.
    
    This includes:
    - Data fetch failures
    - Data validation errors
    - Missing historical data
    - Data format issues
    - Database connection problems
    """
    
    def __init__(self, message: str, data_source: Optional[str] = None, 
                 symbol: Optional[str] = None, timeframe: Optional[str] = None,
                 data_type: Optional[str] = None):
        """
        Initialize data error.
        
        Args:
            message (str): Error message
            data_source (Optional[str]): Source of the data
            symbol (Optional[str]): Trading symbol
            timeframe (Optional[str]): Data timeframe
            data_type (Optional[str]): Type of data (ohlcv, trades, etc.)
        """
        details = {}
        if data_source:
            details['data_source'] = data_source
        if symbol:
            details['symbol'] = symbol
        if timeframe:
            details['timeframe'] = timeframe
        if data_type:
            details['data_type'] = data_type
        
        super().__init__(message, "DATA_ERROR", details)
        self.data_source = data_source
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_type = data_type


class ValidationError(TradingBotException):
    """
    Exception raised for data validation failures.
    
    This includes:
    - Parameter validation errors
    - Data integrity issues
    - Format validation failures
    - Business rule violations
    """
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 field_value: Optional[Any] = None, 
                 validation_rule: Optional[str] = None):
        """
        Initialize validation error.
        
        Args:
            message (str): Error message
            field_name (Optional[str]): Name of the field that failed validation
            field_value (Optional[Any]): Value that failed validation
            validation_rule (Optional[str]): Validation rule that was violated
        """
        details = {}
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            details['field_value'] = str(field_value)
        if validation_rule:
            details['validation_rule'] = validation_rule
        
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule


class NetworkError(TradingBotException):
    """
    Exception raised for network-related errors.
    
    This includes:
    - Connection timeouts
    - DNS resolution failures
    - SSL/TLS errors
    - Network unavailability
    """
    
    def __init__(self, message: str, host: Optional[str] = None, 
                 port: Optional[int] = None, timeout: Optional[float] = None):
        """
        Initialize network error.
        
        Args:
            message (str): Error message
            host (Optional[str]): Host that couldn't be reached
            port (Optional[int]): Port number
            timeout (Optional[float]): Timeout value in seconds
        """
        details = {}
        if host:
            details['host'] = host
        if port:
            details['port'] = port
        if timeout:
            details['timeout'] = timeout
        
        super().__init__(message, "NETWORK_ERROR", details)
        self.host = host
        self.port = port
        self.timeout = timeout


class DatabaseError(TradingBotException):
    """
    Exception raised for database-related errors.
    
    This includes:
    - Connection failures
    - Query execution errors
    - Transaction failures
    - Schema issues
    """
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 table: Optional[str] = None, query: Optional[str] = None):
        """
        Initialize database error.
        
        Args:
            message (str): Error message
            operation (Optional[str]): Database operation (SELECT, INSERT, etc.)
            table (Optional[str]): Database table name
            query (Optional[str]): SQL query that failed
        """
        details = {}
        if operation:
            details['operation'] = operation
        if table:
            details['table'] = table
        if query:
            details['query'] = query[:200] + "..." if query and len(query) > 200 else query
        
        super().__init__(message, "DATABASE_ERROR", details)
        self.operation = operation
        self.table = table
        self.query = query


class NotificationError(TradingBotException):
    """
    Exception raised for notification service errors.
    
    This includes:
    - Message delivery failures
    - Service authentication errors
    - Rate limiting issues
    - Configuration problems
    """
    
    def __init__(self, message: str, service: Optional[str] = None, 
                 message_type: Optional[str] = None, recipient: Optional[str] = None):
        """
        Initialize notification error.
        
        Args:
            message (str): Error message
            service (Optional[str]): Notification service (telegram, discord, email)
            message_type (Optional[str]): Type of notification
            recipient (Optional[str]): Intended recipient
        """
        details = {}
        if service:
            details['service'] = service
        if message_type:
            details['message_type'] = message_type
        if recipient:
            details['recipient'] = recipient
        
        super().__init__(message, "NOTIFICATION_ERROR", details)
        self.service = service
        self.message_type = message_type
        self.recipient = recipient


# Utility functions for exception handling

def handle_exception_with_logging(logger, exception: Exception, context: str = "") -> None:
    """
    Handle exception with proper logging.
    
    Args:
        logger: Logger instance
        exception (Exception): Exception to handle
        context (str): Additional context information
    """
    if isinstance(exception, TradingBotException):
        error_dict = exception.to_dict()
        logger.error(f"❌ {context} - {error_dict}")
    else:
        logger.error(f"❌ {context} - Unexpected error: {type(exception).__name__}: {exception}")


def create_exception_from_api_error(api_response: Dict[str, Any], 
                                  exchange: str = "binance") -> ExchangeConnectionError:
    """
    Create an ExchangeConnectionError from API error response.
    
    Args:
        api_response (Dict[str, Any]): API error response
        exchange (str): Exchange name
        
    Returns:
        ExchangeConnectionError: Formatted exception
    """
    message = api_response.get('msg', 'Unknown API error')
    status_code = api_response.get('code')
    
    return ExchangeConnectionError(
        message=message,
        exchange=exchange,
        api_error_code=str(status_code) if status_code else None
    )


def is_recoverable_error(exception: Exception) -> bool:
    """
    Determine if an error is recoverable and the bot should continue.
    
    Args:
        exception (Exception): Exception to analyze
        
    Returns:
        bool: True if error is recoverable
    """
    # Network errors are usually temporary and recoverable
    if isinstance(exception, (NetworkError, ConnectionError, TimeoutError)):
        return True
    
    # Some exchange errors are recoverable (rate limits, temporary issues)
    if isinstance(exception, ExchangeConnectionError):
        # Rate limiting is recoverable
        if exception.status_code in [429, 418]:  # Too Many Requests, I'm a teapot
            return True
        # Server errors might be temporary
        if exception.status_code and 500 <= exception.status_code < 600:
            return True
    
    # Data errors might be temporary
    if isinstance(exception, DataError):
        return True
    
    # Configuration and initialization errors are usually not recoverable
    if isinstance(exception, (ConfigurationError, InitializationError)):
        return False
    
    # Risk management errors should stop trading
    if isinstance(exception, RiskManagementError):
        return False
    
    # For other exceptions, assume they might be recoverable
    return True