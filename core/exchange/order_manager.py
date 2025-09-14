"""
Order Management System
======================

Advanced order management system for the trading bot that handles order
placement, cancellation, tracking, and validation. Provides a high-level
interface for order operations with comprehensive error handling and logging.

Key Features:
- Support for all order types (MARKET, LIMIT, STOP_MARKET, etc.)
- Order validation and risk checks
- Retry logic with exponential backoff
- Order tracking and status monitoring
- Comprehensive logging and error handling
- Integration with Binance Futures API

Author: dat-ns
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_DOWN
import uuid
import time

from .base_exchange import (
    BaseExchange, OrderSide, OrderType, PositionSide,
    OrderInfo, Position, Balance
)
from core.utils.exceptions import (
    OrderExecutionError, ExchangeConnectionError, 
    ValidationError, RiskManagementError
)


class OrderStatus(Enum):
    """Extended order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderPriority(Enum):
    """Order priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class OrderRequest:
    """Order request structure with validation and metadata."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    client_order_id: Optional[str] = None
    priority: OrderPriority = OrderPriority.NORMAL
    
    # Risk management parameters
    max_slippage: Optional[float] = None
    max_position_size: Optional[float] = None
    
    # Metadata
    strategy_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.client_order_id:
            self.client_order_id = f"bot_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
        
        # Validate required fields
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValidationError("Symbol is required and must be a string")
        
        if self.quantity <= 0:
            raise ValidationError("Quantity must be positive")
        
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
            if self.price is None or self.price <= 0:
                raise ValidationError(f"Price is required for {self.order_type.value} orders")
        
        if self.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
            if self.stop_price is None or self.stop_price <= 0:
                raise ValidationError(f"Stop price is required for {self.order_type.value} orders")


@dataclass
class OrderTracker:
    """Order tracking information."""
    order_request: OrderRequest
    order_info: Optional[OrderInfo] = None
    status: OrderStatus = OrderStatus.PENDING
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    error_messages: List[str] = field(default_factory=list)
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: Optional[float] = None
    total_fees: float = 0.0
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_completed(self) -> bool:
        """Check if order is completed (filled or cancelled)."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.order_request.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.order_request.quantity) * 100


class OrderManager:
    """
    Advanced order management system for trading bot.
    
    Handles order placement, cancellation, tracking, and validation with
    comprehensive error handling and retry logic.
    """
    
    def __init__(self, exchange: BaseExchange, settings):
        """
        Initialize the order manager.
        
        Args:
            exchange (BaseExchange): Exchange interface for order execution
            settings: Configuration settings
        """
        self.exchange = exchange
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Order tracking
        self.active_orders: Dict[str, OrderTracker] = {}
        self.order_history: List[OrderTracker] = []
        
        # Configuration
        self.max_retries = getattr(settings, 'ORDER_MAX_RETRIES', 3)
        self.retry_delay = getattr(settings, 'ORDER_RETRY_DELAY', 1.0)
        self.order_timeout = getattr(settings, 'ORDER_TIMEOUT_SECONDS', 300)
        self.enable_test_orders = getattr(settings, 'ENABLE_TEST_ORDERS', True)
        
        # Risk management limits
        self.max_orders_per_symbol = getattr(settings, 'MAX_ORDERS_PER_SYMBOL', 10)
        self.max_total_orders = getattr(settings, 'MAX_TOTAL_ORDERS', 50)
        self.min_order_value = getattr(settings, 'MIN_ORDER_VALUE_USDT', 10.0)
        
        # Statistics
        self.stats = {
            'total_orders_placed': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'cancelled_orders': 0,
            'total_fees_paid': 0.0,
            'last_order_time': None
        }
        
        self.logger.info("📋 Order Manager initialized")
    
    async def place_order(self, order_request: OrderRequest, 
                         validate_only: bool = False) -> OrderTracker:
        """
        Place a new order with comprehensive validation and error handling.
        
        Args:
            order_request (OrderRequest): Order details and parameters
            validate_only (bool): If True, only validate without placing order
            
        Returns:
            OrderTracker: Order tracking information
            
        Raises:
            ValidationError: If order validation fails
            OrderExecutionError: If order placement fails
        """
        try:
            self.logger.info(f"📝 Placing {order_request.side.value} {order_request.order_type.value} order: "
                           f"{order_request.quantity} {order_request.symbol} @ {order_request.price}")
            
            # Create order tracker
            tracker = OrderTracker(order_request=order_request)
            
            # Comprehensive validation
            await self._validate_order(order_request)
            
            if validate_only:
                self.logger.info("✅ Order validation successful (validation only)")
                return tracker
            
            # Check rate limits and capacity
            await self._check_order_limits(order_request)
            
            # Test order if enabled
            if self.enable_test_orders:
                await self._test_order(order_request)
            
            # Place the actual order with retry logic
            order_info = await self._place_order_with_retry(order_request, tracker)
            
            # Update tracker with order info
            tracker.order_info = order_info
            tracker.status = OrderStatus.SUBMITTED
            tracker.remaining_quantity = order_request.quantity
            
            # Add to active orders
            self.active_orders[order_info.client_order_id] = tracker
            
            # Update statistics
            self.stats['total_orders_placed'] += 1
            self.stats['last_order_time'] = datetime.now(timezone.utc)
            
            self.logger.info(f"✅ Order placed successfully: ID {order_info.order_id}")
            return tracker
            
        except ValidationError:
            self.logger.error(f"❌ Order validation failed: {order_request.symbol}")
            raise
        except OrderExecutionError:
            self.logger.error(f"❌ Order execution failed: {order_request.symbol}")
            self.stats['failed_orders'] += 1
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error placing order: {e}")
            self.stats['failed_orders'] += 1
            raise OrderExecutionError(f"Unexpected error placing order: {e}") from e
    
    async def cancel_order(self, client_order_id: str, reason: str = "Manual cancellation") -> bool:
        """
        Cancel an existing order.
        
        Args:
            client_order_id (str): Client order ID to cancel
            reason (str): Reason for cancellation
            
        Returns:
            bool: True if cancellation successful
            
        Raises:
            OrderExecutionError: If cancellation fails
        """
        try:
            tracker = self.active_orders.get(client_order_id)
            if not tracker:
                self.logger.warning(f"⚠️ Order not found for cancellation: {client_order_id}")
                return False
            
            if not tracker.is_active:
                self.logger.warning(f"⚠️ Order already completed: {client_order_id}")
                return False
            
            self.logger.info(f"🚫 Cancelling order: {client_order_id} - Reason: {reason}")
            
            # Cancel on exchange
            success = await self.exchange.cancel_order(
                symbol=tracker.order_request.symbol,
                order_id=tracker.order_info.order_id
            )
            
            if success:
                # Update tracker
                tracker.status = OrderStatus.CANCELLED
                tracker.error_messages.append(f"Cancelled: {reason}")
                
                # Move to history
                self._move_to_history(client_order_id)
                
                # Update statistics
                self.stats['cancelled_orders'] += 1
                
                self.logger.info(f"✅ Order cancelled successfully: {client_order_id}")
                return True
            else:
                self.logger.error(f"❌ Failed to cancel order: {client_order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error cancelling order {client_order_id}: {e}")
            raise OrderExecutionError(f"Error cancelling order: {e}") from e
    
    async def cancel_all_orders(self, symbol: Optional[str] = None, 
                              reason: str = "Bulk cancellation") -> int:
        """
        Cancel all active orders.
        
        Args:
            symbol (Optional[str]): Symbol to cancel orders for (None for all)
            reason (str): Reason for cancellation
            
        Returns:
            int: Number of orders cancelled
        """
        try:
            self.logger.info(f"🚫 Cancelling all orders" + (f" for {symbol}" if symbol else ""))
            
            # Filter orders to cancel
            orders_to_cancel = []
            for client_order_id, tracker in self.active_orders.items():
                if tracker.is_active:
                    if symbol is None or tracker.order_request.symbol == symbol:
                        orders_to_cancel.append(client_order_id)
            
            if not orders_to_cancel:
                self.logger.info("ℹ️ No active orders to cancel")
                return 0
            
            # Cancel orders concurrently
            cancelled_count = 0
            tasks = []
            for client_order_id in orders_to_cancel:
                task = self.cancel_order(client_order_id, reason)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful cancellations
            for result in results:
                if isinstance(result, bool) and result:
                    cancelled_count += 1
                elif isinstance(result, Exception):
                    self.logger.error(f"❌ Error in bulk cancellation: {result}")
            
            self.logger.info(f"✅ Cancelled {cancelled_count} orders")
            return cancelled_count
            
        except Exception as e:
            self.logger.error(f"❌ Error in bulk cancellation: {e}")
            raise OrderExecutionError(f"Error in bulk cancellation: {e}") from e
    
    async def get_order_status(self, client_order_id: str) -> Optional[OrderTracker]:
        """
        Get current status of an order.
        
        Args:
            client_order_id (str): Client order ID
            
        Returns:
            Optional[OrderTracker]: Order tracking information or None if not found
        """
        try:
            tracker = self.active_orders.get(client_order_id)
            if not tracker:
                # Check history
                for hist_tracker in self.order_history:
                    if hist_tracker.order_request.client_order_id == client_order_id:
                        return hist_tracker
                return None
            
            # Update from exchange if active
            if tracker.is_active and tracker.order_info:
                try:
                    order_info = await self.exchange.get_order(
                        symbol=tracker.order_request.symbol,
                        order_id=tracker.order_info.order_id
                    )
                    
                    if order_info:
                        await self._update_tracker_from_order_info(tracker, order_info)
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to update order status from exchange: {e}")
            
            return tracker
            
        except Exception as e:
            self.logger.error(f"❌ Error getting order status: {e}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderTracker]:
        """
        Get all open/active orders.
        
        Args:
            symbol (Optional[str]): Filter by symbol
            
        Returns:
            List[OrderTracker]: List of active order trackers
        """
        active_orders = []
        for tracker in self.active_orders.values():
            if tracker.is_active:
                if symbol is None or tracker.order_request.symbol == symbol:
                    active_orders.append(tracker)
        
        return active_orders
    
    async def sync_orders_with_exchange(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronize order status with exchange.
        
        Args:
            symbol (Optional[str]): Symbol to sync (None for all)
            
        Returns:
            Dict[str, Any]: Sync results
        """
        try:
            self.logger.info(f"🔄 Syncing orders with exchange" + (f" for {symbol}" if symbol else ""))
            
            # Get open orders from exchange
            exchange_orders = await self.exchange.get_open_orders(symbol)
            exchange_order_ids = {order.client_order_id: order for order in exchange_orders}
            
            sync_results = {
                'updated': 0,
                'cancelled': 0,
                'new_found': 0,
                'errors': []
            }
            
            # Update tracked orders
            for client_order_id, tracker in list(self.active_orders.items()):
                if symbol and tracker.order_request.symbol != symbol:
                    continue
                
                try:
                    if client_order_id in exchange_order_ids:
                        # Order still exists on exchange, update it
                        exchange_order = exchange_order_ids[client_order_id]
                        await self._update_tracker_from_order_info(tracker, exchange_order)
                        sync_results['updated'] += 1
                    else:
                        # Order not found on exchange, mark as cancelled
                        if tracker.is_active:
                            tracker.status = OrderStatus.CANCELLED
                            tracker.error_messages.append("Order not found on exchange during sync")
                            self._move_to_history(client_order_id)
                            sync_results['cancelled'] += 1
                            
                except Exception as e:
                    error_msg = f"Error syncing order {client_order_id}: {e}"
                    self.logger.warning(f"⚠️ {error_msg}")
                    sync_results['errors'].append(error_msg)
            
            # Check for orders on exchange that we're not tracking
            for client_order_id, exchange_order in exchange_order_ids.items():
                if client_order_id not in self.active_orders:
                    self.logger.warning(f"⚠️ Found untracked order on exchange: {client_order_id}")
                    sync_results['new_found'] += 1
            
            self.logger.info(f"✅ Order sync complete: {sync_results}")
            return sync_results
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing orders: {e}")
            return {'error': str(e)}
    
    async def cleanup_completed_orders(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed orders from memory.
        
        Args:
            max_age_hours (int): Maximum age in hours for keeping completed orders
            
        Returns:
            int: Number of orders cleaned up
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            
            # Clean up completed orders in active tracking
            cleaned_active = 0
            for client_order_id in list(self.active_orders.keys()):
                tracker = self.active_orders[client_order_id]
                if tracker.is_completed and tracker.order_request.created_at < cutoff_time:
                    self._move_to_history(client_order_id)
                    cleaned_active += 1
            
            # Clean up old history
            cleaned_history = 0
            self.order_history = [
                tracker for tracker in self.order_history
                if tracker.order_request.created_at >= cutoff_time
            ]
            cleaned_history = len(self.order_history) - len([
                tracker for tracker in self.order_history
                if tracker.order_request.created_at >= cutoff_time
            ])
            
            total_cleaned = cleaned_active + cleaned_history
            if total_cleaned > 0:
                self.logger.info(f"🧹 Cleaned up {total_cleaned} old orders")
            
            return total_cleaned
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up orders: {e}")
            return 0
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """
        Get order management statistics.
        
        Returns:
            Dict[str, Any]: Statistics information
        """
        active_count = len(self.active_orders)
        history_count = len(self.order_history)
        
        # Calculate success rate
        total_completed = self.stats['successful_orders'] + self.stats['failed_orders']
        success_rate = (self.stats['successful_orders'] / total_completed * 100) if total_completed > 0 else 0
        
        # Orders by symbol
        orders_by_symbol = {}
        for tracker in self.active_orders.values():
            symbol = tracker.order_request.symbol
            orders_by_symbol[symbol] = orders_by_symbol.get(symbol, 0) + 1
        
        return {
            'active_orders': active_count,
            'order_history_count': history_count,
            'total_orders_placed': self.stats['total_orders_placed'],
            'successful_orders': self.stats['successful_orders'],
            'failed_orders': self.stats['failed_orders'],
            'cancelled_orders': self.stats['cancelled_orders'],
            'success_rate_percent': round(success_rate, 2),
            'total_fees_paid': self.stats['total_fees_paid'],
            'last_order_time': self.stats['last_order_time'],
            'orders_by_symbol': orders_by_symbol,
            'configuration': {
                'max_retries': self.max_retries,
                'retry_delay': self.retry_delay,
                'order_timeout': self.order_timeout,
                'max_orders_per_symbol': self.max_orders_per_symbol,
                'max_total_orders': self.max_total_orders,
                'min_order_value': self.min_order_value
            }
        }
    
    # Private Helper Methods
    
    async def _validate_order(self, order_request: OrderRequest) -> None:
        """Comprehensive order validation."""
        try:
            # Basic validation
            if not order_request.symbol or not isinstance(order_request.symbol, str):
                raise ValidationError("Invalid symbol")
            
            if order_request.quantity <= 0:
                raise ValidationError("Quantity must be positive")
            
            # Symbol validation
            if not self.exchange.validate_symbol(order_request.symbol):
                raise ValidationError(f"Invalid or unsupported symbol: {order_request.symbol}")
            
            # Order value validation
            if order_request.price and order_request.price > 0:
                order_value = order_request.quantity * order_request.price
                if order_value < self.min_order_value:
                    raise ValidationError(f"Order value {order_value} below minimum {self.min_order_value}")
            
            # Risk management checks
            if order_request.max_position_size:
                current_position = await self.exchange.get_position(order_request.symbol)
                if current_position and abs(current_position.size) >= order_request.max_position_size:
                    raise RiskManagementError(f"Position size limit exceeded: {current_position.size}")
            
            # Price validation
            if order_request.price and order_request.price <= 0:
                raise ValidationError("Price must be positive")
            
            if order_request.stop_price and order_request.stop_price <= 0:
                raise ValidationError("Stop price must be positive")
            
            # Stop order logic validation
            if order_request.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
                if order_request.side == OrderSide.BUY and order_request.price:
                    if order_request.stop_price <= order_request.price:
                        raise ValidationError("Buy stop price must be above current price")
                elif order_request.side == OrderSide.SELL and order_request.price:
                    if order_request.stop_price >= order_request.price:
                        raise ValidationError("Sell stop price must be below current price")
            
            self.logger.debug(f"✅ Order validation passed: {order_request.symbol}")
            
        except (ValidationError, RiskManagementError):
            raise
        except Exception as e:
            raise ValidationError(f"Order validation error: {e}") from e
    
    async def _check_order_limits(self, order_request: OrderRequest) -> None:
        """Check order limits and capacity."""
        # Check total order limit
        if len(self.active_orders) >= self.max_total_orders:
            raise RiskManagementError(f"Maximum total orders limit reached: {self.max_total_orders}")
        
        # Check per-symbol limit
        symbol_orders = sum(1 for tracker in self.active_orders.values() 
                           if tracker.order_request.symbol == order_request.symbol)
        
        if symbol_orders >= self.max_orders_per_symbol:
            raise RiskManagementError(
                f"Maximum orders per symbol limit reached for {order_request.symbol}: {self.max_orders_per_symbol}")
    
    async def _test_order(self, order_request: OrderRequest) -> None:
        """Test order placement without execution."""
        try:
            await self.exchange.place_test_order(
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=order_request.price,
                stop_price=order_request.stop_price
            )
            self.logger.debug(f"✅ Test order successful: {order_request.symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Test order failed: {e}")
            raise OrderExecutionError(f"Test order failed: {e}") from e
    
    async def _place_order_with_retry(self, order_request: OrderRequest, 
                                    tracker: OrderTracker) -> OrderInfo:
        """Place order with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                tracker.attempts = attempt + 1
                tracker.last_attempt = datetime.now(timezone.utc)
                
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    self.logger.warning(f"⏳ Retrying order placement in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                
                # Place the order
                order_info = await self.exchange.place_order(
                    symbol=order_request.symbol,
                    side=order_request.side,
                    order_type=order_request.order_type,
                    quantity=order_request.quantity,
                    price=order_request.price,
                    stop_price=order_request.stop_price,
                    time_in_force=order_request.time_in_force,
                    reduce_only=order_request.reduce_only,
                    client_order_id=order_request.client_order_id
                )
                
                self.logger.info(f"✅ Order placed successfully on attempt {attempt + 1}")
                return order_info
                
            except Exception as e:
                last_exception = e
                error_msg = f"Attempt {attempt + 1} failed: {e}"
                tracker.error_messages.append(error_msg)
                self.logger.warning(f"⚠️ {error_msg}")
                
                # Don't retry for certain errors
                if isinstance(e, ValidationError):
                    break
        
        # All attempts failed
        tracker.status = OrderStatus.FAILED
        raise OrderExecutionError(
            f"Order placement failed after {self.max_retries + 1} attempts. Last error: {last_exception}"
        ) from last_exception
    
    async def _update_tracker_from_order_info(self, tracker: OrderTracker, 
                                            order_info: OrderInfo) -> None:
        """Update order tracker with latest order information."""
        try:
            tracker.order_info = order_info
            tracker.filled_quantity = order_info.filled_quantity
            tracker.remaining_quantity = order_info.remaining_quantity
            tracker.average_price = order_info.average_price
            
            # Update status based on order status
            status_mapping = {
                'NEW': OrderStatus.SUBMITTED,
                'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
                'FILLED': OrderStatus.FILLED,
                'CANCELED': OrderStatus.CANCELLED,
                'CANCELLED': OrderStatus.CANCELLED,
                'REJECTED': OrderStatus.REJECTED,
                'EXPIRED': OrderStatus.EXPIRED
            }
            
            new_status = status_mapping.get(order_info.status.upper(), OrderStatus.SUBMITTED)
            
            # Check if status changed
            if new_status != tracker.status:
                old_status = tracker.status
                tracker.status = new_status
                
                self.logger.info(f"📊 Order status changed: {tracker.order_request.client_order_id} "
                               f"{old_status.value} -> {new_status.value}")
                
                # Move to history if completed
                if tracker.is_completed:
                    self._move_to_history(tracker.order_request.client_order_id)
                    if new_status == OrderStatus.FILLED:
                        self.stats['successful_orders'] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Error updating tracker: {e}")
    
    def _move_to_history(self, client_order_id: str) -> None:
        """Move order from active tracking to history."""
        if client_order_id in self.active_orders:
            tracker = self.active_orders.pop(client_order_id)
            self.order_history.append(tracker)
            
            # Keep history size manageable
            if len(self.order_history) > 1000:
                self.order_history = self.order_history[-1000:]
    
    async def emergency_cancel_all(self) -> int:
        """Emergency cancellation of all orders."""
        try:
            self.logger.error("🚨 EMERGENCY: Cancelling all orders!")
            
            # Cancel on exchange first
            cancelled_on_exchange = await self.exchange.cancel_all_orders()
            
            # Update local tracking
            cancelled_locally = 0
            for client_order_id, tracker in list(self.active_orders.items()):
                if tracker.is_active:
                    tracker.status = OrderStatus.CANCELLED
                    tracker.error_messages.append("Emergency cancellation")
                    self._move_to_history(client_order_id)
                    cancelled_locally += 1
            
            self.logger.error(f"🚨 Emergency cancellation complete: {cancelled_locally} orders")
            return cancelled_locally
            
        except Exception as e:
            self.logger.error(f"❌ Error in emergency cancellation: {e}")
            return 0