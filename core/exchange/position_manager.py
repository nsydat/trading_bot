"""
Position Management System
=========================

Advanced position management system for futures trading that handles position
opening, closing, modification, and monitoring. Provides comprehensive position
tracking with risk management and P&L calculation.

Key Features:
- Position opening with entry strategies
- Position closing with exit strategies
- Position modification (stop loss, take profit)
- Real-time P&L tracking
- Risk management integration
- Position size calculation and validation
- Multi-timeframe position analysis
- Advanced order management integration

Author: dat-ns
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_DOWN
import math

from .base_exchange import (
    BaseExchange, OrderSide, OrderType, PositionSide,
    OrderInfo, Position, Balance
)
from .order_manager import OrderManager, OrderRequest, OrderTracker, OrderPriority
from core.utils.exceptions import (
    PositionError, RiskManagementError, ValidationError,
    OrderExecutionError, ExchangeConnectionError
)


class PositionStatus(Enum):
    """Position status enumeration."""
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    PARTIALLY_FILLED = "partially_filled"
    ERROR = "error"


class ExitReason(Enum):
    """Position exit reasons."""
    MANUAL = "manual"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STRATEGY_SIGNAL = "strategy_signal"
    RISK_MANAGEMENT = "risk_management"
    EMERGENCY = "emergency"
    TIMEOUT = "timeout"


@dataclass
class PositionConfig:
    """Position configuration parameters."""
    symbol: str
    side: PositionSide
    size: float
    entry_price: Optional[float] = None
    
    # Risk management
    stop_loss_price: Optional[float] = None
    stop_loss_percent: Optional[float] = None
    take_profit_price: Optional[float] = None
    take_profit_percent: Optional[float] = None
    max_loss_percent: float = 5.0
    
    # Position timing
    max_hold_time: Optional[timedelta] = None
    entry_timeout: timedelta = timedelta(minutes=5)
    
    # Order configuration
    entry_order_type: OrderType = OrderType.MARKET
    exit_order_type: OrderType = OrderType.MARKET
    reduce_only: bool = True
    
    # Metadata
    strategy_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class PositionMetrics:
    """Position performance metrics."""
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    max_drawdown: float = 0.0
    hold_time: timedelta = field(default_factory=lambda: timedelta())
    entry_slippage: float = 0.0
    exit_slippage: float = 0.0


@dataclass
class ManagedPosition:
    """Managed position with full tracking and control."""
    config: PositionConfig
    position_id: str
    status: PositionStatus = PositionStatus.OPENING
    
    # Order tracking
    entry_orders: List[OrderTracker] = field(default_factory=list)
    exit_orders: List[OrderTracker] = field(default_factory=list)
    stop_loss_order: Optional[OrderTracker] = None
    take_profit_order: Optional[OrderTracker] = None
    
    # Position data
    exchange_position: Optional[Position] = None
    filled_size: float = 0.0
    average_entry_price: Optional[float] = None
    current_price: Optional[float] = None
    
    # Metrics
    metrics: PositionMetrics = field(default_factory=PositionMetrics)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Status tracking
    error_messages: List[str] = field(default_factory=list)
    exit_reason: Optional[ExitReason] = None
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.config.side == PositionSide.LONG
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.config.side == PositionSide.SHORT
    
    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.status in [PositionStatus.OPEN, PositionStatus.PARTIALLY_FILLED]
    
    @property
    def is_active(self) -> bool:
        """Check if position is active (opening, open, or closing)."""
        return self.status in [
            PositionStatus.OPENING, PositionStatus.OPEN, 
            PositionStatus.PARTIALLY_FILLED, PositionStatus.CLOSING
        ]
    
    @property
    def fill_percentage(self) -> float:
        """Calculate position fill percentage."""
        if self.config.size == 0:
            return 0.0
        return (self.filled_size / self.config.size) * 100
    
    @property
    def remaining_size(self) -> float:
        """Calculate remaining size to fill."""
        return max(0.0, self.config.size - self.filled_size)


class PositionManager:
    """
    Advanced position management system for futures trading.
    
    Handles complete position lifecycle including opening, monitoring,
    modification, and closing with comprehensive risk management.
    """
    
    def __init__(self, exchange: BaseExchange, order_manager: OrderManager, settings):
        """
        Initialize the position manager.
        
        Args:
            exchange (BaseExchange): Exchange interface
            order_manager (OrderManager): Order management system
            settings: Configuration settings
        """
        self.exchange = exchange
        self.order_manager = order_manager
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Position tracking
        self.active_positions: Dict[str, ManagedPosition] = {}
        self.position_history: List[ManagedPosition] = []
        
        # Configuration
        self.max_positions = getattr(settings, 'MAX_POSITIONS', 10)
        self.max_position_value = getattr(settings, 'MAX_POSITION_VALUE_USDT', 10000.0)
        self.default_max_loss_percent = getattr(settings, 'DEFAULT_MAX_LOSS_PERCENT', 2.0)
        self.position_timeout_hours = getattr(settings, 'POSITION_TIMEOUT_HOURS', 24)
        self.enable_auto_stop_loss = getattr(settings, 'ENABLE_AUTO_STOP_LOSS', True)
        self.enable_auto_take_profit = getattr(settings, 'ENABLE_AUTO_TAKE_PROFIT', True)
        
        # Risk management
        self.max_risk_per_trade = getattr(settings, 'MAX_RISK_PER_TRADE_PERCENT', 1.0)
        self.max_total_risk = getattr(settings, 'MAX_TOTAL_RISK_PERCENT', 5.0)
        self.correlation_limit = getattr(settings, 'CORRELATION_LIMIT', 0.7)
        
        # Statistics
        self.stats = {
            'total_positions_opened': 0,
            'profitable_positions': 0,
            'losing_positions': 0,
            'total_realized_pnl': 0.0,
            'total_fees_paid': 0.0,
            'best_trade_pnl': 0.0,
            'worst_trade_pnl': 0.0,
            'average_hold_time': timedelta(),
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'last_position_time': None
        }
        
        self.logger.info("📊 Position Manager initialized")
    
    async def open_position(self, config: PositionConfig) -> ManagedPosition:
        """
        Open a new position with comprehensive validation and risk management.
        
        Args:
            config (PositionConfig): Position configuration
            
        Returns:
            ManagedPosition: Managed position object
            
        Raises:
            PositionError: If position opening fails
            RiskManagementError: If risk limits exceeded
        """
        try:
            self.logger.info(f"📈 Opening {config.side.value} position: "
                           f"{config.size} {config.symbol} @ {config.entry_price}")
            
            # Validate position configuration
            await self._validate_position_config(config)
            
            # Check risk management limits
            await self._check_risk_limits(config)
            
            # Create managed position
            position_id = f"{config.symbol}_{config.side.value}_{int(datetime.now().timestamp())}"
            managed_position = ManagedPosition(
                config=config,
                position_id=position_id,
                status=PositionStatus.OPENING
            )
            
            # Add to active positions
            self.active_positions[position_id] = managed_position
            
            # Calculate position size and validate
            validated_size = await self._calculate_position_size(config)
            if validated_size != config.size:
                self.logger.warning(f"⚠️ Position size adjusted: {config.size} -> {validated_size}")
                config.size = validated_size
            
            # Create entry order
            entry_order_request = self._create_entry_order_request(config)
            
            try:
                # Place entry order
                entry_tracker = await self.order_manager.place_order(
                    entry_order_request, 
                    validate_only=False
                )
                
                managed_position.entry_orders.append(entry_tracker)
                
                # Set up stop loss and take profit orders if configured
                if self.enable_auto_stop_loss and (config.stop_loss_price or config.stop_loss_percent):
                    await self._setup_stop_loss(managed_position)
                
                if self.enable_auto_take_profit and (config.take_profit_price or config.take_profit_percent):
                    await self._setup_take_profit(managed_position)
                
                # Update statistics
                self.stats['total_positions_opened'] += 1
                self.stats['last_position_time'] = datetime.now(timezone.utc)
                
                self.logger.info(f"✅ Position opening initiated: {position_id}")
                
                # Start monitoring the position
                asyncio.create_task(self._monitor_position(managed_position))
                
                return managed_position
                
            except Exception as e:
                # Clean up failed position
                self.active_positions.pop(position_id, None)
                managed_position.status = PositionStatus.ERROR
                managed_position.error_messages.append(f"Failed to place entry order: {e}")
                raise PositionError(f"Failed to open position: {e}") from e
            
        except (ValidationError, RiskManagementError):
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error opening position: {e}")
            raise PositionError(f"Unexpected error opening position: {e}") from e
    
    async def close_position(self, position_id: str, reason: ExitReason = ExitReason.MANUAL,
                           partial_close_percent: Optional[float] = None) -> bool:
        """
        Close an existing position.
        
        Args:
            position_id (str): Position ID to close
            reason (ExitReason): Reason for closing
            partial_close_percent (Optional[float]): Percentage to close (None for full close)
            
        Returns:
            bool: True if close order placed successfully
            
        Raises:
            PositionError: If position closing fails
        """
        try:
            managed_position = self.active_positions.get(position_id)
            if not managed_position:
                raise PositionError(f"Position not found: {position_id}")
            
            if not managed_position.is_open:
                self.logger.warning(f"⚠️ Position not open for closing: {position_id}")
                return False
            
            close_size = managed_position.filled_size
            if partial_close_percent:
                close_size = managed_position.filled_size * (partial_close_percent / 100)
                close_size = max(0.1, close_size)  # Minimum close size
            
            self.logger.info(f"📉 Closing position {position_id}: {close_size} units - Reason: {reason.value}")
            
            # Update position status
            managed_position.status = PositionStatus.CLOSING
            managed_position.exit_reason = reason
            managed_position.last_update = datetime.now(timezone.utc)
            
            # Create exit order (opposite side)
            exit_side = OrderSide.SELL if managed_position.is_long else OrderSide.BUY
            
            exit_order_request = OrderRequest(
                symbol=managed_position.config.symbol,
                side=exit_side,
                order_type=managed_position.config.exit_order_type,
                quantity=close_size,
                price=managed_position.config.entry_price,  # Will be adjusted for market orders
                reduce_only=True,
                priority=OrderPriority.HIGH,
                strategy_name=managed_position.config.strategy_name,
                tags=managed_position.config.tags + [f"exit_{reason.value}"],
                notes=f"Position close: {reason.value}"
            )
            
            # Place exit order
            exit_tracker = await self.order_manager.place_order(exit_order_request)
            managed_position.exit_orders.append(exit_tracker)
            
            # Cancel stop loss and take profit orders if full close
            if not partial_close_percent:
                await self._cancel_protective_orders(managed_position)
            
            self.logger.info(f"✅ Position close order placed: {position_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error closing position {position_id}: {e}")
            if position_id in self.active_positions:
                self.active_positions[position_id].error_messages.append(f"Close error: {e}")
            raise PositionError(f"Failed to close position: {e}") from e
    
    async def modify_position(self, position_id: str, 
                            new_stop_loss: Optional[float] = None,
                            new_take_profit: Optional[float] = None) -> bool:
        """
        Modify position stop loss or take profit levels.
        
        Args:
            position_id (str): Position ID to modify
            new_stop_loss (Optional[float]): New stop loss price
            new_take_profit (Optional[float]): New take profit price
            
        Returns:
            bool: True if modification successful
        """
        try:
            managed_position = self.active_positions.get(position_id)
            if not managed_position:
                raise PositionError(f"Position not found: {position_id}")
            
            if not managed_position.is_open:
                raise PositionError(f"Position not open for modification: {position_id}")
            
            self.logger.info(f"🔧 Modifying position {position_id}")
            
            modified = False
            
            # Modify stop loss
            if new_stop_loss is not None:
                # Cancel existing stop loss
                if managed_position.stop_loss_order:
                    await self.order_manager.cancel_order(
                        managed_position.stop_loss_order.order_request.client_order_id,
                        "Stop loss modification"
                    )
                
                # Create new stop loss
                managed_position.config.stop_loss_price = new_stop_loss
                await self._setup_stop_loss(managed_position)
                modified = True
                self.logger.info(f"✅ Stop loss updated: {new_stop_loss}")
            
            # Modify take profit
            if new_take_profit is not None:
                # Cancel existing take profit
                if managed_position.take_profit_order:
                    await self.order_manager.cancel_order(
                        managed_position.take_profit_order.order_request.client_order_id,
                        "Take profit modification"
                    )
                
                # Create new take profit
                managed_position.config.take_profit_price = new_take_profit
                await self._setup_take_profit(managed_position)
                modified = True
                self.logger.info(f"✅ Take profit updated: {new_take_profit}")
            
            if modified:
                managed_position.last_update = datetime.now(timezone.utc)
                self.logger.info(f"✅ Position modified successfully: {position_id}")
            
            return modified
            
        except Exception as e:
            self.logger.error(f"❌ Error modifying position {position_id}: {e}")
            raise PositionError(f"Failed to modify position: {e}") from e
    
    async def get_position_info(self, position_id: str) -> Optional[ManagedPosition]:
        """
        Get comprehensive position information.
        
        Args:
            position_id (str): Position ID
            
        Returns:
            Optional[ManagedPosition]: Position information or None if not found
        """
        managed_position = self.active_positions.get(position_id)
        if not managed_position:
            # Check history
            for hist_position in self.position_history:
                if hist_position.position_id == position_id:
                    return hist_position
            return None
        
        # Update position with latest data
        await self._update_position_metrics(managed_position)
        return managed_position
    
    async def get_all_positions(self, symbol: Optional[str] = None, 
                              include_closed: bool = False) -> List[ManagedPosition]:
        """
        Get all positions with optional filtering.
        
        Args:
            symbol (Optional[str]): Filter by symbol
            include_closed (bool): Include closed positions
            
        Returns:
            List[ManagedPosition]: List of positions
        """
        positions = []
        
        # Active positions
        for managed_position in self.active_positions.values():
            if symbol is None or managed_position.config.symbol == symbol:
                await self._update_position_metrics(managed_position)
                positions.append(managed_position)
        
        # Closed positions if requested
        if include_closed:
            for managed_position in self.position_history:
                if symbol is None or managed_position.config.symbol == symbol:
                    positions.append(managed_position)
        
        return positions
    
    async def close_all_positions(self, symbol: Optional[str] = None, 
                                reason: ExitReason = ExitReason.EMERGENCY) -> int:
        """
        Close all active positions.
        
        Args:
            symbol (Optional[str]): Symbol to close positions for (None for all)
            reason (ExitReason): Reason for closing
            
        Returns:
            int: Number of positions closed
        """
        try:
            self.logger.warning(f"🚨 Closing all positions - Reason: {reason.value}")
            
            positions_to_close = []
            for position_id, managed_position in self.active_positions.items():
                if managed_position.is_open:
                    if symbol is None or managed_position.config.symbol == symbol:
                        positions_to_close.append(position_id)
            
            if not positions_to_close:
                self.logger.info("ℹ️ No active positions to close")
                return 0
            
            # Close positions concurrently
            close_tasks = []
            for position_id in positions_to_close:
                task = self.close_position(position_id, reason)
                close_tasks.append(task)
            
            results = await asyncio.gather(*close_tasks, return_exceptions=True)
            
            # Count successful closes
            closed_count = 0
            for i, result in enumerate(results):
                if isinstance(result, bool) and result:
                    closed_count += 1
                elif isinstance(result, Exception):
                    position_id = positions_to_close[i]
                    self.logger.error(f"❌ Failed to close position {position_id}: {result}")
            
            self.logger.warning(f"🚨 Closed {closed_count} positions")
            return closed_count
            
        except Exception as e:
            self.logger.error(f"❌ Error closing all positions: {e}")
            return 0
    
    async def sync_positions_with_exchange(self) -> Dict[str, Any]:
        """
        Synchronize positions with exchange data.
        
        Returns:
            Dict[str, Any]: Sync results
        """
        try:
            self.logger.info("🔄 Syncing positions with exchange")
            
            # Get positions from exchange
            exchange_positions = await self.exchange.get_positions()
            exchange_position_map = {pos.symbol: pos for pos in exchange_positions}
            
            sync_results = {
                'updated': 0,
                'new_found': 0,
                'closed': 0,
                'errors': []
            }
            
            # Update managed positions
            for position_id, managed_position in list(self.active_positions.items()):
                try:
                    symbol = managed_position.config.symbol
                    
                    if symbol in exchange_position_map:
                        exchange_pos = exchange_position_map[symbol]
                        managed_position.exchange_position = exchange_pos
                        
                        # Check if position was closed on exchange
                        if abs(exchange_pos.size) == 0 and managed_position.is_open:
                            managed_position.status = PositionStatus.CLOSED
                            managed_position.closed_at = datetime.now(timezone.utc)
                            managed_position.exit_reason = ExitReason.MANUAL  # Unknown reason
                            self._move_to_history(position_id)
                            sync_results['closed'] += 1
                        else:
                            await self._update_position_metrics(managed_position)
                            sync_results['updated'] += 1
                    else:
                        # Position not found on exchange
                        if managed_position.is_open:
                            managed_position.status = PositionStatus.CLOSED
                            managed_position.closed_at = datetime.now(timezone.utc)
                            managed_position.exit_reason = ExitReason.MANUAL
                            self._move_to_history(position_id)
                            sync_results['closed'] += 1
                            
                except Exception as e:
                    error_msg = f"Error syncing position {position_id}: {e}"
                    self.logger.warning(f"⚠️ {error_msg}")
                    sync_results['errors'].append(error_msg)
            
            # Check for new positions on exchange not being tracked
            for symbol, exchange_pos in exchange_position_map.items():
                if abs(exchange_pos.size) > 0:
                    # Check if we're tracking this position
                    is_tracked = any(
                        mp.config.symbol == symbol and mp.is_active 
                        for mp in self.active_positions.values()
                    )
                    
                    if not is_tracked:
                        self.logger.warning(f"⚠️ Found untracked position on exchange: {symbol}")
                        sync_results['new_found'] += 1
            
            self.logger.info(f"✅ Position sync complete: {sync_results}")
            return sync_results
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing positions: {e}")
            return {'error': str(e)}
    
    def get_position_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive position statistics.
        
        Returns:
            Dict[str, Any]: Position statistics
        """
        active_count = len(self.active_positions)
        closed_count = len(self.position_history)
        
        # Calculate win rate
        total_closed = self.stats['profitable_positions'] + self.stats['losing_positions']
        win_rate = (self.stats['profitable_positions'] / total_closed * 100) if total_closed > 0 else 0
        
        # Calculate profit factor
        total_profit = sum(pos.metrics.realized_pnl for pos in self.position_history 
                          if pos.metrics.realized_pnl > 0)
        total_loss = abs(sum(pos.metrics.realized_pnl for pos in self.position_history 
                            if pos.metrics.realized_pnl < 0))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        # Positions by symbol
        positions_by_symbol = {}
        for managed_position in self.active_positions.values():
            symbol = managed_position.config.symbol
            positions_by_symbol[symbol] = positions_by_symbol.get(symbol, 0) + 1
        
        # Current total exposure
        total_exposure = 0.0
        for managed_position in self.active_positions.values():
            if managed_position.is_open and managed_position.current_price:
                exposure = managed_position.filled_size * managed_position.current_price
                total_exposure += exposure
        
        return {
            'active_positions': active_count,
            'closed_positions': closed_count,
            'total_positions_opened': self.stats['total_positions_opened'],
            'profitable_positions': self.stats['profitable_positions'],
            'losing_positions': self.stats['losing_positions'],
            'win_rate_percent': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'total_realized_pnl': round(self.stats['total_realized_pnl'], 2),
            'total_fees_paid': round(self.stats['total_fees_paid'], 2),
            'best_trade_pnl': round(self.stats['best_trade_pnl'], 2),
            'worst_trade_pnl': round(self.stats['worst_trade_pnl'], 2),
            'total_exposure_usdt': round(total_exposure, 2),
            'positions_by_symbol': positions_by_symbol,
            'last_position_time': self.stats['last_position_time'],
            'configuration': {
                'max_positions': self.max_positions,
                'max_position_value': self.max_position_value,
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_total_risk': self.max_total_risk
            }
        }
    
    # Private Helper Methods
    
    async def _validate_position_config(self, config: PositionConfig) -> None:
        """Validate position configuration."""
        if not config.symbol or not isinstance(config.symbol, str):
            raise ValidationError("Invalid symbol")
        
        if config.size <= 0:
            raise ValidationError("Position size must be positive")
        
        if not isinstance(config.side, PositionSide):
            raise ValidationError("Invalid position side")
        
        # Validate symbol
        if not self.exchange.validate_symbol(config.symbol):
            raise ValidationError(f"Invalid or unsupported symbol: {config.symbol}")
        
        # Validate stop loss and take profit logic
        if config.stop_loss_price and config.entry_price:
            if config.side == PositionSide.LONG and config.stop_loss_price >= config.entry_price:
                raise ValidationError("Long position stop loss must be below entry price")
            elif config.side == PositionSide.SHORT and config.stop_loss_price <= config.entry_price:
                raise ValidationError("Short position stop loss must be above entry price")
        
        if config.take_profit_price and config.entry_price:
            if config.side == PositionSide.LONG and config.take_profit_price <= config.entry_price:
                raise ValidationError("Long position take profit must be above entry price")
            elif config.side == PositionSide.SHORT and config.take_profit_price >= config.entry_price:
                raise ValidationError("Short position take profit must be below entry price")
    
    async def _check_risk_limits(self, config: PositionConfig) -> None:
        """Check risk management limits."""
        # Check maximum number of positions
        if len(self.active_positions) >= self.max_positions:
            raise RiskManagementError(f"Maximum positions limit reached: {self.max_positions}")
        
        # Check position value limit
        if config.entry_price:
            position_value = config.size * config.entry_price
            if position_value > self.max_position_value:
                raise RiskManagementError(
                    f"Position value {position_value} exceeds limit {self.max_position_value}")
        
        # Check total risk exposure
        current_total_risk = await self._calculate_total_risk_exposure()
        position_risk = self._calculate_position_risk(config)
        
        if (current_total_risk + position_risk) > self.max_total_risk:
            raise RiskManagementError(
                f"Total risk exposure would exceed limit: {current_total_risk + position_risk}% > {self.max_total_risk}%")
    
    async def _calculate_position_size(self, config: PositionConfig) -> float:
        """Calculate and validate position size."""
        # Get symbol info for precision
        try:
            symbol_info = await self.exchange.get_symbol_info(config.symbol)
            
            # Apply precision rules
            precise_size = self.exchange.calculate_quantity_precision(config.symbol, config.size)
            
            # Check minimum size requirements
            min_qty = 0.001  # Default minimum
            for filter_info in symbol_info.get('filters', []):
                if filter_info['filterType'] == 'LOT_SIZE':
                    min_qty = float(filter_info['minQty'])
                    break
            
            if precise_size < min_qty:
                raise ValidationError(f"Position size {precise_size} below minimum {min_qty}")
            
            return precise_size
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating position size, using original: {e}")
            return config.size
    
    def _create_entry_order_request(self, config: PositionConfig) -> OrderRequest:
        """Create order request for position entry."""
        side = OrderSide.BUY if config.side == PositionSide.LONG else OrderSide.SELL
        
        return OrderRequest(
            symbol=config.symbol,
            side=side,
            order_type=config.entry_order_type,
            quantity=config.size,
            price=config.entry_price,
            time_in_force="GTC",
            reduce_only=False,
            priority=OrderPriority.NORMAL,
            strategy_name=config.strategy_name,
            tags=config.tags + ["entry"],
            notes=f"Position entry: {config.side.value}"
        )
    
    async def _setup_stop_loss(self, managed_position: ManagedPosition) -> None:
        """Setup stop loss order for position."""
        try:
            config = managed_position.config
            
            # Calculate stop loss price if percentage provided
            stop_price = config.stop_loss_price
            if not stop_price and config.stop_loss_percent and config.entry_price:
                if config.side == PositionSide.LONG:
                    stop_price = config.entry_price * (1 - config.stop_loss_percent / 100)
                else:
                    stop_price = config.entry_price * (1 + config.stop_loss_percent / 100)
            
            if not stop_price:
                return
            
            # Create stop loss order (opposite side)
            side = OrderSide.SELL if config.side == PositionSide.LONG else OrderSide.BUY
            
            stop_order_request = OrderRequest(
                symbol=config.symbol,
                side=side,
                order_type=OrderType.STOP_MARKET,
                quantity=managed_position.filled_size or config.size,
                stop_price=stop_price,
                reduce_only=True,
                priority=OrderPriority.HIGH,
                strategy_name=config.strategy_name,
                tags=config.tags + ["stop_loss"],
                notes="Stop loss protection"
            )
            
            # Place stop loss order
            stop_tracker = await self.order_manager.place_order(stop_order_request)
            managed_position.stop_loss_order = stop_tracker
            
            self.logger.info(f"✅ Stop loss set at {stop_price} for position {managed_position.position_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup stop loss: {e}")
            managed_position.error_messages.append(f"Stop loss setup failed: {e}")
    
    async def _setup_take_profit(self, managed_position: ManagedPosition) -> None:
        """Setup take profit order for position."""
        try:
            config = managed_position.config
            
            # Calculate take profit price if percentage provided
            tp_price = config.take_profit_price
            if not tp_price and config.take_profit_percent and config.entry_price:
                if config.side == PositionSide.LONG:
                    tp_price = config.entry_price * (1 + config.take_profit_percent / 100)
                else:
                    tp_price = config.entry_price * (1 - config.take_profit_percent / 100)
            
            if not tp_price:
                return
            
            # Create take profit order (opposite side)
            side = OrderSide.SELL if config.side == PositionSide.LONG else OrderSide.BUY
            
            tp_order_request = OrderRequest(
                symbol=config.symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=managed_position.filled_size or config.size,
                price=tp_price,
                reduce_only=True,
                priority=OrderPriority.NORMAL,
                strategy_name=config.strategy_name,
                tags=config.tags + ["take_profit"],
                notes="Take profit target"
            )
            
            # Place take profit order
            tp_tracker = await self.order_manager.place_order(tp_order_request)
            managed_position.take_profit_order = tp_tracker
            
            self.logger.info(f"✅ Take profit set at {tp_price} for position {managed_position.position_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup take profit: {e}")
            managed_position.error_messages.append(f"Take profit setup failed: {e}")
    
    async def _cancel_protective_orders(self, managed_position: ManagedPosition) -> None:
        """Cancel stop loss and take profit orders."""
        try:
            cancelled_orders = []
            
            # Cancel stop loss
            if managed_position.stop_loss_order and managed_position.stop_loss_order.is_active:
                await self.order_manager.cancel_order(
                    managed_position.stop_loss_order.order_request.client_order_id,
                    "Position closing"
                )
                cancelled_orders.append("stop_loss")
            
            # Cancel take profit
            if managed_position.take_profit_order and managed_position.take_profit_order.is_active:
                await self.order_manager.cancel_order(
                    managed_position.take_profit_order.order_request.client_order_id,
                    "Position closing"
                )
                cancelled_orders.append("take_profit")
            
            if cancelled_orders:
                self.logger.info(f"✅ Cancelled protective orders: {', '.join(cancelled_orders)}")
                
        except Exception as e:
            self.logger.error(f"❌ Error cancelling protective orders: {e}")
    
    async def _monitor_position(self, managed_position: ManagedPosition) -> None:
        """Monitor position lifecycle and update metrics."""
        try:
            position_id = managed_position.position_id
            self.logger.info(f"👁️ Starting position monitoring: {position_id}")
            
            while managed_position.is_active:
                try:
                    # Update position metrics
                    await self._update_position_metrics(managed_position)
                    
                    # Check for fills on entry orders
                    await self._check_entry_fills(managed_position)
                    
                    # Check for fills on exit orders
                    await self._check_exit_fills(managed_position)
                    
                    # Check position timeouts
                    await self._check_position_timeouts(managed_position)
                    
                    # Check risk limits
                    await self._check_position_risk(managed_position)
                    
                    # Sleep before next check
                    await asyncio.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in position monitoring {position_id}: {e}")
                    managed_position.error_messages.append(f"Monitoring error: {e}")
                    await asyncio.sleep(10)  # Wait longer on error
            
            self.logger.info(f"👁️ Position monitoring ended: {position_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error in position monitoring: {e}")
            managed_position.status = PositionStatus.ERROR
            managed_position.error_messages.append(f"Fatal monitoring error: {e}")
    
    async def _update_position_metrics(self, managed_position: ManagedPosition) -> None:
        """Update position metrics and P&L."""
        try:
            # Get current market price
            ticker = await self.exchange.get_ticker(managed_position.config.symbol)
            managed_position.current_price = ticker.price
            
            # Get exchange position data
            exchange_position = await self.exchange.get_position(managed_position.config.symbol)
            if exchange_position:
                managed_position.exchange_position = exchange_position
            
            # Update metrics
            metrics = managed_position.metrics
            
            if managed_position.average_entry_price and managed_position.current_price:
                # Calculate unrealized P&L
                if managed_position.is_long:
                    price_diff = managed_position.current_price - managed_position.average_entry_price
                else:
                    price_diff = managed_position.average_entry_price - managed_position.current_price
                
                metrics.unrealized_pnl = price_diff * managed_position.filled_size
                
                if managed_position.average_entry_price > 0:
                    metrics.unrealized_pnl_percent = (price_diff / managed_position.average_entry_price) * 100
                
                # Update max profit and loss
                if metrics.unrealized_pnl > metrics.max_profit:
                    metrics.max_profit = metrics.unrealized_pnl
                
                if metrics.unrealized_pnl < metrics.max_loss:
                    metrics.max_loss = metrics.unrealized_pnl
                
                # Calculate drawdown
                if metrics.max_profit > 0:
                    current_drawdown = (metrics.max_profit - metrics.unrealized_pnl) / metrics.max_profit * 100
                    if current_drawdown > metrics.max_drawdown:
                        metrics.max_drawdown = current_drawdown
            
            # Update hold time
            if managed_position.opened_at:
                metrics.hold_time = datetime.now(timezone.utc) - managed_position.opened_at
            
            # Update net P&L
            metrics.net_pnl = metrics.realized_pnl + metrics.unrealized_pnl - metrics.total_fees
            
            managed_position.last_update = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error updating position metrics: {e}")
    
    async def _check_entry_fills(self, managed_position: ManagedPosition) -> None:
        """Check for fills on entry orders."""
        try:
            total_filled = 0.0
            total_value = 0.0
            
            for entry_tracker in managed_position.entry_orders:
                # Update order status
                updated_tracker = await self.order_manager.get_order_status(
                    entry_tracker.order_request.client_order_id
                )
                
                if updated_tracker and updated_tracker.filled_quantity > 0:
                    total_filled += updated_tracker.filled_quantity
                    
                    if updated_tracker.average_price:
                        total_value += updated_tracker.filled_quantity * updated_tracker.average_price
            
            # Update position fill data
            if total_filled > 0:
                managed_position.filled_size = total_filled
                managed_position.average_entry_price = total_value / total_filled if total_filled > 0 else None
                
                # Update position status
                if total_filled >= managed_position.config.size:
                    if managed_position.status == PositionStatus.OPENING:
                        managed_position.status = PositionStatus.OPEN
                        managed_position.opened_at = datetime.now(timezone.utc)
                        self.logger.info(f"✅ Position fully opened: {managed_position.position_id}")
                elif managed_position.status == PositionStatus.OPENING:
                    managed_position.status = PositionStatus.PARTIALLY_FILLED
                    self.logger.info(f"📊 Position partially filled: {managed_position.position_id} "
                                   f"({managed_position.fill_percentage:.1f}%)")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking entry fills: {e}")
    
    async def _check_exit_fills(self, managed_position: ManagedPosition) -> None:
        """Check for fills on exit orders."""
        try:
            total_exit_filled = 0.0
            
            for exit_tracker in managed_position.exit_orders:
                # Update order status
                updated_tracker = await self.order_manager.get_order_status(
                    exit_tracker.order_request.client_order_id
                )
                
                if updated_tracker and updated_tracker.filled_quantity > 0:
                    total_exit_filled += updated_tracker.filled_quantity
            
            # Check stop loss and take profit fills
            if managed_position.stop_loss_order:
                sl_tracker = await self.order_manager.get_order_status(
                    managed_position.stop_loss_order.order_request.client_order_id
                )
                if sl_tracker and sl_tracker.filled_quantity > 0:
                    total_exit_filled += sl_tracker.filled_quantity
                    managed_position.exit_reason = ExitReason.STOP_LOSS
            
            if managed_position.take_profit_order:
                tp_tracker = await self.order_manager.get_order_status(
                    managed_position.take_profit_order.order_request.client_order_id
                )
                if tp_tracker and tp_tracker.filled_quantity > 0:
                    total_exit_filled += tp_tracker.filled_quantity
                    managed_position.exit_reason = ExitReason.TAKE_PROFIT
            
            # Check if position is fully closed
            if total_exit_filled >= managed_position.filled_size:
                await self._finalize_position_close(managed_position)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking exit fills: {e}")
    
    async def _check_position_timeouts(self, managed_position: ManagedPosition) -> None:
        """Check for position timeouts."""
        try:
            now = datetime.now(timezone.utc)
            
            # Check entry timeout
            if (managed_position.status == PositionStatus.OPENING and
                now - managed_position.created_at > managed_position.config.entry_timeout):
                
                self.logger.warning(f"⏰ Position entry timeout: {managed_position.position_id}")
                await self.close_position(managed_position.position_id, ExitReason.TIMEOUT)
                return
            
            # Check maximum hold time
            if (managed_position.config.max_hold_time and 
                managed_position.opened_at and
                now - managed_position.opened_at > managed_position.config.max_hold_time):
                
                self.logger.warning(f"⏰ Position hold time timeout: {managed_position.position_id}")
                await self.close_position(managed_position.position_id, ExitReason.TIMEOUT)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking timeouts: {e}")
    
    async def _check_position_risk(self, managed_position: ManagedPosition) -> None:
        """Check position risk limits."""
        try:
            metrics = managed_position.metrics
            
            # Check maximum loss limit
            if metrics.unrealized_pnl_percent < -managed_position.config.max_loss_percent:
                self.logger.warning(f"⚠️ Position loss limit exceeded: {managed_position.position_id} "
                                  f"({metrics.unrealized_pnl_percent:.2f}%)")
                await self.close_position(managed_position.position_id, ExitReason.RISK_MANAGEMENT)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking position risk: {e}")
    
    async def _finalize_position_close(self, managed_position: ManagedPosition) -> None:
        """Finalize position closure and update statistics."""
        try:
            managed_position.status = PositionStatus.CLOSED
            managed_position.closed_at = datetime.now(timezone.utc)
            
            # Calculate final metrics
            await self._update_position_metrics(managed_position)
            
            # Update realized P&L
            managed_position.metrics.realized_pnl = managed_position.metrics.unrealized_pnl
            managed_position.metrics.unrealized_pnl = 0.0
            
            # Update statistics
            if managed_position.metrics.realized_pnl > 0:
                self.stats['profitable_positions'] += 1
                if managed_position.metrics.realized_pnl > self.stats['best_trade_pnl']:
                    self.stats['best_trade_pnl'] = managed_position.metrics.realized_pnl
            else:
                self.stats['losing_positions'] += 1
                if managed_position.metrics.realized_pnl < self.stats['worst_trade_pnl']:
                    self.stats['worst_trade_pnl'] = managed_position.metrics.realized_pnl
            
            self.stats['total_realized_pnl'] += managed_position.metrics.realized_pnl
            self.stats['total_fees_paid'] += managed_position.metrics.total_fees
            
            # Move to history
            self._move_to_history(managed_position.position_id)
            
            self.logger.info(f"✅ Position closed: {managed_position.position_id} "
                           f"P&L: {managed_position.metrics.realized_pnl:.2f} USDT "
                           f"({managed_position.metrics.unrealized_pnl_percent:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"❌ Error finalizing position close: {e}")
    
    def _move_to_history(self, position_id: str) -> None:
        """Move position from active tracking to history."""
        if position_id in self.active_positions:
            managed_position = self.active_positions.pop(position_id)
            self.position_history.append(managed_position)
            
            # Keep history size manageable
            if len(self.position_history) > 500:
                self.position_history = self.position_history[-500:]
    
    async def _calculate_total_risk_exposure(self) -> float:
        """Calculate total risk exposure across all positions."""
        total_risk = 0.0
        
        # Get account balance
        try:
            account_info = await self.exchange.get_account_info()
            account_balance = account_info.get('available_balance', 0)
            
            if account_balance <= 0:
                return 100.0  # Maximum risk if no balance info
            
            for managed_position in self.active_positions.values():
                if managed_position.is_open and managed_position.current_price:
                    position_value = managed_position.filled_size * managed_position.current_price
                    position_risk = (position_value / account_balance) * 100
                    total_risk += position_risk
            
            return total_risk
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating total risk exposure: {e}")
            return 0.0
    
    def _calculate_position_risk(self, config: PositionConfig) -> float:
        """Calculate risk for a single position."""
        if not config.entry_price:
            return 0.0
        
        position_value = config.size * config.entry_price
        
        # Estimate risk based on stop loss
        if config.stop_loss_price:
            if config.side == PositionSide.LONG:
                risk_per_unit = config.entry_price - config.stop_loss_price
            else:
                risk_per_unit = config.stop_loss_price - config.entry_price
            
            total_risk = abs(risk_per_unit) * config.size
            return (total_risk / position_value) * 100 if position_value > 0 else 0.0
        
        elif config.stop_loss_percent:
            return config.stop_loss_percent
        
        else:
            # Default risk assumption
            return self.default_max_loss_percent
    
    async def emergency_close_all(self) -> int:
        """Emergency closure of all positions."""
        try:
            self.logger.error("🚨 EMERGENCY: Closing all positions!")
            
            # Close all positions
            closed_count = await self.close_all_positions(reason=ExitReason.EMERGENCY)
            
            # Cancel all orders
            await self.order_manager.emergency_cancel_all()
            
            self.logger.error(f"🚨 Emergency closure complete: {closed_count} positions")
            return closed_count
            
        except Exception as e:
            self.logger.error(f"❌ Error in emergency position closure: {e}")
            return 0