"""
Trading Bot Controller - Complete Implementation
==============================================

Full orchestration controller that integrates all bot components including
data fetching, strategy execution, risk management, and order execution.

This is the central hub that coordinates all trading activities and manages
the complete trading workflow from signal generation to order execution.

Author: dat-ns
Version: 1.0.0
"""

import asyncio
import logging
import json
import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np

from config.settings import Settings
from core.utils.exceptions import (
    TradingBotException,
    InitializationError,
    StrategyError,
    RiskManagementError,
    DataError,
    OrderExecutionError,
    ExchangeConnectionError
)

# Import real components
from core.exchange.binance_exchange import BinanceExchange
from core.data.data_fetcher import DataFetcher
from core.data.data_processor import DataProcessor
from core.indicators.technical_indicators import TechnicalIndicators
from strategies.base_strategy import BaseStrategy, Signal, SignalType
from strategies import get_strategy
from risk_management.risk_calculator import RiskCalculator
from risk_management.position_sizing import PositionSizer


class BotState(Enum):
    """Enumeration of possible bot states."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"  
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


class TradingMode(Enum):
    """Trading mode enumeration."""
    LIVE = "live"
    PAPER = "paper" 
    BACKTEST = "backtest"
    DRY_RUN = "dry_run"


@dataclass
class TradeExecution:
    """Trade execution details."""
    signal: Signal
    order_id: Optional[str] = None
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    commission: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: str = "pending"
    error_message: Optional[str] = None


@dataclass
class Position:
    """Open position details."""
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    entry_time: datetime = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price


class TradingBotController:
    """
    Complete trading bot controller with full component integration.
    
    Features:
    - Real exchange connectivity
    - Multiple strategy support
    - Advanced risk management
    - Real-time data processing
    - Performance monitoring
    - Error recovery systems
    """
    
    def __init__(self, settings: Settings):
        """Initialize the bot controller with all components."""
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Bot state management
        self.state = BotState.STOPPED
        self.trading_mode = self._determine_trading_mode()
        self.start_time: Optional[datetime] = None
        self.last_heartbeat = datetime.now()
        
        # Core components
        self.exchange: Optional[BinanceExchange] = None
        self.data_fetcher: Optional[DataFetcher] = None
        self.data_processor: Optional[DataProcessor] = None
        self.indicators: Optional[TechnicalIndicators] = None
        self.strategies: Dict[str, BaseStrategy] = {}
        self.risk_calculator: Optional[RiskCalculator] = None
        self.position_sizer: Optional[PositionSizer] = None
        
        # Trading state
        self.is_running = False
        self.is_paused = False
        self.emergency_stop_triggered = False
        self.last_signal_time: Optional[datetime] = None
        self.last_data_update: Optional[datetime] = None
        
        # Market data storage
        self.market_data: Optional[pd.DataFrame] = None
        self.current_price: float = 0.0
        self.price_history: List[float] = []
        
        # Positions and orders
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.open_positions: Dict[str, Position] = {}
        self.trade_history: List[TradeExecution] = []
        
        # Performance tracking
        self.stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade_duration': 0.0,
            'last_trade_time': None,
            'uptime_seconds': 0,
            'signals_generated': 0,
            'signals_executed': 0,
            'max_equity': 0.0,
            'current_equity': 0.0,
            'daily_pnl': 0.0,
            'monthly_pnl': 0.0
        }
        
        # Risk and safety
        self.error_count = 0
        self.last_error = None
        self.error_threshold = 10
        self.daily_loss_limit = None
        self.position_limits: Dict[str, float] = {}
        
        # Configuration validation
        self._validate_configuration()
        
        self.logger.info(f"🤖 Bot Controller initialized in {self.trading_mode.value} mode")
    
    def _determine_trading_mode(self) -> TradingMode:
        """Determine trading mode from settings."""
        import os
        
        if os.getenv('DRY_RUN') == 'true':
            return TradingMode.DRY_RUN
        elif self.settings.binance.testnet:
            return TradingMode.PAPER
        else:
            return TradingMode.LIVE
    
    def _validate_configuration(self):
        """Comprehensive configuration validation."""
        try:
            # Validate required settings
            required_settings = ['trading', 'binance']
            for setting in required_settings:
                if not hasattr(self.settings, setting):
                    raise InitializationError(f"Missing required setting section: {setting}")
            
            # Validate trading parameters
            trading = self.settings.trading
            if trading.risk_per_trade <= 0 or trading.risk_per_trade > 0.1:
                raise InitializationError("Risk per trade must be between 0 and 0.1")
            
            if trading.max_positions <= 0 or trading.max_positions > 10:
                raise InitializationError("Max positions must be between 1 and 10")
            
            # Validate symbol format
            if not trading.default_symbol or len(trading.default_symbol) < 6:
                raise InitializationError("Invalid trading symbol")
            
            # Set risk limits
            self.daily_loss_limit = trading.risk_per_trade * 5  # 5x single trade risk
            
            self.logger.info("✅ Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            raise InitializationError(f"Invalid configuration: {e}")
    
    async def initialize(self):
        """Initialize all bot components in correct dependency order."""
        try:
            self.logger.info("🔧 Initializing Trading Bot Controller...")
            self.state = BotState.INITIALIZING
            
            # Initialize components in dependency order
            await self._initialize_exchange()
            await self._initialize_data_components()
            await self._initialize_trading_components()
            await self._initialize_strategies()
            
            # Perform system health checks
            await self._perform_health_checks()
            
            # Load initial market data
            await self._load_initial_data()
            
            self.state = BotState.STOPPED
            self.logger.info("✅ Bot Controller initialization complete")
            
        except Exception as e:
            self.state = BotState.ERROR
            self.last_error = str(e)
            self.logger.error(f"❌ Initialization failed: {e}")
            raise InitializationError(f"Failed to initialize: {e}") from e
    
    async def _initialize_exchange(self):
        """Initialize exchange connection."""
        self.logger.info("🔌 Initializing exchange connection...")
        
        self.exchange = BinanceExchange(self.settings)
        await self.exchange.initialize()
        
        # Test connection
        try:
            account_info = await self.exchange.get_account_info()
            self.logger.info(f"✅ Exchange connected - Account type: {account_info.get('accountType', 'Unknown')}")
            
            # Log available balances (without amounts for security)
            balances = [b for b in account_info.get('balances', []) if float(b['free']) > 0]
            self.logger.info(f"📊 Available assets: {len(balances)} types")
            
        except Exception as e:
            raise ExchangeConnectionError(f"Failed to connect to exchange: {e}")
    
    async def _initialize_data_components(self):
        """Initialize data fetching and processing components."""
        self.logger.info("📊 Initializing data components...")
        
        # Initialize data fetcher
        self.data_fetcher = DataFetcher(self.exchange, self.settings)
        await self.data_fetcher.initialize()
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.settings)
        
        # Initialize technical indicators
        self.indicators = TechnicalIndicators()
        
        self.logger.info("✅ Data components initialized")
    
    async def _initialize_trading_components(self):
        """Initialize trading and risk management components."""
        self.logger.info("🛡️ Initializing trading components...")
        
        # Initialize risk calculator
        self.risk_calculator = RiskCalculator(self.settings)
        
        # Initialize position sizer
        self.position_sizer = PositionSizer(self.settings)
        
        self.logger.info("✅ Trading components initialized")
    
    async def _initialize_strategies(self):
        """Initialize trading strategies."""
        self.logger.info("🧠 Initializing strategies...")
        
        # For now, initialize a default strategy
        # In production, this would load from configuration
        strategy_name = getattr(self.settings, 'strategy_name', 'ema_crossover')
        
        try:
            strategy = get_strategy(
                strategy_name,
                symbol=self.settings.trading.default_symbol,
                timeframe=self.settings.trading.default_timeframe
            )
            
            await strategy.initialize()
            self.strategies[strategy_name] = strategy
            
            self.logger.info(f"✅ Strategy initialized: {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize strategy {strategy_name}: {e}")
            # Continue with mock strategy for testing
            from strategies.base_strategy import BaseStrategy, Signal, SignalType, SignalStrength
            
            class MockStrategy(BaseStrategy):
                async def initialize(self):
                    self.is_initialized = True
                    return True
                
                async def generate_signal(self, data):
                    return None  # No signals for mock
                
                def calculate_indicators(self, data):
                    return {}
                
                def validate_signal(self, signal, data):
                    return True
            
            mock_strategy = MockStrategy()
            await mock_strategy.initialize()
            self.strategies['mock'] = mock_strategy
            self.logger.info("✅ Mock strategy initialized as fallback")
    
    async def _perform_health_checks(self):
        """Perform comprehensive system health checks."""
        self.logger.info("🔍 Performing health checks...")
        
        # Check exchange connectivity
        try:
            await self.exchange.get_server_time()
        except Exception as e:
            raise InitializationError(f"Exchange health check failed: {e}")
        
        # Check market data availability
        try:
            test_data = await self.data_fetcher.fetch_ohlcv(
                self.settings.trading.default_symbol,
                self.settings.trading.default_timeframe,
                limit=10
            )
            if test_data.empty:
                raise DataError("No market data available")
        except Exception as e:
            raise InitializationError(f"Market data health check failed: {e}")
        
        # Check strategy readiness
        for name, strategy in self.strategies.items():
            if not strategy.is_initialized:
                raise InitializationError(f"Strategy {name} not properly initialized")
        
        self.logger.info("✅ All health checks passed")
    
    async def _load_initial_data(self):
        """Load initial market data for strategies."""
        self.logger.info("📈 Loading initial market data...")
        
        try:
            # Fetch initial historical data
            data = await self.data_fetcher.fetch_ohlcv(
                self.settings.trading.default_symbol,
                self.settings.trading.default_timeframe,
                limit=200  # Enough for most indicators
            )
            
            # Process the data
            self.market_data = await self.data_processor.process_ohlcv(data)
            
            if not self.market_data.empty:
                self.current_price = float(self.market_data.iloc[-1]['close'])
                self.price_history = self.market_data['close'].tolist()[-50:]  # Keep last 50
                self.last_data_update = datetime.now()
                
                self.logger.info(f"📊 Loaded {len(self.market_data)} data points")
                self.logger.info(f"💰 Current price: ${self.current_price:,.2f}")
            else:
                raise DataError("No initial data loaded")
                
        except Exception as e:
            raise InitializationError(f"Failed to load initial data: {e}")
    
    async def start(self):
        """Start the trading bot and begin operations."""
        if self.state not in [BotState.STOPPED, BotState.PAUSED]:
            raise TradingBotException(f"Cannot start bot in state: {self.state.value}")
        
        try:
            self.logger.info("🚀 Starting Trading Bot Controller...")
            self.state = BotState.RUNNING
            self.is_running = True
            self.is_paused = False
            self.start_time = datetime.now()
            self.error_count = 0  # Reset error count on restart
            
            # Initialize performance tracking
            await self._initialize_performance_tracking()
            
            # Send startup notification
            await self._log_startup_info()
            
            # Start main trading loop
            await self._run_trading_loop()
            
        except KeyboardInterrupt:
            self.logger.info("👋 Received keyboard interrupt, stopping bot...")
            await self.stop()
        except Exception as e:
            self.state = BotState.ERROR
            self.last_error = str(e)
            self.logger.error(f"❌ Error starting bot: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            await self.emergency_stop()
            raise
    
    async def _initialize_performance_tracking(self):
        """Initialize performance tracking systems."""
        try:
            # Get current account balance
            account_info = await self.exchange.get_account_info()
            
            # Calculate initial equity (USDT balance + BTC value)
            usdt_balance = 0.0
            btc_balance = 0.0
            
            for balance in account_info.get('balances', []):
                asset = balance['asset']
                free_amount = float(balance['free'])
                
                if asset == 'USDT':
                    usdt_balance = free_amount
                elif asset == 'BTC':
                    btc_balance = free_amount
            
            # Calculate total equity in USDT
            btc_value_usdt = btc_balance * self.current_price if btc_balance > 0 else 0.0
            initial_equity = usdt_balance + btc_value_usdt
            
            self.stats.update({
                'current_equity': initial_equity,
                'max_equity': initial_equity,
                'daily_pnl': 0.0,
                'monthly_pnl': 0.0
            })
            
            self.logger.info(f"💰 Initial equity: ${initial_equity:,.2f}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize performance tracking: {e}")
            # Continue with default values
            self.stats.update({
                'current_equity': 10000.0,  # Default value
                'max_equity': 10000.0
            })
    
    async def _log_startup_info(self):
        """Log comprehensive startup information."""
        info = {
            'Mode': self.trading_mode.value.upper(),
            'Symbol': self.settings.trading.default_symbol,
            'Timeframe': self.settings.trading.default_timeframe,
            'Strategies': list(self.strategies.keys()),
            'Risk per Trade': f"{self.settings.trading.risk_per_trade*100:.1f}%",
            'Max Positions': self.settings.trading.max_positions,
            'Daily Loss Limit': f"{self.daily_loss_limit*100:.1f}%"
        }
        
        self.logger.info("🎯 Trading Bot Started Successfully!")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
    
    async def stop(self):
        """Stop the trading bot gracefully."""
        if not self.is_running:
            self.logger.info("🔄 Bot is already stopped")
            return
        
        try:
            self.logger.info("🛑 Stopping Trading Bot Controller...")
            self.state = BotState.STOPPING
            self.is_running = False
            
            # Cancel all pending orders
            await self._cancel_all_pending_orders()
            
            # Close positions if configured
            if getattr(self.settings, 'close_positions_on_stop', False):
                await self._close_all_positions()
            
            # Save final statistics
            await self._save_final_statistics()
            
            # Log shutdown summary
            await self._log_shutdown_summary()
            
            self.state = BotState.STOPPED
            self.logger.info("✅ Trading Bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during graceful stop: {e}")
            await self.emergency_stop()
    
    async def emergency_stop(self):
        """Emergency stop - immediately halt all operations."""
        self.logger.error("🚨 EMERGENCY STOP ACTIVATED!")
        self.state = BotState.EMERGENCY_STOP
        self.emergency_stop_triggered = True
        self.is_running = False
        
        try:
            # Cancel all orders immediately
            await self._emergency_cancel_orders()
            
            # Close all positions at market price
            if self.trading_mode == TradingMode.LIVE:
                await self._emergency_close_positions()
            
            # Log emergency details
            await self._log_emergency_stop()
            
        except Exception as e:
            self.logger.error(f"❌ Critical error during emergency stop: {e}")
    
    async def _run_trading_loop(self):
        """Main trading loop with comprehensive error handling."""
        self.logger.info("🔄 Starting main trading loop...")
        
        loop_count = 0
        last_status_log = datetime.now()
        last_data_fetch = datetime.now() - timedelta(minutes=5)  # Force initial fetch
        
        try:
            while self.is_running and not self.emergency_stop_triggered:
                loop_start_time = datetime.now()
                self.last_heartbeat = loop_start_time
                
                # Skip processing if paused
                if self.is_paused:
                    await asyncio.sleep(1)
                    continue
                
                try:
                    # 1. Fetch latest market data (rate limited)
                    if (loop_start_time - last_data_fetch).total_seconds() >= 5:  # Every 5 seconds
                        await self._update_market_data()
                        last_data_fetch = loop_start_time
                    
                    # 2. Update positions and calculate PnL
                    await self._update_positions()
                    
                    # 3. Check risk limits and safety conditions
                    if not await self._check_safety_conditions():
                        self.logger.warning("⚠️ Safety conditions not met, skipping cycle")
                        await asyncio.sleep(5)
                        continue
                    
                    # 4. Process strategies and generate signals
                    signals = await self._process_strategies()
                    
                    # 5. Process signals and execute trades
                    for signal in signals:
                        await self._process_signal(signal)
                    
                    # 6. Update performance metrics
                    await self._update_performance_metrics()
                    
                    # 7. Check exit conditions for open positions
                    await self._check_exit_conditions()
                    
                    # 8. Periodic housekeeping
                    if loop_count % 20 == 0:  # Every 20 cycles
                        await self._perform_housekeeping()
                    
                    # 9. Log status periodically
                    if (loop_start_time - last_status_log).total_seconds() > 300:  # Every 5 minutes
                        await self._log_periodic_status()
                        last_status_log = loop_start_time
                
                except Exception as e:
                    await self._handle_loop_error(f"Trading loop error: {e}")
                
                # Calculate sleep time to maintain consistent loop timing
                loop_duration = (datetime.now() - loop_start_time).total_seconds()
                target_interval = 1.0  # 1 second base interval
                sleep_time = max(0.1, target_interval - loop_duration)
                
                await asyncio.sleep(sleep_time)
                loop_count += 1
                
        except Exception as e:
            self.logger.error(f"❌ Critical error in trading loop: {e}")
            await self.emergency_stop()
            raise
    
    async def _update_market_data(self):
        """Update market data with rate limiting."""
        try:
            # Fetch latest data
            new_data = await self.data_fetcher.fetch_ohlcv(
                self.settings.trading.default_symbol,
                self.settings.trading.default_timeframe,
                limit=50  # Just get recent data
            )
            
            if not new_data.empty:
                # Process new data
                processed_data = await self.data_processor.process_ohlcv(new_data)
                
                # Update market data (keep last 200 rows for indicators)
                if self.market_data is not None:
                    # Append new data and keep recent history
                    combined_data = pd.concat([self.market_data, processed_data]).drop_duplicates()
                    self.market_data = combined_data.tail(200).reset_index(drop=True)
                else:
                    self.market_data = processed_data
                
                # Update current price and history
                latest_candle = self.market_data.iloc[-1]
                self.current_price = float(latest_candle['close'])
                
                # Update price history (keep last 50)
                self.price_history.append(self.current_price)
                if len(self.price_history) > 50:
                    self.price_history = self.price_history[-50:]
                
                self.last_data_update = datetime.now()
                
        except Exception as e:
            raise DataError(f"Failed to update market data: {e}")
    
    async def _update_positions(self):
        """Update open positions and calculate current PnL."""
        try:
            if not self.open_positions:
                return
            
            total_unrealized_pnl = 0.0
            
            for symbol, position in self.open_positions.items():
                # Update current price
                position.current_price = self.current_price
                
                # Calculate unrealized PnL
                if position.side == "long":
                    pnl_per_unit = position.current_price - position.entry_price
                else:  # short
                    pnl_per_unit = position.entry_price - position.current_price
                
                position.unrealized_pnl = pnl_per_unit * position.quantity
                total_unrealized_pnl += position.unrealized_pnl
            
            # Update stats
            self.stats['unrealized_pnl'] = total_unrealized_pnl
            
            # Calculate current equity
            account_info = await self.exchange.get_account_info()
            usdt_balance = 0.0
            
            for balance in account_info.get('balances', []):
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            
            current_equity = usdt_balance + total_unrealized_pnl
            self.stats['current_equity'] = current_equity
            
            # Update max equity
            if current_equity > self.stats['max_equity']:
                self.stats['max_equity'] = current_equity
            
            # Calculate drawdown
            if self.stats['max_equity'] > 0:
                drawdown = (self.stats['max_equity'] - current_equity) / self.stats['max_equity']
                self.stats['max_drawdown'] = max(self.stats['max_drawdown'], drawdown)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update positions: {e}")
    
    async def _check_safety_conditions(self) -> bool:
        """Check various safety conditions before trading."""
        try:
            # Check daily loss limit
            if self.stats['daily_pnl'] < -self.daily_loss_limit * self.stats['current_equity']:
                self.logger.warning("⚠️ Daily loss limit reached")
                return False
            
            # Check maximum drawdown limit (10%)
            if self.stats['max_drawdown'] > 0.10:
                self.logger.warning("⚠️ Maximum drawdown exceeded")
                return False
            
            # Check position limits
            if len(self.open_positions) >= self.settings.trading.max_positions:
                return False
            
            # Check data freshness
            if self.last_data_update:
                data_age = (datetime.now() - self.last_data_update).total_seconds()
                if data_age > 60:  # 1 minute old data
                    self.logger.warning("⚠️ Market data is stale")
                    return False
            
            # Check error rate
            if self.error_count > self.error_threshold / 2:  # Half of emergency threshold
                self.logger.warning("⚠️ High error rate detected")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking safety conditions: {e}")
            return False
    
    async def _process_strategies(self) -> List[Signal]:
        """Process all strategies and collect signals."""
        signals = []
        
        if self.market_data is None or len(self.market_data) < 20:
            return signals
        
        try:
            for name, strategy in self.strategies.items():
                try:
                    # Update strategy with latest data
                    await strategy.update_data(self.market_data)
                    
                    # Get current signal
                    signal = strategy.get_current_signal()
                    
                    if signal and self._is_new_signal(signal):
                        # Validate signal
                        if strategy.validate_signal(signal, self.market_data):
                            signals.append(signal)
                            self.stats['signals_generated'] += 1
                            
                            self.logger.info(
                                f"📊 {name}: {signal.type.value} signal "
                                f"(strength: {signal.strength.value}, confidence: {signal.confidence:.2f})"
                            )
                        else:
                            self.logger.debug(f"Signal validation failed for {name}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing strategy {name}: {e}")
                    continue
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error in strategy processing: {e}")
            return []
    
    def _is_new_signal(self, signal: Signal) -> bool:
        """Check if signal is new (not already processed)."""
        if not signal.timestamp:
            return True
        
        # Check if we've processed this signal recently
        if self.last_signal_time:
            time_diff = abs((signal.timestamp - self.last_signal_time).total_seconds())
            return time_diff > 60  # Consider signals more than 1 minute apart as new
        
        return True
    
    async def _process_signal(self, signal: Signal) -> bool:
        """Process a trading signal with full risk management."""
        try:
            # Skip HOLD signals
            if signal.type == SignalType.HOLD:
                return True
            
            self.logger.info(f"🎯 Processing {signal.type.value} signal...")
            
            # 1. Risk management validation
            if not await self._validate_signal_risk(signal):
                self.logger.warning("⚠️ Signal rejected by risk management")
                return False
            
            # 2. Calculate position size
            position_size = await self._calculate_position_size(signal)
            if not position_size or position_size <= 0:
                self.logger.warning("⚠️ Invalid position size calculated")
                return False
            
            # 3. Check available balance
            if not await self._check_available_balance(position_size, signal.price):
                self.logger.warning("⚠️ Insufficient balance for trade")
                return False
            
            # 4. Execute trade
            success = await self._execute_signal(signal, position_size)
            
            if success:
                self.stats['signals_executed'] += 1
                self.last_signal_time = datetime.now()
                
                self.logger.info(f"✅ Signal executed successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error processing signal: {e}")
            await self._handle_error(f"Signal processing error: {e}")
            return False
    
    async def _validate_signal_risk(self, signal: Signal) -> bool:
        """Validate signal against risk management rules."""
        try:
            # Check if we already have a position in this symbol
            if signal.type in [SignalType.BUY, SignalType.SELL]:
                if self.settings.trading.default_symbol in self.open_positions:
                    current_position = self.open_positions[self.settings.trading.default_symbol]
                    
                    # Don't open opposite position
                    if ((signal.type == SignalType.BUY and current_position.side == "short") or
                        (signal.type == SignalType.SELL and current_position.side == "long")):
                        # This could be a close signal, allow it
                        pass
                    elif ((signal.type == SignalType.BUY and current_position.side == "long") or
                          (signal.type == SignalType.SELL and current_position.side == "short")):
                        # Don't add to existing position for now
                        return False
            
            # Check signal strength threshold
            if signal.confidence < 0.6:  # Minimum 60% confidence
                return False
            
            # Check stop loss requirements
            if not signal.stop_loss and signal.type in [SignalType.BUY, SignalType.SELL]:
                # Calculate default stop loss (2% from entry)
                stop_distance = 0.02
                if signal.type == SignalType.BUY:
                    signal.stop_loss = signal.price * (1 - stop_distance)
                else:
                    signal.stop_loss = signal.price * (1 + stop_distance)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Risk validation error: {e}")
            return False
    
    async def _calculate_position_size(self, signal: Signal) -> Optional[float]:
        """Calculate appropriate position size based on risk management."""
        try:
            # Get current account balance
            account_info = await self.exchange.get_account_info()
            
            usdt_balance = 0.0
            for balance in account_info.get('balances', []):
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            
            if usdt_balance <= 0:
                return None
            
            # Calculate risk amount
            risk_amount = usdt_balance * self.settings.trading.risk_per_trade
            
            # Calculate stop loss distance
            if signal.stop_loss:
                stop_distance = abs(signal.price - signal.stop_loss) / signal.price
            else:
                stop_distance = 0.02  # Default 2% stop loss
            
            # Calculate position size based on risk
            if stop_distance > 0:
                position_value = risk_amount / stop_distance
                position_size = position_value / signal.price
            else:
                # Fallback to percentage of balance
                position_value = usdt_balance * self.settings.trading.risk_per_trade
                position_size = position_value / signal.price
            
            # Apply position size limits
            max_position_value = usdt_balance * 0.9  # Max 90% of balance
            if position_size * signal.price > max_position_value:
                position_size = max_position_value / signal.price
            
            # Round down to exchange precision
            position_size = float(Decimal(str(position_size)).quantize(
                Decimal('0.00001'), rounding=ROUND_DOWN
            ))
            
            return position_size if position_size > 0 else None
            
        except Exception as e:
            self.logger.error(f"❌ Position sizing error: {e}")
            return None
    
    async def _check_available_balance(self, position_size: float, price: float) -> bool:
        """Check if we have sufficient balance for the trade."""
        try:
            required_balance = position_size * price
            
            account_info = await self.exchange.get_account_info()
            
            for balance in account_info.get('balances', []):
                if balance['asset'] == 'USDT':
                    available_balance = float(balance['free'])
                    return available_balance >= required_balance
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Balance check error: {e}")
            return False
    
    async def _execute_signal(self, signal: Signal, position_size: float) -> bool:
        """Execute trading signal with proper error handling."""
        try:
            if self.trading_mode in [TradingMode.DRY_RUN, TradingMode.PAPER]:
                # Simulate execution
                return await self._simulate_execution(signal, position_size)
            
            # Real execution for live trading
            symbol = self.settings.trading.default_symbol
            
            # Determine order side
            if signal.type == SignalType.BUY:
                side = 'BUY'
            elif signal.type == SignalType.SELL:
                side = 'SELL'
            elif signal.type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                # Handle position closing
                return await self._close_position(symbol)
            else:
                self.logger.warning(f"Unknown signal type: {signal.type}")
                return False
            
            # Execute market order
            order_result = await self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                quantity=position_size
            )
            
            if order_result:
                # Record successful execution
                execution = TradeExecution(
                    signal=signal,
                    order_id=order_result.get('orderId'),
                    executed_price=float(order_result.get('price', signal.price)),
                    executed_quantity=float(order_result.get('executedQty', position_size)),
                    commission=float(order_result.get('commission', 0)),
                    timestamp=datetime.now(),
                    status='filled'
                )
                
                self.trade_history.append(execution)
                
                # Create or update position
                await self._update_position_from_execution(execution)
                
                # Update statistics
                self.stats['total_trades'] += 1
                self.stats['successful_trades'] += 1
                self.stats['last_trade_time'] = datetime.now()
                
                self.logger.info(f"✅ Order executed: {execution.executed_quantity} @ ${execution.executed_price}")
                
                return True
            else:
                self.stats['failed_trades'] += 1
                return False
                
        except Exception as e:
            self.stats['failed_trades'] += 1
            self.logger.error(f"❌ Execution error: {e}")
            return False
    
    async def _simulate_execution(self, signal: Signal, position_size: float) -> bool:
        """Simulate trade execution for paper trading."""
        try:
            execution = TradeExecution(
                signal=signal,
                order_id=f"SIM_{len(self.trade_history)}",
                executed_price=signal.price,
                executed_quantity=position_size,
                commission=position_size * signal.price * 0.001,  # 0.1% commission
                timestamp=datetime.now(),
                status='simulated'
            )
            
            self.trade_history.append(execution)
            await self._update_position_from_execution(execution)
            
            # Update statistics
            self.stats['total_trades'] += 1
            self.stats['successful_trades'] += 1
            self.stats['last_trade_time'] = datetime.now()
            
            self.logger.info(f"📝 Simulated execution: {position_size} @ ${signal.price}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Simulation error: {e}")
            return False
    
    async def _update_position_from_execution(self, execution: TradeExecution):
        """Update position tracking from trade execution."""
        try:
            symbol = self.settings.trading.default_symbol
            signal = execution.signal
            
            if symbol in self.open_positions:
                # Update existing position
                position = self.open_positions[symbol]
                
                if signal.type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                    # Close position
                    pnl = self._calculate_position_pnl(position, execution.executed_price)
                    position.realized_pnl += pnl
                    
                    self.stats['total_pnl'] += pnl
                    del self.open_positions[symbol]
                    
                    self.logger.info(f"📊 Position closed - PnL: ${pnl:.2f}")
                else:
                    # Add to position (for now, just average the price)
                    total_quantity = position.quantity + execution.executed_quantity
                    weighted_price = (
                        (position.entry_price * position.quantity) + 
                        (execution.executed_price * execution.executed_quantity)
                    ) / total_quantity
                    
                    position.quantity = total_quantity
                    position.entry_price = weighted_price
            else:
                # Create new position
                side = "long" if signal.type == SignalType.BUY else "short"
                
                position = Position(
                    symbol=symbol,
                    side=side,
                    quantity=execution.executed_quantity,
                    entry_price=execution.executed_price,
                    current_price=execution.executed_price,
                    unrealized_pnl=0.0,
                    entry_time=execution.timestamp,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
                
                self.open_positions[symbol] = position
                
                self.logger.info(f"📊 New {side} position: {position.quantity} @ ${position.entry_price}")
                
        except Exception as e:
            self.logger.error(f"❌ Position update error: {e}")
    
    def _calculate_position_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate PnL for position closure."""
        if position.side == "long":
            return (exit_price - position.entry_price) * position.quantity
        else:  # short
            return (position.entry_price - exit_price) * position.quantity
    
    async def _close_position(self, symbol: str) -> bool:
        """Close an existing position."""
        if symbol not in self.open_positions:
            return False
        
        try:
            position = self.open_positions[symbol]
            
            # Determine order side for closing
            close_side = 'SELL' if position.side == 'long' else 'BUY'
            
            if self.trading_mode in [TradingMode.DRY_RUN, TradingMode.PAPER]:
                # Simulate close
                pnl = self._calculate_position_pnl(position, self.current_price)
                position.realized_pnl += pnl
                self.stats['total_pnl'] += pnl
                
                del self.open_positions[symbol]
                self.logger.info(f"📊 Position closed (simulated) - PnL: ${pnl:.2f}")
                return True
            
            # Real close for live trading
            order_result = await self.exchange.create_market_order(
                symbol=symbol,
                side=close_side,
                quantity=position.quantity
            )
            
            if order_result:
                exit_price = float(order_result.get('price', self.current_price))
                pnl = self._calculate_position_pnl(position, exit_price)
                
                position.realized_pnl += pnl
                self.stats['total_pnl'] += pnl
                
                del self.open_positions[symbol]
                
                self.logger.info(f"📊 Position closed - PnL: ${pnl:.2f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Position close error: {e}")
            return False
    
    async def _update_performance_metrics(self):
        """Update comprehensive performance metrics."""
        try:
            # Calculate win rate
            if self.stats['total_trades'] > 0:
                self.stats['win_rate'] = self.stats['successful_trades'] / self.stats['total_trades'] * 100
            
            # Calculate profit factor
            if self.trade_history:
                profits = sum(t.signal.take_profit or 0 for t in self.trade_history if t.status in ['filled', 'simulated'])
                losses = abs(sum(t.signal.stop_loss or 0 for t in self.trade_history if t.status in ['filled', 'simulated']))
                
                self.stats['profit_factor'] = profits / losses if losses > 0 else float('inf')
            
            # Calculate average trade duration
            completed_trades = [t for t in self.trade_history if t.timestamp]
            if len(completed_trades) > 1:
                total_duration = 0
                for i in range(1, len(completed_trades)):
                    duration = (completed_trades[i].timestamp - completed_trades[i-1].timestamp).total_seconds()
                    total_duration += duration
                
                self.stats['avg_trade_duration'] = total_duration / (len(completed_trades) - 1) / 3600  # hours
            
            # Update uptime
            if self.start_time:
                self.stats['uptime_seconds'] = int((datetime.now() - self.start_time).total_seconds())
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance metrics update failed: {e}")
    
    async def _check_exit_conditions(self):
        """Check and execute exit conditions for open positions."""
        if not self.open_positions:
            return
        
        try:
            positions_to_close = []
            
            for symbol, position in self.open_positions.items():
                should_close = False
                close_reason = ""
                
                # Check stop loss
                if position.stop_loss:
                    if position.side == "long" and self.current_price <= position.stop_loss:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif position.side == "short" and self.current_price >= position.stop_loss:
                        should_close = True
                        close_reason = "Stop Loss"
                
                # Check take profit
                if position.take_profit and not should_close:
                    if position.side == "long" and self.current_price >= position.take_profit:
                        should_close = True
                        close_reason = "Take Profit"
                    elif position.side == "short" and self.current_price <= position.take_profit:
                        should_close = True
                        close_reason = "Take Profit"
                
                # Check time-based exit (e.g., end of day)
                if position.entry_time and not should_close:
                    hours_open = (datetime.now() - position.entry_time).total_seconds() / 3600
                    if hours_open > 24:  # Close positions older than 24 hours
                        should_close = True
                        close_reason = "Time Exit"
                
                if should_close:
                    positions_to_close.append((symbol, close_reason))
            
            # Execute closes
            for symbol, reason in positions_to_close:
                self.logger.info(f"🎯 Closing position {symbol} - Reason: {reason}")
                await self._close_position(symbol)
                
        except Exception as e:
            self.logger.error(f"❌ Exit conditions check error: {e}")
    
    async def _perform_housekeeping(self):
        """Perform periodic housekeeping tasks."""
        try:
            # Clean old trade history (keep last 1000)
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
            
            # Update daily PnL reset
            now = datetime.now()
            if hasattr(self, '_last_daily_reset'):
                if now.date() != self._last_daily_reset:
                    self.stats['daily_pnl'] = 0.0
                    self._last_daily_reset = now.date()
            else:
                self._last_daily_reset = now.date()
            
            # Validate data integrity
            await self._validate_data_integrity()
            
            # Reset error count if no recent errors
            if self.error_count > 0 and hasattr(self, '_last_error_time'):
                time_since_error = (now - self._last_error_time).total_seconds()
                if time_since_error > 3600:  # 1 hour without errors
                    self.error_count = max(0, self.error_count - 1)
            
            self.logger.debug("🧹 Housekeeping completed")
            
        except Exception as e:
            self.logger.error(f"❌ Housekeeping error: {e}")
    
    async def _validate_data_integrity(self):
        """Validate data integrity and consistency."""
        try:
            # Check if positions match account balances
            if self.trading_mode == TradingMode.LIVE:
                account_info = await self.exchange.get_account_info()
                
                # This is a simplified check - in production you'd want more comprehensive validation
                for balance in account_info.get('balances', []):
                    asset = balance['asset']
                    free_amount = float(balance['free'])
                    locked_amount = float(balance['locked'])
                    
                    # Log any significant locked amounts (might indicate pending orders)
                    if locked_amount > 0.01:  # $0.01 threshold
                        self.logger.debug(f"💰 {asset}: {free_amount} free, {locked_amount} locked")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data integrity check failed: {e}")
    
    async def _log_periodic_status(self):
        """Log periodic comprehensive status."""
        try:
            status = await self.get_status()
            
            # Core metrics
            uptime_hours = status['bot_info']['uptime_seconds'] / 3600
            
            self.logger.info("📊 === PERIODIC STATUS REPORT ===")
            self.logger.info(f"🟢 State: {status['bot_info']['state']} | Uptime: {uptime_hours:.1f}h")
            self.logger.info(f"💰 Equity: ${status['performance']['current_equity']:.2f} | PnL: ${status['performance']['total_pnl']:.2f}")
            self.logger.info(f"📈 Trades: {status['performance']['total_trades']} | Win Rate: {status['performance']['win_rate']:.1f}%")
            self.logger.info(f"🎯 Signals: {status['performance']['signals_generated']} generated, {status['performance']['signals_executed']} executed")
            self.logger.info(f"📍 Positions: {len(self.open_positions)} open | Orders: {len(self.active_orders)} active")
            self.logger.info(f"💹 Current Price: ${self.current_price:,.2f}")
            
            # Position details
            if self.open_positions:
                for symbol, position in self.open_positions.items():
                    pnl_pct = (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100
                    self.logger.info(f"   📊 {symbol}: {position.side} {position.quantity} @ ${position.entry_price:.2f} | PnL: {pnl_pct:+.2f}%")
            
            self.logger.info("📊 === END STATUS REPORT ===")
            
        except Exception as e:
            self.logger.error(f"❌ Status logging error: {e}")
    
    async def _handle_loop_error(self, error_message: str):
        """Handle errors in the main trading loop."""
        self.error_count += 1
        self.last_error = error_message
        self._last_error_time = datetime.now()
        
        self.logger.error(f"❌ Loop Error #{self.error_count}: {error_message}")
        
        # Check if emergency stop is needed
        if self.error_count >= self.error_threshold:
            self.logger.error(f"🚨 Error threshold ({self.error_threshold}) exceeded!")
            await self.emergency_stop()
        else:
            # Implement progressive backoff
            backoff_time = min(10, self.error_count)
            self.logger.warning(f"⏸️ Backing off for {backoff_time} seconds due to error")
            await asyncio.sleep(backoff_time)
    
    async def _handle_error(self, error_message: str):
        """General error handler."""
        await self._handle_loop_error(error_message)
    
    # Order and position management methods
    
    async def _cancel_all_pending_orders(self):
        """Cancel all pending orders."""
        try:
            if not self.active_orders:
                return
            
            cancelled_count = 0
            for order_id, order_info in list(self.active_orders.items()):
                try:
                    if self.trading_mode == TradingMode.LIVE:
                        await self.exchange.cancel_order(
                            symbol=order_info['symbol'],
                            order_id=order_id
                        )
                    
                    del self.active_orders[order_id]
                    cancelled_count += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to cancel order {order_id}: {e}")
            
            if cancelled_count > 0:
                self.logger.info(f"📝 Cancelled {cancelled_count} pending orders")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to cancel orders: {e}")
    
    async def _close_all_positions(self):
        """Close all open positions."""
        try:
            if not self.open_positions:
                return
            
            closed_count = 0
            for symbol in list(self.open_positions.keys()):
                if await self._close_position(symbol):
                    closed_count += 1
            
            if closed_count > 0:
                self.logger.info(f"📝 Closed {closed_count} positions")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to close positions: {e}")
    
    async def _emergency_cancel_orders(self):
        """Emergency cancel all orders."""
        try:
            await self._cancel_all_pending_orders()
        except Exception as e:
            self.logger.error(f"❌ Emergency order cancellation failed: {e}")
    
    async def _emergency_close_positions(self):
        """Emergency close all positions."""
        try:
            await self._close_all_positions()
        except Exception as e:
            self.logger.error(f"❌ Emergency position closure failed: {e}")
    
    # Statistics and reporting methods
    
    async def _save_final_statistics(self):
        """Save final statistics before shutdown."""
        try:
            final_stats = {
                'session_stats': self.stats.copy(),
                'final_equity': self.stats['current_equity'],
                'total_runtime_hours': self.stats['uptime_seconds'] / 3600,
                'shutdown_time': datetime.now().isoformat(),
                'positions_at_shutdown': len(self.open_positions),
                'final_positions': {symbol: asdict(pos) for symbol, pos in self.open_positions.items()}
            }
            
            # Save to file (optional)
            try:
                import json
                from pathlib import Path
                
                stats_file = Path('data/session_stats.json')
                stats_file.parent.mkdir(exist_ok=True)
                
                with open(stats_file, 'w') as f:
                    json.dump(final_stats, f, indent=2, default=str)
                
                self.logger.info(f"💾 Final statistics saved to {stats_file}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to save statistics file: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save final statistics: {e}")
    
    async def _log_shutdown_summary(self):
        """Log comprehensive shutdown summary."""
        try:
            runtime_hours = self.stats['uptime_seconds'] / 3600
            
            self.logger.info("📊 === SHUTDOWN SUMMARY ===")
            self.logger.info(f"⏱️ Runtime: {runtime_hours:.2f} hours")
            self.logger.info(f"📈 Total Trades: {self.stats['total_trades']}")
            self.logger.info(f"✅ Success Rate: {self.stats['win_rate']:.1f}%")
            self.logger.info(f"💰 Final P&L: ${self.stats['total_pnl']:.2f}")
            self.logger.info(f"💹 Final Equity: ${self.stats['current_equity']:.2f}")
            self.logger.info(f"📉 Max Drawdown: {self.stats['max_drawdown']*100:.2f}%")
            self.logger.info(f"🎯 Signals Generated: {self.stats['signals_generated']}")
            self.logger.info(f"⚡ Signals Executed: {self.stats['signals_executed']}")
            self.logger.info(f"📊 Open Positions: {len(self.open_positions)}")
            self.logger.info(f"❌ Error Count: {self.error_count}")
            self.logger.info("📊 === END SUMMARY ===")
            
        except Exception as e:
            self.logger.error(f"❌ Shutdown summary error: {e}")
    
    async def _log_emergency_stop(self):
        """Log detailed emergency stop information."""
        try:
            emergency_data = {
                'timestamp': datetime.now().isoformat(),
                'reason': self.last_error or 'Manual trigger',
                'bot_state': self.state.value,
                'active_orders': len(self.active_orders),
                'open_positions': len(self.open_positions),
                'current_equity': self.stats['current_equity'],
                'total_pnl': self.stats['total_pnl'],
                'error_count': self.error_count,
                'uptime_seconds': self.stats['uptime_seconds']
            }
            
            self.logger.error(f"🚨 EMERGENCY STOP LOG:")
            for key, value in emergency_data.items():
                self.logger.error(f"  {key}: {value}")
            
            # Save emergency log
            try:
                emergency_file = Path('logs/emergency_stop.json')
                emergency_file.parent.mkdir(exist_ok=True)
                
                with open(emergency_file, 'w') as f:
                    json.dump(emergency_data, f, indent=2, default=str)
                
            except Exception as e:
                self.logger.error(f"Failed to save emergency log: {e}")
                
        except Exception as e:
            self.logger.error(f"❌ Emergency logging error: {e}")
    
    # Pause/Resume functionality
    
    async def pause_trading(self):
        """Pause trading operations."""
        if self.state != BotState.RUNNING:
            return
        
        self.logger.info("⏸️ Pausing trading operations...")
        self.is_paused = True
        self.state = BotState.PAUSED
        
        # Cancel pending orders but keep positions
        await self._cancel_all_pending_orders()
        
        self.logger.info("⏸️ Trading paused successfully")
    
    async def resume_trading(self):
        """Resume trading operations."""
        if self.state != BotState.PAUSED:
            return
        
        self.logger.info("▶️ Resuming trading operations...")
        self.is_paused = False
        self.state = BotState.RUNNING
        
        self.logger.info("▶️ Trading resumed successfully")
    
    # Status and monitoring
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status."""
        try:
            uptime = 0
            if self.start_time:
                uptime = int((datetime.now() - self.start_time).total_seconds())
            
            # Get component status
            component_status = {
                'exchange': 'connected' if self.exchange else 'disconnected',
                'data_fetcher': 'active' if self.data_fetcher else 'inactive',
                'strategies': f"{len(self.strategies)} loaded",
                'risk_management': 'active' if self.risk_calculator else 'inactive'
            }
            
            return {
                'bot_info': {
                    'state': self.state.value,
                    'trading_mode': self.trading_mode.value,
                    'is_running': self.is_running,
                    'is_paused': self.is_paused,
                    'emergency_stop_triggered': self.emergency_stop_triggered,
                    'start_time': self.start_time.isoformat() if self.start_time else None,
                    'uptime_seconds': uptime,
                    'last_heartbeat': self.last_heartbeat.isoformat(),
                    'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
                    'last_data_update': self.last_data_update.isoformat() if self.last_data_update else None
                },
                'trading_info': {
                    'symbol': self.settings.trading.default_symbol,
                    'timeframe': self.settings.trading.default_timeframe,
                    'current_price': self.current_price,
                    'active_orders': len(self.active_orders),
                    'open_positions': len(self.open_positions),
                    'max_positions': self.settings.trading.max_positions
                },
                'performance': self.stats.copy(),
                'positions': {
                    symbol: {
                        'side': pos.side,
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'unrealized_pnl_pct': (pos.unrealized_pnl / (pos.entry_price * pos.quantity)) * 100 if pos.quantity > 0 else 0,
                        'entry_time': pos.entry_time.isoformat() if pos.entry_time else None
                    }
                    for symbol, pos in self.open_positions.items()
                },
                'components': component_status,
                'risk_info': {
                    'daily_loss_limit': self.daily_loss_limit,
                    'max_drawdown': self.stats['max_drawdown'],
                    'current_equity': self.stats['current_equity'],
                    'max_equity': self.stats['max_equity']
                },
                'errors': {
                    'error_count': self.error_count,
                    'last_error': self.last_error,
                    'error_threshold': self.error_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Status generation error: {e}")
            return {'error': str(e)}
    
    def __repr__(self) -> str:
        """String representation of the controller."""
        return f"TradingBotController(state={self.state.value}, mode={self.trading_mode.value})"