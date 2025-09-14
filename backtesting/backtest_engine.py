"""
Backtest Engine
===============

Complete backtesting engine for trading strategies.
Simulates trading with historical data, tracks portfolio performance,
and generates comprehensive results.

Features:
- Strategy backtesting with historical data
- Portfolio tracking and performance monitoring
- Trade simulation with commission and slippage
- Position management (long/short)
- Risk management integration
- Detailed trade logging and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

# Import local modules
from .data_loader import DataLoader
from .performance_metrics import PerformanceCalculator, PerformanceReport


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    
    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized PnL based on current price."""
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float = 0.0
    slippage: float = 0.0
    duration_hours: float = 0.0
    
    def __post_init__(self):
        """Calculate additional fields after initialization."""
        if self.entry_time and self.exit_time:
            self.duration_hours = (self.exit_time - self.entry_time).total_seconds() / 3600
        
        # Calculate PnL if not provided
        if self.pnl == 0.0:
            if self.side == PositionSide.LONG:
                self.pnl = (self.exit_price - self.entry_price) * self.quantity - self.commission - self.slippage
            else:  # SHORT
                self.pnl = (self.entry_price - self.exit_price) * self.quantity - self.commission - self.slippage


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    max_positions: int = 1
    position_sizing: str = "fixed_amount"  # "fixed_amount", "percent_capital", "risk_parity"
    position_size: float = 1000.0  # Amount per position or percentage
    enable_short_selling: bool = False
    margin_requirement: float = 1.0  # 1.0 = no margin, 0.5 = 2x leverage


@dataclass
class BacktestResult:
    """Complete backtesting results."""
    config: BacktestConfig
    portfolio_values: pd.Series = field(default_factory=pd.Series)
    trades: List[Trade] = field(default_factory=list)
    positions_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    performance_report: Optional[PerformanceReport] = None
    benchmark_comparison: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'config': self.config.__dict__,
            'final_value': float(self.portfolio_values.iloc[-1]) if not self.portfolio_values.empty else 0,
            'total_trades': len(self.trades),
            'performance_report': self.performance_report.to_dict() if self.performance_report else None,
            'benchmark_comparison': self.benchmark_comparison
        }


class Signal:
    """Simple signal class for backtesting."""
    def __init__(self, timestamp: datetime, action: str, price: float, 
                 confidence: float = 1.0, stop_loss: Optional[float] = None, 
                 take_profit: Optional[float] = None):
        self.timestamp = timestamp
        self.action = action.upper()  # BUY, SELL, HOLD, CLOSE
        self.price = price
        self.confidence = confidence
        self.stop_loss = stop_loss
        self.take_profit = take_profit


class BacktestEngine:
    """
    Complete backtesting engine for trading strategies.
    
    Features:
    - Historical data backtesting
    - Multiple position management
    - Commission and slippage simulation
    - Portfolio tracking
    - Performance analysis
    - Risk management
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize the backtest engine."""
        self.config = config or BacktestConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.performance_calculator = PerformanceCalculator()
        
        # Trading state
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self.peak_value = self.config.initial_capital
        self.drawdown = 0.0
        self.trade_count = 0
        
        self.logger.info(f"BacktestEngine initialized with ${self.config.initial_capital:,.2f} capital")
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    strategy_signals: Union[List[Signal], pd.DataFrame],
                    symbol: str = "BTCUSDT") -> BacktestResult:
        """
        Run complete backtesting process.
        
        Args:
            data: Historical OHLCV data
            strategy_signals: Trading signals from strategy
            symbol: Trading symbol
            
        Returns:
            BacktestResult: Complete backtesting results
        """
        try:
            self.logger.info(f"Starting backtest for {symbol}")
            self.logger.info(f"Data period: {data.index[0]} to {data.index[-1]}")
            self.logger.info(f"Total data points: {len(data)}")
            
            # Reset state
            self._reset_state()
            
            # Convert signals if needed
            if isinstance(strategy_signals, pd.DataFrame):
                signals = self._convert_dataframe_to_signals(strategy_signals, data)
            else:
                signals = strategy_signals
            
            self.logger.info(f"Processing {len(signals)} signals")
            
            # Process each timestamp
            signal_index = 0
            
            for timestamp, row in data.iterrows():
                current_price = float(row['close'])
                
                # Update portfolio value and positions
                self._update_portfolio_value(timestamp, current_price, symbol)
                
                # Process signals at this timestamp
                while (signal_index < len(signals) and 
                       signals[signal_index].timestamp <= timestamp):
                    
                    signal = signals[signal_index]
                    self.logger.debug(f"Processing signal at {timestamp}: {signal}")
                    self._process_signal(signal, current_price, timestamp, symbol)
                    signal_index += 1
                
                # Check exit conditions
                self._check_exit_conditions(current_price, timestamp, symbol)
                
                # Record portfolio value
                self.portfolio_history.append((timestamp, self.portfolio_value))
            
            # Create results
            result = self._create_backtest_result(data, symbol)
            
            self.logger.info(f"Backtest completed:")
            self.logger.info(f"  Final value: ${result.portfolio_values.iloc[-1]:,.2f}")
            self.logger.info(f"  Total return: {((result.portfolio_values.iloc[-1] / self.config.initial_capital) - 1) * 100:.2f}%")
            self.logger.info(f"  Total trades: {len(result.trades)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            raise
    
    def _reset_state(self):
        """Reset backtesting state."""
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.portfolio_history.clear()
        self.peak_value = self.config.initial_capital
        self.drawdown = 0.0
        self.trade_count = 0
    
    def _convert_dataframe_to_signals(self, signals_df: pd.DataFrame, 
                                    price_data: pd.DataFrame) -> List[Signal]:
        """Convert DataFrame signals to Signal objects."""
        signals = []
        
        for timestamp, row in signals_df.iterrows():
            # Get price at signal time
            if timestamp in price_data.index:
                price = float(price_data.loc[timestamp]['close'])
            else:
                # Find nearest price
                nearest_idx = price_data.index.get_indexer([timestamp], method='nearest')[0]
                price = float(price_data.iloc[nearest_idx]['close'])
            
            # Determine action from signal data
            action = "HOLD"
            if 'signal' in row:
                if row['signal'] == 1 or row['signal'] == 'BUY':
                    action = "BUY"
                elif row['signal'] == -1 or row['signal'] == 'SELL':
                    action = "SELL"
                elif row['signal'] == 0:
                    action = "CLOSE"
            elif 'action' in row:
                action = str(row['action']).upper()
            
            # Get additional signal parameters
            confidence = float(row.get('confidence', 1.0))
            stop_loss = row.get('stop_loss', None)
            take_profit = row.get('take_profit', None)
            
            if action != "HOLD":
                signal = Signal(
                    timestamp=timestamp,
                    action=action,
                    price=price,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                signals.append(signal)
        
        return signals
    
    def _update_portfolio_value(self, timestamp: datetime, current_price: float, symbol: str):
        """Update portfolio value and position PnL."""
        # Update unrealized PnL for open positions
        total_unrealized_pnl = 0.0
        
        for pos_symbol, position in self.positions.items():
            if pos_symbol == symbol:  # For simplicity, assuming single symbol
                position.update_unrealized_pnl(current_price)
                total_unrealized_pnl += position.unrealized_pnl
        
        # Calculate total portfolio value
        position_value = sum(pos.quantity * current_price for pos in self.positions.values())
        self.portfolio_value = self.cash + position_value + total_unrealized_pnl
        
        # Update peak and drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        self.drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
    
    def _process_signal(self, signal: Signal, current_price: float, 
                       timestamp: datetime, symbol: str):
        """Process a trading signal."""
        try:
            # Handle both signal types (base_strategy.Signal and backtest_engine.Signal)
            signal_action = getattr(signal, 'action', None) or getattr(signal, 'type', None)
            self.logger.debug(f"Signal action: {signal_action}")
            
            if not signal_action:
                self.logger.debug("No signal action found, skipping")
                return
                
            # Handle enum values properly
            if hasattr(signal_action, 'value'):
                signal_action = signal_action.value.upper()
            else:
                signal_action = str(signal_action).upper()
            
            self.logger.debug(f"Processing {signal_action} signal at price {current_price}")
            
            if signal_action in ["BUY", "BUY"]:
                self.logger.debug("Executing BUY signal")
                self._execute_buy_signal(signal, current_price, timestamp, symbol)
            elif signal_action in ["SELL", "SELL"]:
                self.logger.debug("Executing SELL signal")
                if self.config.enable_short_selling:
                    self._execute_sell_signal(signal, current_price, timestamp, symbol)
                else:
                    # If short selling disabled, treat SELL as close long position
                    if symbol in self.positions and self.positions[symbol].side == PositionSide.LONG:
                        self._close_position(symbol, current_price, timestamp, "SELL signal (long close)")
            elif signal_action in ["CLOSE", "CLOSE_LONG", "CLOSE_SHORT"]:
                self.logger.debug("Executing CLOSE signal")
                if symbol in self.positions:
                    self._close_position(symbol, current_price, timestamp, "CLOSE signal")
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    def _execute_buy_signal(self, signal: Signal, current_price: float, 
                           timestamp: datetime, symbol: str):
        """Execute a buy signal."""
        # Check if we can open a new position
        if len(self.positions) >= self.config.max_positions:
            self.logger.debug("Cannot open position: max positions reached")
            return
        
        # Calculate position size
        confidence = getattr(signal, 'confidence', 1.0)
        position_size = self._calculate_position_size(current_price, confidence)
        
        if position_size <= 0:
            self.logger.debug("Position size <= 0, skipping trade")
            return
        
        # Calculate costs
        position_value = position_size * current_price
        commission = position_value * self.config.commission_rate
        slippage = position_value * self.config.slippage_rate
        total_cost = position_value + commission + slippage
        
        # Check if we have enough cash
        if total_cost > self.cash:
            self.logger.debug(f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}")
            return
        
        # Apply slippage to entry price
        entry_price = current_price * (1 + self.config.slippage_rate)
        
        # Create position
        position = Position(
            symbol=symbol,
            side=PositionSide.LONG,
            quantity=position_size,
            entry_price=entry_price,
            entry_time=timestamp,
            stop_loss=getattr(signal, 'stop_loss', None),
            take_profit=getattr(signal, 'take_profit', None)
        )
        
        # Update cash
        self.cash -= total_cost
        
        # Store position
        self.positions[symbol] = position
        
        self.trade_count += 1
        self.logger.debug(f"Opened LONG position: {position_size:.4f} @ ${entry_price:.2f}")
    
    def _execute_sell_signal(self, signal: Signal, current_price: float, 
                            timestamp: datetime, symbol: str):
        """Execute a sell signal (short position)."""
        if not self.config.enable_short_selling:
            return
        
        # Check if we can open a new position
        if len(self.positions) >= self.config.max_positions:
            self.logger.debug("Cannot open position: max positions reached")
            return
        
        # Calculate position size
        confidence = getattr(signal, 'confidence', 1.0)
        position_size = self._calculate_position_size(current_price, confidence)
        
        if position_size <= 0:
            return
        
        # For short selling, we need margin
        position_value = position_size * current_price
        margin_required = position_value * self.config.margin_requirement
        commission = position_value * self.config.commission_rate
        slippage = position_value * self.config.slippage_rate
        total_cost = margin_required + commission + slippage
        
        # Check if we have enough cash for margin
        if total_cost > self.cash:
            self.logger.debug(f"Insufficient cash for short: need ${total_cost:.2f}, have ${self.cash:.2f}")
            return
        
        # Apply slippage to entry price (negative for short)
        entry_price = current_price * (1 - self.config.slippage_rate)
        
        # Create short position
        position = Position(
            symbol=symbol,
            side=PositionSide.SHORT,
            quantity=position_size,
            entry_price=entry_price,
            entry_time=timestamp,
            stop_loss=getattr(signal, 'stop_loss', None),
            take_profit=getattr(signal, 'take_profit', None)
        )
        
        # Update cash (margin requirement)
        self.cash -= total_cost
        
        # Store position
        self.positions[symbol] = position
        
        self.trade_count += 1
        self.logger.debug(f"Opened SHORT position: {position_size:.4f} @ ${entry_price:.2f}")
    
    def _calculate_position_size(self, current_price: float, confidence: float = 1.0) -> float:
        """Calculate position size based on configuration."""
        if self.config.position_sizing == "fixed_amount":
            return self.config.position_size / current_price
        
        elif self.config.position_sizing == "percent_capital":
            available_capital = self.portfolio_value * (self.config.position_size / 100.0) * confidence
            return available_capital / current_price
        
        elif self.config.position_sizing == "risk_parity":
            # Simple risk parity - equal risk per position
            risk_per_trade = self.portfolio_value * 0.02  # 2% risk
            return risk_per_trade / current_price
        
        else:
            return self.config.position_size / current_price
    
    def _check_exit_conditions(self, current_price: float, timestamp: datetime, symbol: str):
        """Check stop loss and take profit conditions."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        should_close = False
        close_reason = ""
        
        # Check stop loss
        if position.stop_loss:
            if position.side == PositionSide.LONG and current_price <= position.stop_loss:
                should_close = True
                close_reason = "Stop Loss"
            elif position.side == PositionSide.SHORT and current_price >= position.stop_loss:
                should_close = True
                close_reason = "Stop Loss"
        
        # Check take profit
        if position.take_profit and not should_close:
            if position.side == PositionSide.LONG and current_price >= position.take_profit:
                should_close = True
                close_reason = "Take Profit"
            elif position.side == PositionSide.SHORT and current_price <= position.take_profit:
                should_close = True
                close_reason = "Take Profit"
        
        if should_close:
            self._close_position(symbol, current_price, timestamp, close_reason)
    
    def _close_position(self, symbol: str, exit_price: float, 
                       timestamp: datetime, reason: str = "Manual"):
        """Close an open position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Apply slippage to exit price
        if position.side == PositionSide.LONG:
            adjusted_exit_price = exit_price * (1 - self.config.slippage_rate)
        else:  # SHORT
            adjusted_exit_price = exit_price * (1 + self.config.slippage_rate)
        
        # Calculate PnL
        if position.side == PositionSide.LONG:
            gross_pnl = (adjusted_exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            gross_pnl = (position.entry_price - adjusted_exit_price) * position.quantity
        
        # Calculate costs
        position_value = position.quantity * adjusted_exit_price
        commission = position_value * self.config.commission_rate
        slippage_cost = position.quantity * abs(exit_price - adjusted_exit_price)
        
        net_pnl = gross_pnl - commission - slippage_cost
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=adjusted_exit_price,
            entry_time=position.entry_time,
            exit_time=timestamp,
            pnl=net_pnl,
            commission=commission,
            slippage=slippage_cost
        )
        
        # Update cash
        if position.side == PositionSide.LONG:
            self.cash += position_value - commission
        else:  # SHORT
            # For short positions, return margin plus PnL
            margin_returned = position.quantity * position.entry_price * self.config.margin_requirement
            self.cash += margin_returned + net_pnl
        
        # Store trade
        self.trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        self.logger.debug(f"Closed {position.side.value} position: PnL ${net_pnl:.2f} ({reason})")
    
    def _create_backtest_result(self, data: pd.DataFrame, symbol: str) -> BacktestResult:
        """Create comprehensive backtest results."""
        # Create portfolio values series
        portfolio_df = pd.DataFrame(self.portfolio_history, columns=['timestamp', 'value'])
        portfolio_df.set_index('timestamp', inplace=True)
        portfolio_values = portfolio_df['value']
        
        # Calculate performance metrics
        performance_report = self.performance_calculator.calculate_performance(
            portfolio_values=portfolio_values,
            trades=[trade.__dict__ for trade in self.trades],
            initial_capital=self.config.initial_capital
        )
        
        # Create benchmark (buy and hold)
        benchmark_values = self._calculate_buy_and_hold_benchmark(data, symbol)
        benchmark_comparison = self.performance_calculator.compare_to_benchmark(
            portfolio_values, benchmark_values
        )
        
        # Create positions history DataFrame
        positions_history = self._create_positions_history()
        
        # Create result object
        result = BacktestResult(
            config=self.config,
            portfolio_values=portfolio_values,
            trades=self.trades.copy(),
            positions_history=positions_history,
            performance_report=performance_report,
            benchmark_comparison=benchmark_comparison
        )
        
        return result
    
    def _calculate_buy_and_hold_benchmark(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Calculate buy and hold benchmark performance."""
        try:
            # Buy at first price, hold until end
            first_price = float(data.iloc[0]['close'])
            shares_bought = self.config.initial_capital / first_price
            
            benchmark_values = []
            for timestamp, row in data.iterrows():
                current_price = float(row['close'])
                portfolio_value = shares_bought * current_price
                benchmark_values.append((timestamp, portfolio_value))
            
            benchmark_df = pd.DataFrame(benchmark_values, columns=['timestamp', 'value'])
            benchmark_df.set_index('timestamp', inplace=True)
            
            return benchmark_df['value']
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmark: {e}")
            # Return flat line if error
            timestamps = data.index
            flat_values = pd.Series(self.config.initial_capital, index=timestamps)
            return flat_values
    
    def _create_positions_history(self) -> pd.DataFrame:
        """Create DataFrame of position history."""
        positions_data = []
        
        for trade in self.trades:
            positions_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'duration_hours': trade.duration_hours,
                'commission': trade.commission,
                'slippage': trade.slippage
            })
        
        if positions_data:
            return pd.DataFrame(positions_data)
        else:
            return pd.DataFrame()
    
    async def run_strategy_backtest(self, strategy, data: pd.DataFrame, 
                            symbol: str = "BTCUSDT") -> BacktestResult:
        """
        Run backtest using a strategy object.
        
        Args:
            strategy: Strategy object with generate_signals method
            data: Historical OHLCV data
            symbol: Trading symbol
            
        Returns:
            BacktestResult: Complete backtesting results
        """
        try:
            self.logger.info(f"Running strategy backtest for {strategy.__class__.__name__}")
            
            # Generate signals from strategy
            signals = []
            
            # Process data in chunks to simulate real-time signal generation
            for i in range(len(data)):
                # Get data up to current point as DataFrame
                current_data = data.iloc[:i+1].copy()
                
                if len(current_data) < 20:  # Need minimum data for indicators
                    continue
                
                # Generate signal at this timestamp
                try:
                    signal = await strategy.generate_signal(current_data)
                    if signal:
                        # Handle both signal types (base_strategy.Signal and backtest_engine.Signal)
                        signal_action = getattr(signal, 'action', None) or getattr(signal, 'type', None)
                        if signal_action:
                            # Handle enum values properly
                            if hasattr(signal_action, 'value'):
                                signal_action_str = signal_action.value.upper()
                            else:
                                signal_action_str = str(signal_action).upper()
                            
                            if signal_action_str not in ["HOLD", "HOLD"]:
                                signals.append(signal)
                except Exception as e:
                    self.logger.debug(f"Strategy signal generation error at {data.index[i]}: {e}")
                    continue
            
            self.logger.info(f"Generated {len(signals)} signals from strategy")
            
            # Run backtest with generated signals
            return self.run_backtest(data, signals, symbol)
            
        except Exception as e:
            self.logger.error(f"Strategy backtest error: {e}")
            raise
    
    def optimize_parameters(self, strategy_class, data: pd.DataFrame, 
                          parameter_ranges: Dict[str, List], 
                          symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Simple parameter optimization for strategies.
        
        Args:
            strategy_class: Strategy class to optimize
            data: Historical data
            parameter_ranges: Dict of parameter names to lists of values to test
            symbol: Trading symbol
            
        Returns:
            Dict with optimization results
        """
        try:
            self.logger.info("Starting parameter optimization...")
            
            best_params = None
            best_score = -float('inf')
            optimization_results = []
            
            # Generate parameter combinations
            from itertools import product
            
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            total_combinations = 1
            for values in param_values:
                total_combinations *= len(values)
            
            self.logger.info(f"Testing {total_combinations} parameter combinations")
            
            for i, combination in enumerate(product(*param_values)):
                # Create parameter dict
                params = dict(zip(param_names, combination))
                
                try:
                    # Create strategy with parameters
                    strategy = strategy_class(**params)
                    
                    # Run backtest
                    result = self.run_strategy_backtest(strategy, data, symbol)
                    
                    # Calculate score (using Sharpe ratio as default)
                    score = result.performance_report.sharpe_ratio if result.performance_report else -999
                    
                    # Store result
                    optimization_results.append({
                        'parameters': params.copy(),
                        'score': score,
                        'total_return': result.performance_report.total_return if result.performance_report else 0,
                        'max_drawdown': result.performance_report.max_drawdown if result.performance_report else 0,
                        'win_rate': result.performance_report.win_rate if result.performance_report else 0,
                        'total_trades': len(result.trades)
                    })
                    
                    # Update best parameters
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Completed {i + 1}/{total_combinations} combinations")
                
                except Exception as e:
                    self.logger.debug(f"Error testing parameters {params}: {e}")
                    continue
            
            # Sort results by score
            optimization_results.sort(key=lambda x: x['score'], reverse=True)
            
            self.logger.info(f"Optimization complete. Best Sharpe ratio: {best_score:.3f}")
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'all_results': optimization_results[:20],  # Top 20 results
                'total_tested': len(optimization_results)
            }
            
        except Exception as e:
            self.logger.error(f"Parameter optimization error: {e}")
            raise
    
    def generate_sample_signals(self, data: pd.DataFrame, 
                              strategy_type: str = "simple_ma") -> List[Signal]:
        """
        Generate sample signals for testing purposes.
        
        Args:
            data: Historical OHLCV data
            strategy_type: Type of simple strategy to use
            
        Returns:
            List[Signal]: Generated signals
        """
        signals = []
        
        if strategy_type == "simple_ma":
            # Simple moving average crossover
            data['ma_short'] = data['close'].rolling(window=10).mean()
            data['ma_long'] = data['close'].rolling(window=30).mean()
            
            prev_signal = 0
            for timestamp, row in data.iterrows():
                if pd.isna(row['ma_short']) or pd.isna(row['ma_long']):
                    continue
                
                current_signal = 1 if row['ma_short'] > row['ma_long'] else -1
                
                # Generate signal on crossover
                if current_signal != prev_signal:
                    action = "BUY" if current_signal == 1 else "SELL"
                    
                    # Simple stop loss and take profit
                    price = float(row['close'])
                    stop_loss = price * 0.95 if current_signal == 1 else price * 1.05
                    take_profit = price * 1.10 if current_signal == 1 else price * 0.90
                    
                    signal = Signal(
                        timestamp=timestamp,
                        action=action,
                        price=price,
                        confidence=0.7,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    signals.append(signal)
                    prev_signal = current_signal
        
        elif strategy_type == "rsi":
            # RSI-based signals
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            data['rsi'] = calculate_rsi(data['close'])
            
            for timestamp, row in data.iterrows():
                if pd.isna(row['rsi']):
                    continue
                
                price = float(row['close'])
                
                if row['rsi'] < 30:  # Oversold
                    signal = Signal(
                        timestamp=timestamp,
                        action="BUY",
                        price=price,
                        confidence=0.8,
                        stop_loss=price * 0.95,
                        take_profit=price * 1.08
                    )
                    signals.append(signal)
                elif row['rsi'] > 70:  # Overbought
                    signal = Signal(
                        timestamp=timestamp,
                        action="SELL",
                        price=price,
                        confidence=0.8,
                        stop_loss=price * 1.05,
                        take_profit=price * 0.92
                    )
                    signals.append(signal)
        
        self.logger.info(f"Generated {len(signals)} sample signals using {strategy_type} strategy")
        return signals