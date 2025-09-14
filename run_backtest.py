#!/usr/bin/env python3
"""
Backtest Runner
===============

Simple script to run backtests using the trading bot's backtesting system.
Supports multiple strategies, data sources, and comprehensive performance analysis.

Usage:
    python run_backtest.py --strategy ema_crossover --symbol BTCUSDT --days 365
    python run_backtest.py --strategy simple_ma --data-file data/btc_data.csv
    python run_backtest.py --list-strategies
"""

import sys
import argparse
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import backtesting components
from backtesting import BacktestEngine, BacktestConfig, DataLoader, PerformanceCalculator
from strategies.trend_following.ema_crossover import EMACrossoverStrategy
from strategies.base_strategy import BaseStrategy, SignalType, SignalStrength, Signal


class SimpleMAStrategy(BaseStrategy):
    """Simple Moving Average crossover strategy for backtesting."""
    
    def __init__(self, symbol: str = "BTCUSDT", timeframe: str = "1h", **kwargs):
        super().__init__(symbol, timeframe, **kwargs)
        self.name = "Simple MA Crossover"
        self.description = "Simple moving average crossover strategy"
        
        # Strategy parameters
        self.parameters = {
            'fast_ma': kwargs.get('fast_ma', 10),
            'slow_ma': kwargs.get('slow_ma', 30),
            'stop_loss_pct': kwargs.get('stop_loss_pct', 2.0),
            'take_profit_pct': kwargs.get('take_profit_pct', 4.0)
        }
    
    async def initialize(self) -> bool:
        """Initialize the strategy."""
        self.is_initialized = True
        return True
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate moving averages."""
        indicators = {}
        indicators['fast_ma'] = data['close'].rolling(window=self.parameters['fast_ma']).mean()
        indicators['slow_ma'] = data['close'].rolling(window=self.parameters['slow_ma']).mean()
        return indicators
    
    async def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate trading signals based on MA crossover."""
        if len(data) < self.parameters['slow_ma'] + 5:
            return None
        
        indicators = self.calculate_indicators(data)
        
        # Check for crossover
        current_fast = indicators['fast_ma'].iloc[-1]
        current_slow = indicators['slow_ma'].iloc[-1]
        prev_fast = indicators['fast_ma'].iloc[-2]
        prev_slow = indicators['slow_ma'].iloc[-2]
        
        current_price = float(data['close'].iloc[-1])
        
        # Bullish crossover
        if prev_fast <= prev_slow and current_fast > current_slow:
            stop_loss = current_price * (1 - self.parameters['stop_loss_pct'] / 100)
            take_profit = current_price * (1 + self.parameters['take_profit_pct'] / 100)
            
            return Signal(
                type=SignalType.BUY,
                strength=SignalStrength.MODERATE,
                price=current_price,
                timestamp=data.index[-1],
                confidence=0.7,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        # Bearish crossover (only if we have a long position)
        elif prev_fast >= prev_slow and current_fast < current_slow:
            if self.current_position == "long":
                return Signal(
                    type=SignalType.SELL,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    timestamp=data.index[-1],
                    confidence=0.7
                )
        
        return None
    
    def validate_signal(self, signal: Signal, data: pd.DataFrame) -> bool:
        """Validate generated signals."""
        return signal is not None and signal.confidence > 0.5


class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy."""
    
    def __init__(self, symbol: str = "BTCUSDT", timeframe: str = "1h", **kwargs):
        super().__init__(symbol, timeframe, **kwargs)
        self.name = "RSI Mean Reversion"
        self.description = "RSI-based mean reversion strategy"
        
        self.parameters = {
            'rsi_period': kwargs.get('rsi_period', 14),
            'oversold_level': kwargs.get('oversold_level', 30),
            'overbought_level': kwargs.get('overbought_level', 70),
            'stop_loss_pct': kwargs.get('stop_loss_pct', 2.0),
            'take_profit_pct': kwargs.get('take_profit_pct', 3.0)
        }
    
    async def initialize(self) -> bool:
        """Initialize the strategy."""
        self.is_initialized = True
        return True
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate RSI indicator."""
        indicators = {}
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.parameters['rsi_period']).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        return indicators
    
    async def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate RSI-based signals."""
        if len(data) < self.parameters['rsi_period'] + 5:
            return None
        
        indicators = self.calculate_indicators(data)
        current_rsi = indicators['rsi'].iloc[-1]
        current_price = float(data['close'].iloc[-1])
        
        # Oversold - Buy signal
        if current_rsi < self.parameters['oversold_level']:
            stop_loss = current_price * (1 - self.parameters['stop_loss_pct'] / 100)
            take_profit = current_price * (1 + self.parameters['take_profit_pct'] / 100)
            
            return Signal(
                type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                price=current_price,
                timestamp=data.index[-1],
                confidence=0.8,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        # Overbought - Sell signal (if we have a position)
        elif current_rsi > self.parameters['overbought_level']:
            if self.current_position == "long":
                return Signal(
                    type=SignalType.SELL,
                    strength=SignalStrength.STRONG,
                    price=current_price,
                    timestamp=data.index[-1],
                    confidence=0.8
                )
        
        return None
    
    def validate_signal(self, signal: Signal, data: pd.DataFrame) -> bool:
        """Validate RSI signals."""
        return signal is not None and signal.confidence > 0.6


class BacktestRunner:
    """Main backtest runner class."""
    
    def __init__(self):
        """Initialize the backtest runner."""
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader()
        self.performance_calculator = PerformanceCalculator()
        
        # Available strategies
        self.strategies = {
            'ema_crossover': EMACrossoverStrategy,
            'simple_ma': SimpleMAStrategy,
            'rsi': RSIStrategy
        }
    
    def list_strategies(self):
        """List available strategies."""
        print("\n📊 Available Strategies:")
        print("=" * 50)
        
        for name, strategy_class in self.strategies.items():
            print(f"\n🎯 {name.upper()}")
            print(f"   Class: {strategy_class.__name__}")
            print(f"   Name: {strategy_class.name}")
            print(f"   Description: {strategy_class.description}")
            
            # Show default parameters
            try:
                temp_strategy = strategy_class()
                params = temp_strategy.parameters
                if params:
                    print(f"   Parameters: {params}")
            except:
                pass
    
    async def run_backtest(self, 
                          strategy_name: str,
                          symbol: str = "BTCUSDT",
                          days: int = 365,
                          initial_capital: float = 10000.0,
                          data_file: Optional[str] = None,
                          strategy_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a backtest with the specified parameters.
        
        Args:
            strategy_name: Name of the strategy to use
            symbol: Trading symbol
            days: Number of days of data to use
            initial_capital: Starting capital
            data_file: Optional CSV file with historical data
            strategy_params: Optional strategy parameters
            
        Returns:
            Dict with backtest results
        """
        try:
            self.logger.info(f"🚀 Starting backtest: {strategy_name} on {symbol}")
            
            # Load data
            if data_file:
                self.logger.info(f"📁 Loading data from file: {data_file}")
                data = self.data_loader.load_csv_data(data_file)
            else:
                self.logger.info(f"📊 Generating {days} days of sample data for {symbol}")
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                data = self.data_loader.load_sample_data(symbol, start_date, end_date)
            
            self.logger.info(f"📈 Data loaded: {len(data)} periods from {data.index[0]} to {data.index[-1]}")
            
            # Create strategy
            if strategy_name not in self.strategies:
                raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(self.strategies.keys())}")
            
            strategy_class = self.strategies[strategy_name]
            strategy_params = strategy_params or {}
            strategy = strategy_class(symbol=symbol, **strategy_params)
            
            self.logger.info(f"🎯 Strategy: {strategy.name}")
            self.logger.info(f"⚙️ Parameters: {strategy.parameters}")
            
            # Configure backtest
            config = BacktestConfig(
                initial_capital=initial_capital,
                commission_rate=0.001,  # 0.1%
                slippage_rate=0.0005,   # 0.05%
                max_positions=1,
                position_sizing="percent_capital",
                position_size=50.0,  # 50% of capital per position
                enable_short_selling=False
            )
            
            # Create backtest engine
            engine = BacktestEngine(config)
            
            # Run backtest
            self.logger.info("🔄 Running backtest...")
            result = await engine.run_strategy_backtest(strategy, data, symbol)
            
            # Generate performance report
            self.logger.info("📊 Calculating performance metrics...")
            performance_report = self.performance_calculator.calculate_performance(
                portfolio_values=result.portfolio_values,
                trades=[trade.__dict__ for trade in result.trades],
                initial_capital=initial_capital,
                benchmark_values=result.benchmark_comparison.get('benchmark_values', None)
            )
            
            # Update result with performance report
            result.performance_report = performance_report
            
            self.logger.info("✅ Backtest completed successfully!")
            
            return {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'config': config.__dict__,
                'result': result,
                'performance_report': performance_report,
                'data_info': {
                    'periods': len(data),
                    'start_date': data.index[0],
                    'end_date': data.index[-1],
                    'price_range': f"${data['low'].min():.2f} - ${data['high'].max():.2f}"
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            raise
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted backtest results."""
        result = results['result']
        performance = results['performance_report']
        data_info = results['data_info']
        
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS")
        print("="*80)
        
        print(f"\n🎯 Strategy: {results['strategy_name'].upper()}")
        print(f"📈 Symbol: {results['symbol']}")
        print(f"📅 Period: {data_info['start_date']} to {data_info['end_date']} ({data_info['periods']} periods)")
        print(f"💰 Initial Capital: ${results['config']['initial_capital']:,.2f}")
        print(f"💰 Final Value: ${result.portfolio_values.iloc[-1]:,.2f}")
        print(f"📊 Price Range: {data_info['price_range']}")
        
        # Print performance summary
        if performance:
            performance.print_summary()
        
        # Print trade summary
        print(f"\n📋 TRADE SUMMARY:")
        print(f"  Total Trades: {len(result.trades)}")
        
        if result.trades:
            winning_trades = [t for t in result.trades if t.pnl > 0]
            losing_trades = [t for t in result.trades if t.pnl < 0]
            
            print(f"  Winning Trades: {len(winning_trades)}")
            print(f"  Losing Trades: {len(losing_trades)}")
            print(f"  Win Rate: {len(winning_trades)/len(result.trades)*100:.1f}%")
            
            if winning_trades:
                avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
                print(f"  Average Win: ${avg_win:.2f}")
            
            if losing_trades:
                avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
                print(f"  Average Loss: ${avg_loss:.2f}")
        
        print("="*80)
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save backtest results to file."""
        try:
            import json
            
            # Prepare data for JSON serialization
            save_data = {
                'strategy_name': results['strategy_name'],
                'symbol': results['symbol'],
                'config': results['config'],
                'data_info': results['data_info'],
                'performance_report': results['performance_report'].to_dict() if results['performance_report'] else None,
                'trade_summary': {
                    'total_trades': len(results['result'].trades),
                    'winning_trades': len([t for t in results['result'].trades if t.pnl > 0]),
                    'losing_trades': len([t for t in results['result'].trades if t.pnl < 0]),
                    'total_pnl': sum(t.pnl for t in results['result'].trades),
                    'final_value': float(results['result'].portfolio_values.iloc[-1])
                }
            }
            
            # Save to JSON file
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            self.logger.info(f"💾 Results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")


async def main():
    """Main function to run backtests from command line."""
    parser = argparse.ArgumentParser(
        description='Run backtests using the trading bot system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py --strategy ema_crossover --symbol BTCUSDT --days 365
  python run_backtest.py --strategy simple_ma --data-file data/btc_data.csv
  python run_backtest.py --strategy rsi --symbol ETHUSDT --days 180 --capital 50000
  python run_backtest.py --list-strategies
        """
    )
    
    parser.add_argument('--strategy', '-s', 
                       choices=['ema_crossover', 'simple_ma', 'rsi'],
                       help='Strategy to backtest')
    
    parser.add_argument('--symbol', default='BTCUSDT',
                       help='Trading symbol (default: BTCUSDT)')
    
    parser.add_argument('--days', '-d', type=int, default=365,
                       help='Number of days of data to use (default: 365)')
    
    parser.add_argument('--capital', '-c', type=float, default=10000.0,
                       help='Initial capital (default: 10000)')
    
    parser.add_argument('--data-file', '-f',
                       help='CSV file with historical data')
    
    parser.add_argument('--output', '-o',
                       help='Output file to save results (JSON format)')
    
    parser.add_argument('--list-strategies', action='store_true',
                       help='List available strategies')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Log level (default: INFO)')
    
    # Strategy-specific parameters
    parser.add_argument('--fast-ma', type=int, help='Fast MA period (for simple_ma strategy)')
    parser.add_argument('--slow-ma', type=int, help='Slow MA period (for simple_ma strategy)')
    parser.add_argument('--fast-ema', type=int, help='Fast EMA period (for ema_crossover strategy)')
    parser.add_argument('--slow-ema', type=int, help='Slow EMA period (for ema_crossover strategy)')
    parser.add_argument('--rsi-period', type=int, help='RSI period (for rsi strategy)')
    parser.add_argument('--stop-loss', type=float, help='Stop loss percentage')
    parser.add_argument('--take-profit', type=float, help='Take profit percentage')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create backtest runner
    runner = BacktestRunner()
    
    # List strategies if requested
    if args.list_strategies:
        runner.list_strategies()
        return
    
    # Validate required arguments
    if not args.strategy:
        print("❌ Error: --strategy is required")
        parser.print_help()
        return
    
    try:
        # Prepare strategy parameters
        strategy_params = {}
        if args.fast_ma:
            strategy_params['fast_ma'] = args.fast_ma
        if args.slow_ma:
            strategy_params['slow_ma'] = args.slow_ma
        if args.fast_ema:
            strategy_params['fast_ema'] = args.fast_ema
        if args.slow_ema:
            strategy_params['slow_ema'] = args.slow_ema
        if args.rsi_period:
            strategy_params['rsi_period'] = args.rsi_period
        if args.stop_loss:
            strategy_params['stop_loss_pct'] = args.stop_loss
        if args.take_profit:
            strategy_params['take_profit_pct'] = args.take_profit
        
        # Run backtest
        results = await runner.run_backtest(
            strategy_name=args.strategy,
            symbol=args.symbol,
            days=args.days,
            initial_capital=args.capital,
            data_file=args.data_file,
            strategy_params=strategy_params
        )
        
        # Print results
        runner.print_results(results)
        
        # Save results if requested
        if args.output:
            runner.save_results(results, args.output)
        
    except Exception as e:
        logging.error(f"❌ Backtest failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
