#!/usr/bin/env python3
"""
Example Backtest
================

Simple example showing how to run a backtest using the trading bot system.
This script demonstrates the basic usage of the backtesting framework.
"""

import asyncio
import logging
from datetime import datetime, timedelta

# Import backtesting components
from backtesting import BacktestEngine, BacktestConfig, DataLoader
from strategies.trend_following.ema_crossover import EMACrossoverStrategy


async def run_example_backtest():
    """Run a simple example backtest."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Example Backtest")
    
    try:
        # 1. Load sample data
        logger.info("📊 Loading sample data...")
        data_loader = DataLoader()
        
        # Generate 6 months of sample BTCUSDT data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        
        data = data_loader.load_sample_data("BTCUSDT", start_date, end_date)
        logger.info(f"📈 Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")
        
        # 2. Create strategy
        logger.info("🎯 Creating EMA Crossover strategy...")
        strategy = EMACrossoverStrategy(
            symbol="BTCUSDT",
            timeframe="1h",
            fast_ema=12,
            slow_ema=26,
            stop_loss_pct=2.0,
            take_profit_pct=4.0
        )
        
        # 3. Configure backtest
        logger.info("⚙️ Configuring backtest...")
        config = BacktestConfig(
            initial_capital=10000.0,
            commission_rate=0.001,  # 0.1%
            slippage_rate=0.0005,   # 0.05%
            max_positions=1,
            position_sizing="percent_capital",
            position_size=50.0,  # 50% of capital per position
            enable_short_selling=False
        )
        
        # 4. Run backtest
        logger.info("🔄 Running backtest...")
        engine = BacktestEngine(config)
        result = await engine.run_strategy_backtest(strategy, data, "BTCUSDT")
        
        # 5. Display results
        logger.info("📊 Backtest Results:")
        logger.info("=" * 50)
        
        final_value = result.portfolio_values.iloc[-1]
        total_return = (final_value - config.initial_capital) / config.initial_capital * 100
        
        logger.info(f"💰 Initial Capital: ${config.initial_capital:,.2f}")
        logger.info(f"💰 Final Value: ${final_value:,.2f}")
        logger.info(f"📈 Total Return: {total_return:.2f}%")
        logger.info(f"📊 Total Trades: {len(result.trades)}")
        
        if result.trades:
            winning_trades = [t for t in result.trades if t.pnl > 0]
            win_rate = len(winning_trades) / len(result.trades) * 100
            logger.info(f"🎯 Win Rate: {win_rate:.1f}%")
            
            total_pnl = sum(t.pnl for t in result.trades)
            logger.info(f"💵 Total P&L: ${total_pnl:.2f}")
        
        # 6. Performance metrics
        if result.performance_report:
            logger.info("\n📊 Performance Metrics:")
            logger.info(f"   Sharpe Ratio: {result.performance_report.sharpe_ratio:.2f}")
            logger.info(f"   Max Drawdown: {result.performance_report.max_drawdown:.2%}")
            logger.info(f"   Annualized Return: {result.performance_report.annualized_return:.2%}")
            logger.info(f"   Annualized Volatility: {result.performance_report.annualized_volatility:.2%}")
        
        # 7. Benchmark comparison
        if result.benchmark_comparison:
            logger.info("\n📈 Benchmark Comparison:")
            for key, value in result.benchmark_comparison.items():
                if isinstance(value, float):
                    logger.info(f"   {key}: {value:.2%}")
                else:
                    logger.info(f"   {key}: {value}")
        
        logger.info("✅ Example backtest completed successfully!")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        raise


async def run_multiple_strategies():
    """Run backtests for multiple strategies and compare results."""
    
    logger = logging.getLogger(__name__)
    logger.info("🔄 Running Multiple Strategy Comparison")
    
    # Load data once
    data_loader = DataLoader()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    data = data_loader.load_sample_data("BTCUSDT", start_date, end_date)
    
    # Define strategies to test
    strategies = [
        {
            'name': 'EMA Crossover (12,26)',
            'strategy': EMACrossoverStrategy(
                symbol="BTCUSDT",
                fast_ema=12,
                slow_ema=26,
                stop_loss_pct=2.0,
                take_profit_pct=4.0
            )
        },
        {
            'name': 'EMA Crossover (8,21)',
            'strategy': EMACrossoverStrategy(
                symbol="BTCUSDT",
                fast_ema=8,
                slow_ema=21,
                stop_loss_pct=1.5,
                take_profit_pct=3.0
            )
        }
    ]
    
    # Common config
    config = BacktestConfig(
        initial_capital=10000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_positions=1,
        position_sizing="percent_capital",
        position_size=50.0
    )
    
    results = []
    
    for strategy_info in strategies:
        logger.info(f"🎯 Testing: {strategy_info['name']}")
        
        engine = BacktestEngine(config)
        result = await engine.run_strategy_backtest(
            strategy_info['strategy'], 
            data, 
            "BTCUSDT"
        )
        
        final_value = result.portfolio_values.iloc[-1]
        total_return = (final_value - config.initial_capital) / config.initial_capital * 100
        
        results.append({
            'name': strategy_info['name'],
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': len(result.trades),
            'win_rate': len([t for t in result.trades if t.pnl > 0]) / len(result.trades) * 100 if result.trades else 0,
            'sharpe_ratio': result.performance_report.sharpe_ratio if result.performance_report else 0,
            'max_drawdown': result.performance_report.max_drawdown if result.performance_report else 0
        })
    
    # Display comparison
    logger.info("\n📊 Strategy Comparison Results:")
    logger.info("=" * 80)
    logger.info(f"{'Strategy':<25} {'Return':<10} {'Trades':<8} {'Win Rate':<10} {'Sharpe':<8} {'Max DD':<10}")
    logger.info("-" * 80)
    
    for result in results:
        logger.info(
            f"{result['name']:<25} "
            f"{result['total_return']:>8.2f}% "
            f"{result['total_trades']:>6} "
            f"{result['win_rate']:>8.1f}% "
            f"{result['sharpe_ratio']:>6.2f} "
            f"{result['max_drawdown']:>8.2%}"
        )
    
    # Find best strategy
    best_strategy = max(results, key=lambda x: x['total_return'])
    logger.info(f"\n🏆 Best Strategy: {best_strategy['name']} ({best_strategy['total_return']:.2f}% return)")


if __name__ == "__main__":
    print("🎯 Trading Bot Backtest Examples")
    print("=" * 40)
    
    # Run single strategy example
    print("\n1️⃣ Single Strategy Example:")
    asyncio.run(run_example_backtest())
    
    # Run multiple strategies comparison
    print("\n2️⃣ Multiple Strategy Comparison:")
    asyncio.run(run_multiple_strategies())
    
    print("\n✅ All examples completed!")
    print("\n💡 To run your own backtests, use:")
    print("   python run_backtest.py --strategy ema_crossover --symbol BTCUSDT --days 365")
    print("   python run_backtest.py --list-strategies")
