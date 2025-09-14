# 🚀 Trading Bot Backtesting Guide

This guide shows you how to run backtests using the trading bot's comprehensive backtesting system.

## 📋 Quick Start

### 1. Run a Simple Backtest

```bash
# Run EMA Crossover strategy on BTCUSDT for 365 days
python run_backtest.py --strategy ema_crossover --symbol BTCUSDT --days 365

# Run Simple MA strategy with custom parameters
python run_backtest.py --strategy simple_ma --fast-ma 10 --slow-ma 30 --days 180

# Run RSI strategy on Ethereum
python run_backtest.py --strategy rsi --symbol ETHUSDT --days 90
```

### 2. List Available Strategies

```bash
python run_backtest.py --list-strategies
```

### 3. Run Backtest from Main Script

```bash
# Using the main trading bot script
python main.py --backtest --strategy ema_crossover --backtest-days 365
```

## 🎯 Available Strategies

### 1. EMA Crossover (`ema_crossover`)
- **Description**: Exponential Moving Average crossover strategy
- **Parameters**: 
  - `fast_ema`: Fast EMA period (default: 12)
  - `slow_ema`: Slow EMA period (default: 26)
  - `stop_loss_pct`: Stop loss percentage (default: 2.0)
  - `take_profit_pct`: Take profit percentage (default: 4.0)

### 2. Simple MA (`simple_ma`)
- **Description**: Simple Moving Average crossover strategy
- **Parameters**:
  - `fast_ma`: Fast MA period (default: 10)
  - `slow_ma`: Slow MA period (default: 30)
  - `stop_loss_pct`: Stop loss percentage (default: 2.0)
  - `take_profit_pct`: Take profit percentage (default: 4.0)

### 3. RSI Mean Reversion (`rsi`)
- **Description**: RSI-based mean reversion strategy
- **Parameters**:
  - `rsi_period`: RSI calculation period (default: 14)
  - `oversold_level`: RSI oversold threshold (default: 30)
  - `overbought_level`: RSI overbought threshold (default: 70)
  - `stop_loss_pct`: Stop loss percentage (default: 2.0)
  - `take_profit_pct`: Take profit percentage (default: 3.0)

## 📊 Command Line Options

### Basic Options
- `--strategy, -s`: Strategy to use (required)
- `--symbol`: Trading symbol (default: BTCUSDT)
- `--days, -d`: Number of days of data (default: 365)
- `--capital, -c`: Initial capital (default: 10000)
- `--data-file, -f`: CSV file with historical data
- `--output, -o`: Save results to JSON file
- `--log-level`: Log level (DEBUG, INFO, WARNING, ERROR)

### Strategy Parameters
- `--fast-ma`: Fast MA period (for simple_ma)
- `--slow-ma`: Slow MA period (for simple_ma)
- `--fast-ema`: Fast EMA period (for ema_crossover)
- `--slow-ema`: Slow EMA period (for ema_crossover)
- `--rsi-period`: RSI period (for rsi)
- `--stop-loss`: Stop loss percentage
- `--take-profit`: Take profit percentage

## 📈 Example Commands

### Basic Backtests
```bash
# EMA Crossover with default parameters
python run_backtest.py --strategy ema_crossover

# Simple MA with custom periods
python run_backtest.py --strategy simple_ma --fast-ma 5 --slow-ma 20

# RSI strategy with custom levels
python run_backtest.py --strategy rsi --rsi-period 21 --oversold-level 25 --overbought-level 75
```

### Advanced Backtests
```bash
# High capital backtest
python run_backtest.py --strategy ema_crossover --capital 100000 --days 730

# Custom data file
python run_backtest.py --strategy simple_ma --data-file data/my_data.csv

# Save results to file
python run_backtest.py --strategy rsi --output results/rsi_backtest.json

# Debug mode
python run_backtest.py --strategy ema_crossover --log-level DEBUG
```

## 📊 Understanding Results

The backtest results include:

### Portfolio Performance
- **Total Return**: Overall percentage return
- **Annualized Return**: Return annualized
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Price volatility measure

### Trade Statistics
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Average Win/Loss**: Average profit/loss per trade
- **Largest Win/Loss**: Best and worst individual trades

### Risk Metrics
- **VaR (95%)**: Value at Risk at 95% confidence
- **CVaR**: Conditional Value at Risk
- **Downside Deviation**: Volatility of negative returns
- **Sortino Ratio**: Risk-adjusted return using downside deviation

### Benchmark Comparison
- **Alpha**: Excess return over benchmark
- **Beta**: Sensitivity to market movements
- **Information Ratio**: Risk-adjusted excess return
- **Tracking Error**: Volatility of excess returns

## 🔧 Custom Data

### Using Your Own Data
Create a CSV file with the following format:
```csv
datetime,open,high,low,close,volume
2023-01-01 00:00:00,16000.0,16100.0,15900.0,16050.0,1000000
2023-01-01 01:00:00,16050.0,16150.0,16000.0,16100.0,1200000
...
```

Then run:
```bash
python run_backtest.py --strategy ema_crossover --data-file your_data.csv
```

## 🎮 Interactive Examples

### Run Example Scripts
```bash
# Run comprehensive examples
python example_backtest.py

# This will show:
# 1. Single strategy backtest
# 2. Multiple strategy comparison
# 3. Performance analysis
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Unknown strategy" error**
   - Use `--list-strategies` to see available strategies
   - Check spelling of strategy name

2. **"Insufficient data" error**
   - Increase `--days` parameter
   - Ensure data file has enough historical data

3. **"No trades generated"**
   - Strategy parameters may be too restrictive
   - Try different parameter values
   - Check if data has sufficient volatility

4. **Performance issues**
   - Reduce `--days` for faster testing
   - Use `--log-level WARNING` to reduce output

### Getting Help
```bash
# Show help
python run_backtest.py --help

# List strategies
python run_backtest.py --list-strategies

# Run with debug logging
python run_backtest.py --strategy ema_crossover --log-level DEBUG
```

## 📚 Next Steps

1. **Experiment with Parameters**: Try different strategy parameters to optimize performance
2. **Compare Strategies**: Run multiple strategies and compare results
3. **Custom Strategies**: Create your own strategies by extending `BaseStrategy`
4. **Data Analysis**: Use the saved JSON results for further analysis
5. **Live Trading**: Once satisfied with backtest results, consider live trading

## 🎯 Tips for Better Backtests

1. **Use Sufficient Data**: At least 6 months of data for reliable results
2. **Include Transaction Costs**: Commission and slippage are included by default
3. **Test Multiple Timeframes**: Try different time periods
4. **Validate Out-of-Sample**: Test on unseen data
5. **Consider Market Conditions**: Results may vary in different market environments

Happy backtesting! 🚀📈
