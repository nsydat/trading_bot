"""
Performance Metrics for Backtesting
===================================

Comprehensive performance analysis and metrics calculation for trading strategies.
Calculates portfolio performance, risk metrics, trade statistics, and benchmark comparisons.

Features:
- Portfolio performance metrics (returns, Sharpe ratio, max drawdown)
- Trade analysis (win rate, profit factor, average trade)
- Risk metrics (VaR, volatility, downside deviation)
- Benchmark comparison
- Performance visualization and reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class PerformanceReport:
    """Comprehensive performance report with all calculated metrics."""
    
    # Portfolio Performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown Analysis
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    current_drawdown: float = 0.0
    drawdown_periods: int = 0
    
    # Risk Metrics
    var_95: float = 0.0  # Value at Risk (95% confidence)
    var_99: float = 0.0  # Value at Risk (99% confidence)
    cvar_95: float = 0.0  # Conditional Value at Risk
    downside_deviation: float = 0.0
    upside_deviation: float = 0.0
    
    # Trade Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_trade: float = 0.0
    
    # Trade Duration
    average_win_duration: float = 0.0
    average_loss_duration: float = 0.0
    average_trade_duration: float = 0.0
    
    # Consecutive Trades
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0
    current_streak_type: str = "none"  # "win", "loss", "none"
    
    # Benchmark Comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    
    # Additional Metrics
    recovery_factor: float = 0.0
    sterling_ratio: float = 0.0
    burke_ratio: float = 0.0
    kappa_three: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance report to dictionary."""
        return {
            'portfolio_performance': {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
                'annualized_volatility': self.annualized_volatility,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio
            },
            'drawdown_analysis': {
                'max_drawdown': self.max_drawdown,
                'max_drawdown_duration': self.max_drawdown_duration,
                'current_drawdown': self.current_drawdown,
                'drawdown_periods': self.drawdown_periods
            },
            'risk_metrics': {
                'var_95': self.var_95,
                'var_99': self.var_99,
                'cvar_95': self.cvar_95,
                'downside_deviation': self.downside_deviation,
                'upside_deviation': self.upside_deviation
            },
            'trade_statistics': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'average_win': self.average_win,
                'average_loss': self.average_loss,
                'largest_win': self.largest_win,
                'largest_loss': self.largest_loss,
                'average_trade': self.average_trade
            },
            'trade_duration': {
                'average_win_duration': self.average_win_duration,
                'average_loss_duration': self.average_loss_duration,
                'average_trade_duration': self.average_trade_duration
            },
            'consecutive_trades': {
                'max_consecutive_wins': self.max_consecutive_wins,
                'max_consecutive_losses': self.max_consecutive_losses,
                'current_streak': self.current_streak,
                'current_streak_type': self.current_streak_type
            },
            'benchmark_comparison': {
                'benchmark_return': self.benchmark_return,
                'alpha': self.alpha,
                'beta': self.beta,
                'information_ratio': self.information_ratio,
                'tracking_error': self.tracking_error
            },
            'additional_metrics': {
                'recovery_factor': self.recovery_factor,
                'sterling_ratio': self.sterling_ratio,
                'burke_ratio': self.burke_ratio,
                'kappa_three': self.kappa_three
            }
        }
    
    def print_summary(self):
        """Print a formatted summary of the performance report."""
        print("\n" + "="*60)
        print("PERFORMANCE REPORT SUMMARY")
        print("="*60)
        
        print(f"\n📊 PORTFOLIO PERFORMANCE:")
        print(f"  Total Return:        {self.total_return:>8.2%}")
        print(f"  Annualized Return:   {self.annualized_return:>8.2%}")
        print(f"  Annualized Vol:      {self.annualized_volatility:>8.2%}")
        print(f"  Sharpe Ratio:        {self.sharpe_ratio:>8.2f}")
        print(f"  Sortino Ratio:       {self.sortino_ratio:>8.2f}")
        print(f"  Calmar Ratio:        {self.calmar_ratio:>8.2f}")
        
        print(f"\n📉 DRAWDOWN ANALYSIS:")
        print(f"  Max Drawdown:        {self.max_drawdown:>8.2%}")
        print(f"  Max DD Duration:     {self.max_drawdown_duration:>8} periods")
        print(f"  Current Drawdown:    {self.current_drawdown:>8.2%}")
        print(f"  Drawdown Periods:    {self.drawdown_periods:>8}")
        
        print(f"\n🎯 TRADE STATISTICS:")
        print(f"  Total Trades:        {self.total_trades:>8}")
        print(f"  Win Rate:            {self.win_rate:>8.2%}")
        print(f"  Profit Factor:       {self.profit_factor:>8.2f}")
        print(f"  Average Win:         ${self.average_win:>8.2f}")
        print(f"  Average Loss:        ${self.average_loss:>8.2f}")
        print(f"  Largest Win:         ${self.largest_win:>8.2f}")
        print(f"  Largest Loss:        ${self.largest_loss:>8.2f}")
        
        print(f"\n⏱️  TRADE DURATION:")
        print(f"  Avg Win Duration:    {self.average_win_duration:>8.1f} hours")
        print(f"  Avg Loss Duration:   {self.average_loss_duration:>8.1f} hours")
        print(f"  Avg Trade Duration:  {self.average_trade_duration:>8.1f} hours")
        
        print(f"\n🔄 CONSECUTIVE TRADES:")
        print(f"  Max Consecutive Wins: {self.max_consecutive_wins:>6}")
        print(f"  Max Consecutive Losses: {self.max_consecutive_losses:>4}")
        print(f"  Current Streak:      {self.current_streak:>8} ({self.current_streak_type})")
        
        print(f"\n📈 BENCHMARK COMPARISON:")
        print(f"  Benchmark Return:    {self.benchmark_return:>8.2%}")
        print(f"  Alpha:               {self.alpha:>8.2%}")
        print(f"  Beta:                {self.beta:>8.2f}")
        print(f"  Information Ratio:   {self.information_ratio:>8.2f}")
        
        print(f"\n⚠️  RISK METRICS:")
        print(f"  VaR (95%):           {self.var_95:>8.2%}")
        print(f"  VaR (99%):           {self.var_99:>8.2%}")
        print(f"  CVaR (95%):          {self.cvar_95:>8.2%}")
        print(f"  Downside Deviation:  {self.downside_deviation:>8.2%}")
        
        print("="*60)


class PerformanceCalculator:
    """
    Comprehensive performance calculator for trading strategies.
    
    Calculates portfolio performance, risk metrics, trade statistics,
    and benchmark comparisons for backtesting results.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate_performance(self, 
                            portfolio_values: pd.Series,
                            trades: List[Dict[str, Any]],
                            initial_capital: float,
                            benchmark_values: Optional[pd.Series] = None) -> PerformanceReport:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_values: Series of portfolio values over time
            trades: List of trade dictionaries
            initial_capital: Initial capital amount
            benchmark_values: Optional benchmark portfolio values
            
        Returns:
            PerformanceReport: Complete performance analysis
        """
        try:
            self.logger.info("Calculating performance metrics...")
            
            # Calculate portfolio returns
            returns = self._calculate_returns(portfolio_values)
            
            # Create performance report
            report = PerformanceReport()
            
            # Portfolio performance metrics
            self._calculate_portfolio_metrics(report, returns, initial_capital, portfolio_values)
            
            # Drawdown analysis
            self._calculate_drawdown_metrics(report, portfolio_values)
            
            # Risk metrics
            self._calculate_risk_metrics(report, returns)
            
            # Trade statistics
            self._calculate_trade_metrics(report, trades)
            
            # Benchmark comparison
            if benchmark_values is not None:
                self._calculate_benchmark_metrics(report, returns, benchmark_values)
            
            # Additional metrics
            self._calculate_additional_metrics(report, returns, portfolio_values)
            
            self.logger.info("Performance calculation completed")
            return report
            
        except Exception as e:
            self.logger.error(f"Error calculating performance: {e}")
            raise
    
    def _calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate portfolio returns."""
        if len(portfolio_values) < 2:
            return pd.Series(dtype=float)
        
        returns = portfolio_values.pct_change().dropna()
        return returns
    
    def _calculate_portfolio_metrics(self, report: PerformanceReport, 
                                   returns: pd.Series, 
                                   initial_capital: float,
                                   portfolio_values: pd.Series):
        """Calculate portfolio performance metrics."""
        if len(returns) == 0:
            return
        
        final_value = portfolio_values.iloc[-1]
        
        # Total return
        report.total_return = (final_value - initial_capital) / initial_capital
        
        # Annualized return
        if len(returns) > 1:
            # Estimate trading days per year based on data frequency
            time_span = (portfolio_values.index[-1] - portfolio_values.index[0]).days
            if time_span > 0:
                periods_per_year = 365 / time_span * len(returns)
                report.annualized_return = (1 + report.total_return) ** (365 / time_span) - 1
            else:
                report.annualized_return = report.total_return
        else:
            report.annualized_return = report.total_return
        
        # Annualized volatility
        if len(returns) > 1:
            time_span = (portfolio_values.index[-1] - portfolio_values.index[0]).days
            if time_span > 0:
                periods_per_year = 365 / time_span * len(returns)
                report.annualized_volatility = returns.std() * np.sqrt(periods_per_year)
            else:
                report.annualized_volatility = returns.std()
        else:
            report.annualized_volatility = 0.0
        
        # Sharpe ratio
        if report.annualized_volatility > 0:
            excess_return = report.annualized_return - self.risk_free_rate
            report.sharpe_ratio = excess_return / report.annualized_volatility
        else:
            report.sharpe_ratio = 0.0
    
    def _calculate_drawdown_metrics(self, report: PerformanceReport, 
                                  portfolio_values: pd.Series):
        """Calculate drawdown analysis metrics."""
        if len(portfolio_values) < 2:
            return
        
        # Calculate running maximum
        running_max = portfolio_values.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_values - running_max) / running_max
        
        # Maximum drawdown
        report.max_drawdown = drawdown.min()
        
        # Current drawdown
        report.current_drawdown = drawdown.iloc[-1]
        
        # Drawdown periods
        report.drawdown_periods = (drawdown < 0).sum()
        
        # Maximum drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        report.max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_risk_metrics(self, report: PerformanceReport, returns: pd.Series):
        """Calculate risk metrics."""
        if len(returns) == 0:
            return
        
        # Value at Risk (VaR)
        report.var_95 = np.percentile(returns, 5)  # 5th percentile
        report.var_99 = np.percentile(returns, 1)  # 1st percentile
        
        # Conditional Value at Risk (CVaR)
        report.cvar_95 = returns[returns <= report.var_95].mean()
        
        # Downside and upside deviation
        negative_returns = returns[returns < 0]
        positive_returns = returns[returns > 0]
        
        report.downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0.0
        report.upside_deviation = positive_returns.std() if len(positive_returns) > 0 else 0.0
        
        # Sortino ratio
        if report.downside_deviation > 0:
            excess_return = report.annualized_return - self.risk_free_rate
            report.sortino_ratio = excess_return / (report.downside_deviation * np.sqrt(252))
        else:
            report.sortino_ratio = 0.0
    
    def _calculate_trade_metrics(self, report: PerformanceReport, trades: List[Dict[str, Any]]):
        """Calculate trade statistics."""
        if not trades:
            return
        
        # Convert trades to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades)
        
        # Basic trade counts
        report.total_trades = len(trades_df)
        report.winning_trades = (trades_df['pnl'] > 0).sum()
        report.losing_trades = (trades_df['pnl'] < 0).sum()
        
        # Win rate
        if report.total_trades > 0:
            report.win_rate = report.winning_trades / report.total_trades
        
        # Profit and loss analysis
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        if len(winning_trades) > 0:
            report.average_win = winning_trades['pnl'].mean()
            report.largest_win = winning_trades['pnl'].max()
        else:
            report.average_win = 0.0
            report.largest_win = 0.0
        
        if len(losing_trades) > 0:
            report.average_loss = losing_trades['pnl'].mean()
            report.largest_loss = losing_trades['pnl'].min()
        else:
            report.average_loss = 0.0
            report.largest_loss = 0.0
        
        # Profit factor
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0.0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.0
        
        if total_loss > 0:
            report.profit_factor = total_profit / total_loss
        else:
            report.profit_factor = float('inf') if total_profit > 0 else 0.0
        
        # Average trade
        report.average_trade = trades_df['pnl'].mean()
        
        # Trade duration analysis
        if 'duration_hours' in trades_df.columns:
            if len(winning_trades) > 0:
                report.average_win_duration = winning_trades['duration_hours'].mean()
            if len(losing_trades) > 0:
                report.average_loss_duration = losing_trades['duration_hours'].mean()
            report.average_trade_duration = trades_df['duration_hours'].mean()
        
        # Consecutive trades analysis
        self._calculate_consecutive_trades(report, trades_df)
    
    def _calculate_consecutive_trades(self, report: PerformanceReport, trades_df: pd.DataFrame):
        """Calculate consecutive wins/losses statistics."""
        if len(trades_df) == 0:
            return
        
        # Create win/loss sequence
        is_win = trades_df['pnl'] > 0
        win_sequence = is_win.astype(int)
        
        # Find consecutive sequences
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        current_streak_type = "none"
        
        if len(win_sequence) > 0:
            # Calculate max consecutive wins
            consecutive_wins = 0
            consecutive_losses = 0
            
            for i, is_win_trade in enumerate(win_sequence):
                if is_win_trade == 1:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            # Current streak
            if len(win_sequence) > 0:
                last_trade_win = win_sequence.iloc[-1] == 1
                current_streak = 1
                
                # Count backwards to find current streak
                for i in range(len(win_sequence) - 2, -1, -1):
                    if win_sequence.iloc[i] == win_sequence.iloc[-1]:
                        current_streak += 1
                    else:
                        break
                
                current_streak_type = "win" if last_trade_win else "loss"
        
        report.max_consecutive_wins = max_consecutive_wins
        report.max_consecutive_losses = max_consecutive_losses
        report.current_streak = current_streak
        report.current_streak_type = current_streak_type
    
    def _calculate_benchmark_metrics(self, report: PerformanceReport, 
                                   returns: pd.Series, 
                                   benchmark_values: pd.Series):
        """Calculate benchmark comparison metrics."""
        try:
            # Align benchmark data with portfolio data
            common_index = returns.index.intersection(benchmark_values.index)
            if len(common_index) < 2:
                return
            
            portfolio_returns = returns.loc[common_index]
            benchmark_returns = benchmark_values.pct_change().loc[common_index].dropna()
            
            # Align the series
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) < 2:
                return
            
            portfolio_returns = portfolio_returns.loc[common_dates]
            benchmark_returns = benchmark_returns.loc[common_dates]
            
            # Benchmark return
            report.benchmark_return = (benchmark_values.iloc[-1] / benchmark_values.iloc[0]) - 1
            
            # Beta calculation
            if len(portfolio_returns) > 1 and benchmark_returns.std() > 0:
                covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
                benchmark_variance = benchmark_returns.var()
                report.beta = covariance / benchmark_variance
            else:
                report.beta = 1.0
            
            # Alpha calculation
            if report.beta != 0:
                report.alpha = report.annualized_return - (self.risk_free_rate + report.beta * (report.benchmark_return - self.risk_free_rate))
            else:
                report.alpha = report.annualized_return - report.benchmark_return
            
            # Tracking error
            excess_returns = portfolio_returns - benchmark_returns
            report.tracking_error = excess_returns.std() * np.sqrt(252)  # Annualized
            
            # Information ratio
            if report.tracking_error > 0:
                report.information_ratio = (report.annualized_return - report.benchmark_return) / report.tracking_error
            else:
                report.information_ratio = 0.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating benchmark metrics: {e}")
    
    def _calculate_additional_metrics(self, report: PerformanceReport, 
                                    returns: pd.Series, 
                                    portfolio_values: pd.Series):
        """Calculate additional performance metrics."""
        if len(returns) == 0:
            return
        
        # Calmar ratio
        if abs(report.max_drawdown) > 0:
            report.calmar_ratio = report.annualized_return / abs(report.max_drawdown)
        else:
            report.calmar_ratio = 0.0
        
        # Recovery factor
        if abs(report.max_drawdown) > 0:
            report.recovery_factor = report.total_return / abs(report.max_drawdown)
        else:
            report.recovery_factor = 0.0
        
        # Sterling ratio (using average drawdown)
        if len(portfolio_values) > 1:
            running_max = portfolio_values.expanding().max()
            drawdowns = (portfolio_values - running_max) / running_max
            avg_drawdown = abs(drawdowns[drawdowns < 0].mean()) if (drawdowns < 0).any() else 0.0
            
            if avg_drawdown > 0:
                report.sterling_ratio = report.annualized_return / avg_drawdown
            else:
                report.sterling_ratio = 0.0
        else:
            report.sterling_ratio = 0.0
        
        # Burke ratio
        if len(portfolio_values) > 1:
            running_max = portfolio_values.expanding().max()
            drawdowns = (portfolio_values - running_max) / running_max
            negative_drawdowns = drawdowns[drawdowns < 0]
            
            if len(negative_drawdowns) > 0:
                sum_squared_drawdowns = (negative_drawdowns ** 2).sum()
                if sum_squared_drawdowns > 0:
                    report.burke_ratio = report.annualized_return / np.sqrt(sum_squared_drawdowns)
                else:
                    report.burke_ratio = 0.0
            else:
                report.burke_ratio = 0.0
        else:
            report.burke_ratio = 0.0
        
        # Kappa Three (third moment)
        if len(returns) > 2 and returns.std() > 0:
            third_moment = ((returns - returns.mean()) ** 3).mean()
            report.kappa_three = third_moment / (returns.std() ** 3)
        else:
            report.kappa_three = 0.0
    
    def compare_to_benchmark(self, 
                           portfolio_values: pd.Series, 
                           benchmark_values: pd.Series) -> Dict[str, float]:
        """
        Compare portfolio performance to benchmark.
        
        Args:
            portfolio_values: Portfolio values over time
            benchmark_values: Benchmark values over time
            
        Returns:
            Dict with comparison metrics
        """
        try:
            # Calculate returns
            portfolio_returns = self._calculate_returns(portfolio_values)
            benchmark_returns = self._calculate_returns(benchmark_values)
            
            # Align data
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) < 2:
                return {}
            
            portfolio_returns = portfolio_returns.loc[common_index]
            benchmark_returns = benchmark_returns.loc[common_index]
            
            # Calculate metrics
            portfolio_total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            benchmark_total_return = (benchmark_values.iloc[-1] / benchmark_values.iloc[0]) - 1
            
            # Excess return
            excess_return = portfolio_total_return - benchmark_total_return
            
            # Beta
            if len(portfolio_returns) > 1 and benchmark_returns.std() > 0:
                covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
                beta = covariance / benchmark_returns.var()
            else:
                beta = 1.0
            
            # Alpha
            alpha = portfolio_total_return - (self.risk_free_rate + beta * (benchmark_total_return - self.risk_free_rate))
            
            # Tracking error
            excess_returns = portfolio_returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)  # Annualized
            
            # Information ratio
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0.0
            
            return {
                'portfolio_return': portfolio_total_return,
                'benchmark_return': benchmark_total_return,
                'excess_return': excess_return,
                'alpha': alpha,
                'beta': beta,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'outperformed': excess_return > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing to benchmark: {e}")
            return {}
    
    def calculate_rolling_metrics(self, 
                                portfolio_values: pd.Series,
                                window: int = 30) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            portfolio_values: Portfolio values over time
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        try:
            if len(portfolio_values) < window:
                return pd.DataFrame()
            
            returns = self._calculate_returns(portfolio_values)
            
            # Calculate rolling metrics
            rolling_returns = returns.rolling(window=window)
            rolling_volatility = rolling_returns.std() * np.sqrt(252)  # Annualized
            
            # Rolling Sharpe ratio
            rolling_sharpe = (rolling_returns.mean() * 252 - self.risk_free_rate) / rolling_volatility
            
            # Rolling drawdown
            rolling_max = portfolio_values.rolling(window=window).max()
            rolling_drawdown = (portfolio_values - rolling_max) / rolling_max
            
            # Create DataFrame
            rolling_metrics = pd.DataFrame({
                'returns': returns,
                'volatility': rolling_volatility,
                'sharpe_ratio': rolling_sharpe,
                'drawdown': rolling_drawdown
            })
            
            return rolling_metrics.dropna()
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling metrics: {e}")
            return pd.DataFrame()
    
    def generate_performance_summary(self, report: PerformanceReport) -> str:
        """
        Generate a text summary of performance metrics.
        
        Args:
            report: Performance report
            
        Returns:
            Formatted text summary
        """
        summary = f"""
PERFORMANCE SUMMARY
==================

Portfolio Performance:
  Total Return: {report.total_return:.2%}
  Annualized Return: {report.annualized_return:.2%}
  Annualized Volatility: {report.annualized_volatility:.2%}
  Sharpe Ratio: {report.sharpe_ratio:.2f}
  Sortino Ratio: {report.sortino_ratio:.2f}
  Calmar Ratio: {report.calmar_ratio:.2f}

Risk Analysis:
  Maximum Drawdown: {report.max_drawdown:.2%}
  VaR (95%): {report.var_95:.2%}
  VaR (99%): {report.var_99:.2%}
  Downside Deviation: {report.downside_deviation:.2%}

Trade Statistics:
  Total Trades: {report.total_trades}
  Win Rate: {report.win_rate:.2%}
  Profit Factor: {report.profit_factor:.2f}
  Average Win: ${report.average_win:.2f}
  Average Loss: ${report.average_loss:.2f}
  Largest Win: ${report.largest_win:.2f}
  Largest Loss: ${report.largest_loss:.2f}

Consecutive Trades:
  Max Consecutive Wins: {report.max_consecutive_wins}
  Max Consecutive Losses: {report.max_consecutive_losses}
  Current Streak: {report.current_streak} ({report.current_streak_type})
"""
        
        if report.benchmark_return != 0:
            summary += f"""
Benchmark Comparison:
  Benchmark Return: {report.benchmark_return:.2%}
  Alpha: {report.alpha:.2%}
  Beta: {report.beta:.2f}
  Information Ratio: {report.information_ratio:.2f}
"""
        
        return summary
