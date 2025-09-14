"""
Data Loader for Backtesting
============================

Handles loading historical data for backtesting from various sources:
- Built-in sample data
- CSV files
- Mock/synthetic data generation

No API keys or live connections required.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Union
import logging


class DataLoader:
    """
    Data loader for backtesting with multiple data source support.
    
    Features:
    - Load sample BTCUSDT historical data
    - Load custom CSV data
    - Generate synthetic price data
    - Data validation and cleaning
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.logger = logging.getLogger(__name__)
        
    def load_sample_data(self, symbol: str = "BTCUSDT", 
                        start_date: str = "2023-01-01",
                        end_date: str = "2023-12-31") -> pd.DataFrame:
        """
        Load built-in sample historical data for immediate testing.
        
        Args:
            symbol: Trading symbol (default: BTCUSDT)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index
        """
        try:
            self.logger.info(f"Loading sample data for {symbol} from {start_date} to {end_date}")
            
            # Generate sample BTCUSDT data based on realistic price movements
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Create hourly data
            dates = pd.date_range(start_dt, end_dt, freq='1H')
            
            # Generate realistic Bitcoin price data
            np.random.seed(42)  # For reproducible results
            
            # Starting price around $16,000 (early 2023)
            initial_price = 16000.0
            
            # Generate price movements with trend and volatility
            n_periods = len(dates)
            
            # Create trending price with noise
            trend = np.linspace(0, 0.8, n_periods)  # 80% growth over the period
            volatility = 0.02  # 2% hourly volatility
            
            # Generate random walk with trend
            returns = np.random.normal(trend/n_periods, volatility, n_periods)
            prices = [initial_price]
            
            for i in range(1, n_periods):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(max(new_price, 1.0))  # Prevent negative prices
            
            # Generate OHLC data from price series
            data = []
            for i, (date, close_price) in enumerate(zip(dates, prices)):
                # Generate realistic OHLC from close price
                volatility_factor = np.random.uniform(0.5, 2.0)
                spread = close_price * 0.001 * volatility_factor  # 0.1% base spread
                
                high = close_price + np.random.uniform(0, spread)
                low = close_price - np.random.uniform(0, spread)
                
                # Ensure logical OHLC relationships
                if i > 0:
                    open_price = prices[i-1] + np.random.uniform(-spread/2, spread/2)
                else:
                    open_price = close_price
                
                # Ensure high >= max(open, close) and low <= min(open, close)
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
                
                # Generate realistic volume (higher during volatile periods)
                base_volume = 1000000  # 1M base volume
                volatility_multiplier = abs(returns[i]) * 50 + 1
                volume = int(base_volume * volatility_multiplier * np.random.uniform(0.5, 2.0))
                
                data.append({
                    'datetime': date,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('datetime', inplace=True)
            
            self.logger.info(f"Generated {len(df)} sample data points")
            self.logger.info(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return self.validate_data(df)
            
        except Exception as e:
            self.logger.error(f"Error loading sample data: {e}")
            raise
    
    def load_csv_data(self, file_path: Union[str, Path],
                     datetime_col: str = 'datetime',
                     parse_dates: bool = True) -> pd.DataFrame:
        """
        Load historical data from CSV file.
        
        Expected CSV format:
        datetime,open,high,low,close,volume
        2023-01-01 00:00:00,16000.0,16100.0,15900.0,16050.0,1000000
        
        Args:
            file_path: Path to CSV file
            datetime_col: Name of datetime column
            parse_dates: Whether to parse datetime column
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            self.logger.info(f"Loading data from {file_path}")
            
            # Read CSV with proper datetime parsing
            df = pd.read_csv(
                file_path,
                parse_dates=[datetime_col] if parse_dates else False
            )
            
            # Set datetime as index
            if datetime_col in df.columns:
                df.set_index(datetime_col, inplace=True)
            
            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            self.logger.info(f"Loaded {len(df)} data points from CSV")
            self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return self.validate_data(df)
            
        except Exception as e:
            self.logger.error(f"Error loading CSV data: {e}")
            raise
    
    def generate_mock_data(self, 
                          symbol: str = "BTCUSDT",
                          start_price: float = 20000.0,
                          num_periods: int = 1000,
                          freq: str = '1H',
                          trend: float = 0.0,
                          volatility: float = 0.02) -> pd.DataFrame:
        """
        Generate synthetic price data for testing strategies.
        
        Args:
            symbol: Trading symbol
            start_price: Starting price
            num_periods: Number of data points to generate
            freq: Frequency (1H, 1D, etc.)
            trend: Daily trend (0.01 = 1% daily growth)
            volatility: Volatility factor (0.02 = 2% std dev)
            
        Returns:
            pd.DataFrame: Generated OHLCV data
        """
        try:
            self.logger.info(f"Generating {num_periods} mock data points for {symbol}")
            
            # Create date range
            end_date = datetime.now()
            if freq == '1H':
                start_date = end_date - timedelta(hours=num_periods)
            elif freq == '1D':
                start_date = end_date - timedelta(days=num_periods)
            elif freq == '1m':
                start_date = end_date - timedelta(minutes=num_periods)
            else:
                start_date = end_date - timedelta(hours=num_periods)
            
            dates = pd.date_range(start_date, end_date, freq=freq)[:num_periods]
            
            # Generate price series with geometric Brownian motion
            np.random.seed(None)  # Use current time for randomness
            
            # Convert trend to per-period
            if freq == '1H':
                trend_per_period = trend / 24  # Daily trend to hourly
            elif freq == '1D':
                trend_per_period = trend
            else:
                trend_per_period = trend / 24
            
            # Generate returns
            returns = np.random.normal(trend_per_period, volatility, num_periods)
            
            # Generate price series
            prices = [start_price]
            for i in range(1, num_periods):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(max(new_price, 0.01))  # Prevent negative prices
            
            # Generate OHLCV data
            data = []
            for i, (date, close_price) in enumerate(zip(dates, prices)):
                # Generate OHLC with some randomness
                price_range = close_price * volatility * np.random.uniform(0.5, 1.5)
                
                if i > 0:
                    open_price = prices[i-1] + np.random.uniform(-price_range/4, price_range/4)
                else:
                    open_price = close_price
                
                high = max(open_price, close_price) + np.random.uniform(0, price_range/2)
                low = min(open_price, close_price) - np.random.uniform(0, price_range/2)
                
                # Ensure OHLC relationships
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
                
                # Generate volume
                volume = int(np.random.lognormal(13, 1))  # Log-normal distribution
                
                data.append({
                    'datetime': date,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2), 
                    'close': round(close_price, 2),
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('datetime', inplace=True)
            
            self.logger.info(f"Generated mock data: {len(df)} periods")
            self.logger.info(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return self.validate_data(df)
            
        except Exception as e:
            self.logger.error(f"Error generating mock data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            pd.DataFrame: Cleaned and validated data
        """
        try:
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Remove rows with NaN values
            initial_length = len(df)
            df = df.dropna()
            
            if len(df) < initial_length:
                self.logger.warning(f"Removed {initial_length - len(df)} rows with NaN values")
            
            # Validate OHLC relationships
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )
            
            if invalid_ohlc.any():
                self.logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
                # Fix invalid relationships
                df.loc[invalid_ohlc, 'high'] = df.loc[invalid_ohlc, ['open', 'close']].max(axis=1)
                df.loc[invalid_ohlc, 'low'] = df.loc[invalid_ohlc, ['open', 'close']].min(axis=1)
            
            # Ensure positive prices
            negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
            if negative_prices.any():
                self.logger.warning(f"Found {negative_prices.sum()} rows with non-positive prices")
                df = df[~negative_prices]
            
            # Ensure positive volume
            df.loc[df['volume'] <= 0, 'volume'] = 1
            
            # Sort by datetime index
            df = df.sort_index()
            
            # Remove duplicate timestamps
            duplicates = df.index.duplicated()
            if duplicates.any():
                self.logger.warning(f"Removed {duplicates.sum()} duplicate timestamps")
                df = df[~duplicates]
            
            if len(df) == 0:
                raise ValueError("No valid data remaining after cleaning")
            
            self.logger.info(f"Data validation complete: {len(df)} valid rows")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            raise
    
    def save_sample_csv(self, file_path: Union[str, Path], 
                       symbol: str = "BTCUSDT",
                       num_days: int = 365) -> None:
        """
        Save sample data to CSV file for future use.
        
        Args:
            file_path: Path where to save CSV
            symbol: Trading symbol
            num_days: Number of days of data
        """
        try:
            start_date = (datetime.now() - timedelta(days=num_days)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            data = self.load_sample_data(symbol, start_date, end_date)
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Reset index to include datetime as column
            data_to_save = data.reset_index()
            data_to_save.to_csv(file_path, index=False)
            
            self.logger.info(f"Sample data saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving sample CSV: {e}")
            raise