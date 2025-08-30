"""
Technical Indicators Implementation
==================================

High-performance technical analysis indicators with comprehensive error handling.
All indicators accept pandas DataFrame/Series as input and return calculated values.

Author: dat-ns
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..utils.exceptions import ValidationError, DataError


@dataclass
class IndicatorResult:
    """Container for indicator calculation results."""
    values: pd.Series
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate result after initialization."""
        if not isinstance(self.values, pd.Series):
            raise ValidationError("Indicator values must be a pandas Series")


class TechnicalIndicators:
    """
    Technical Analysis Indicators Implementation.
    
    This class provides static methods for calculating various technical indicators
    commonly used in trading strategies. All methods are optimized for performance
    and include comprehensive error handling.
    
    Features:
    - Vectorized calculations using numpy/pandas
    - Comprehensive input validation
    - Detailed error handling and logging
    - Memory-efficient operations
    - Support for various data types
    """
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def _validate_data(data: Union[pd.Series, pd.DataFrame, np.ndarray], 
                      min_length: int = 1, column: str = None) -> pd.Series:
        """
        Validate and prepare input data for indicator calculation.
        
        Args:
            data: Input price data (Series, DataFrame, or array)
            min_length: Minimum required data points
            column: Column name if DataFrame is provided
            
        Returns:
            pd.Series: Validated price series
            
        Raises:
            ValidationError: If data validation fails
            DataError: If data is insufficient or invalid
        """
        try:
            # Handle different input types
            if isinstance(data, pd.DataFrame):
                if column:
                    if column not in data.columns:
                        raise ValidationError(f"Column '{column}' not found in DataFrame")
                    price_data = data[column].copy()
                else:
                    # Try common price column names
                    price_columns = ['close', 'Close', 'price', 'Price']
                    found_column = None
                    for col in price_columns:
                        if col in data.columns:
                            found_column = col
                            break
                    
                    if found_column:
                        price_data = data[found_column].copy()
                    else:
                        raise ValidationError(
                            "No price column found. Available columns: " + 
                            ", ".join(data.columns.tolist())
                        )
            
            elif isinstance(data, pd.Series):
                price_data = data.copy()
            
            elif isinstance(data, (list, np.ndarray)):
                price_data = pd.Series(data)
            
            else:
                raise ValidationError(f"Unsupported data type: {type(data)}")
            
            # Validate data length
            if len(price_data) < min_length:
                raise DataError(
                    f"Insufficient data points. Required: {min_length}, Got: {len(price_data)}"
                )
            
            # Check for valid numeric data
            if not pd.api.types.is_numeric_dtype(price_data):
                raise ValidationError("Data must be numeric")
            
            # Handle missing values
            if price_data.isnull().any():
                logging.warning(f"Found {price_data.isnull().sum()} null values in data")
                # Forward fill missing values
                price_data = price_data.fillna(method='ffill').fillna(method='bfill')
            
            # Check for infinite values
            if np.isinf(price_data).any():
                raise ValidationError("Data contains infinite values")
            
            # Check for negative prices (assuming price data)
            if (price_data <= 0).any():
                logging.warning("Found non-positive values in price data")
            
            return price_data
            
        except Exception as e:
            if isinstance(e, (ValidationError, DataError)):
                raise
            raise DataError(f"Data validation failed: {str(e)}")
    
    @staticmethod
    def sma(data: Union[pd.Series, pd.DataFrame, np.ndarray], 
            period: int = 20, column: str = None) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        SMA = (Sum of closing prices over n periods) / n
        
        Args:
            data: Price data (Series, DataFrame, or array)
            period: Number of periods for moving average (default: 20)
            column: Column name if DataFrame provided
            
        Returns:
            pd.Series: SMA values with original index
            
        Raises:
            ValidationError: If parameters are invalid
            DataError: If data is insufficient
            
        Example:
            >>> sma_20 = TechnicalIndicators.sma(df['close'], period=20)
            >>> sma_50 = TechnicalIndicators.sma(df, period=50, column='close')
        """
        try:
            # Validate inputs
            if period <= 0:
                raise ValidationError("Period must be positive")
            
            if period > 200:
                logging.warning(f"Large SMA period ({period}) may reduce data significantly")
            
            # Validate and prepare data
            price_series = TechnicalIndicators._validate_data(data, period, column)
            
            # Calculate SMA using rolling window
            sma_values = price_series.rolling(window=period, min_periods=period).mean()
            
            # Preserve original index
            sma_values.name = f'SMA_{period}'
            
            logging.debug(f"Calculated SMA({period}) for {len(price_series)} data points")
            
            return sma_values
            
        except Exception as e:
            if isinstance(e, (ValidationError, DataError)):
                raise
            raise DataError(f"SMA calculation failed: {str(e)}")
    
    @staticmethod
    def ema(data: Union[pd.Series, pd.DataFrame, np.ndarray], 
            period: int = 20, column: str = None, alpha: Optional[float] = None) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        EMA = (Current Price * Alpha) + (Previous EMA * (1 - Alpha))
        Where Alpha = 2 / (Period + 1)
        
        Args:
            data: Price data (Series, DataFrame, or array)
            period: Number of periods for EMA (default: 20)
            column: Column name if DataFrame provided
            alpha: Smoothing factor (optional, calculated from period if not provided)
            
        Returns:
            pd.Series: EMA values with original index
            
        Raises:
            ValidationError: If parameters are invalid
            DataError: If data is insufficient
            
        Example:
            >>> ema_12 = TechnicalIndicators.ema(df['close'], period=12)
            >>> ema_custom = TechnicalIndicators.ema(df['close'], period=26, alpha=0.1)
        """
        try:
            # Validate inputs
            if period <= 0:
                raise ValidationError("Period must be positive")
            
            if alpha is not None:
                if not (0 < alpha <= 1):
                    raise ValidationError("Alpha must be between 0 and 1")
            else:
                alpha = 2.0 / (period + 1)
            
            # Validate and prepare data
            price_series = TechnicalIndicators._validate_data(data, period, column)
            
            # Calculate EMA using pandas ewm (exponentially weighted moving average)
            ema_values = price_series.ewm(alpha=alpha, adjust=False).mean()
            
            # Preserve original index
            ema_values.name = f'EMA_{period}'
            
            logging.debug(f"Calculated EMA({period}, α={alpha:.4f}) for {len(price_series)} data points")
            
            return ema_values
            
        except Exception as e:
            if isinstance(e, (ValidationError, DataError)):
                raise
            raise DataError(f"EMA calculation failed: {str(e)}")
    
    @staticmethod
    def rsi(data: Union[pd.Series, pd.DataFrame, np.ndarray], 
            period: int = 14, column: str = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        Where RS = Average Gain / Average Loss over the period
        
        Args:
            data: Price data (Series, DataFrame, or array)
            period: Number of periods for RSI calculation (default: 14)
            column: Column name if DataFrame provided
            
        Returns:
            pd.Series: RSI values (0-100) with original index
            
        Raises:
            ValidationError: If parameters are invalid
            DataError: If data is insufficient
            
        Example:
            >>> rsi_14 = TechnicalIndicators.rsi(df['close'], period=14)
            >>> rsi_21 = TechnicalIndicators.rsi(df, period=21, column='close')
        """
        try:
            # Validate inputs
            if period <= 0:
                raise ValidationError("Period must be positive")
            
            if period < 2:
                raise ValidationError("RSI period should be at least 2")
            
            # Validate and prepare data (need extra data point for price changes)
            price_series = TechnicalIndicators._validate_data(data, period + 1, column)
            
            # Calculate price changes
            price_changes = price_series.diff()
            
            # Separate gains and losses
            gains = price_changes.where(price_changes > 0, 0.0)
            losses = -price_changes.where(price_changes < 0, 0.0)
            
            # Calculate average gains and losses using EMA
            avg_gains = gains.ewm(alpha=1.0/period, adjust=False).mean()
            avg_losses = losses.ewm(alpha=1.0/period, adjust=False).mean()
            
            # Calculate RSI
            # Handle division by zero case
            rs = np.where(avg_losses != 0, avg_gains / avg_losses, np.inf)
            rsi_values = 100 - (100 / (1 + rs))
            
            # Convert back to Series with original index
            rsi_series = pd.Series(rsi_values, index=price_series.index, name=f'RSI_{period}')
            
            # Handle edge cases
            rsi_series = rsi_series.fillna(50)  # Neutral RSI for NaN values
            rsi_series = rsi_series.clip(0, 100)  # Ensure RSI stays in 0-100 range
            
            logging.debug(f"Calculated RSI({period}) for {len(price_series)} data points")
            
            return rsi_series
            
        except Exception as e:
            if isinstance(e, (ValidationError, DataError)):
                raise
            raise DataError(f"RSI calculation failed: {str(e)}")
    
    @staticmethod
    def macd(data: Union[pd.Series, pd.DataFrame, np.ndarray], 
             fast: int = 12, slow: int = 26, signal: int = 9, 
             column: str = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal_period)
        Histogram = MACD Line - Signal Line
        
        Args:
            data: Price data (Series, DataFrame, or array)
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            column: Column name if DataFrame provided
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (MACD line, Signal line, Histogram)
            
        Raises:
            ValidationError: If parameters are invalid
            DataError: If data is insufficient
            
        Example:
            >>> macd_line, signal_line, histogram = TechnicalIndicators.macd(df['close'])
            >>> macd_line, signal_line, histogram = TechnicalIndicators.macd(
            ...     df, fast=8, slow=21, signal=5, column='close'
            ... )
        """
        try:
            # Validate inputs
            if fast <= 0 or slow <= 0 or signal <= 0:
                raise ValidationError("All periods must be positive")
            
            if fast >= slow:
                raise ValidationError("Fast period must be less than slow period")
            
            # Need enough data for the slow EMA plus signal calculation
            min_data = slow + signal
            
            # Validate and prepare data
            price_series = TechnicalIndicators._validate_data(data, min_data, column)
            
            # Calculate fast and slow EMAs
            fast_ema = TechnicalIndicators.ema(price_series, period=fast)
            slow_ema = TechnicalIndicators.ema(price_series, period=slow)
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            macd_line.name = f'MACD_{fast}_{slow}'
            
            # Calculate signal line (EMA of MACD line)
            signal_line = TechnicalIndicators.ema(macd_line, period=signal)
            signal_line.name = f'MACD_Signal_{signal}'
            
            # Calculate histogram
            histogram = macd_line - signal_line
            histogram.name = f'MACD_Histogram_{fast}_{slow}_{signal}'
            
            logging.debug(
                f"Calculated MACD({fast}, {slow}, {signal}) for {len(price_series)} data points"
            )
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            if isinstance(e, (ValidationError, DataError)):
                raise
            raise DataError(f"MACD calculation failed: {str(e)}")
    
    @staticmethod
    def bollinger_bands(data: Union[pd.Series, pd.DataFrame, np.ndarray], 
                       period: int = 20, std_dev: float = 2.0, 
                       column: str = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Middle Band = SMA(period)
        Upper Band = Middle Band + (Standard Deviation * std_dev)
        Lower Band = Middle Band - (Standard Deviation * std_dev)
        
        Args:
            data: Price data (Series, DataFrame, or array)
            period: Number of periods for SMA and standard deviation (default: 20)
            std_dev: Number of standard deviations (default: 2.0)
            column: Column name if DataFrame provided
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (Upper band, Middle band, Lower band)
            
        Raises:
            ValidationError: If parameters are invalid
            DataError: If data is insufficient
            
        Example:
            >>> upper, middle, lower = TechnicalIndicators.bollinger_bands(df['close'])
            >>> upper, middle, lower = TechnicalIndicators.bollinger_bands(
            ...     df, period=10, std_dev=1.5, column='close'
            ... )
        """
        try:
            # Validate inputs
            if period <= 0:
                raise ValidationError("Period must be positive")
            
            if std_dev <= 0:
                raise ValidationError("Standard deviation multiplier must be positive")
            
            if period < 2:
                raise ValidationError("Period should be at least 2 for meaningful standard deviation")
            
            # Validate and prepare data
            price_series = TechnicalIndicators._validate_data(data, period, column)
            
            # Calculate middle band (SMA)
            middle_band = TechnicalIndicators.sma(price_series, period)
            middle_band.name = f'BB_Middle_{period}'
            
            # Calculate rolling standard deviation
            rolling_std = price_series.rolling(window=period, min_periods=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (rolling_std * std_dev)
            upper_band.name = f'BB_Upper_{period}_{std_dev}'
            
            lower_band = middle_band - (rolling_std * std_dev)
            lower_band.name = f'BB_Lower_{period}_{std_dev}'
            
            logging.debug(
                f"Calculated Bollinger Bands({period}, {std_dev}) for {len(price_series)} data points"
            )
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            if isinstance(e, (ValidationError, DataError)):
                raise
            raise DataError(f"Bollinger Bands calculation failed: {str(e)}")
    
    @staticmethod
    def stochastic_oscillator(data: Union[pd.DataFrame, np.ndarray], 
                             k_period: int = 14, d_period: int = 3,
                             high_column: str = 'high', low_column: str = 'low',
                             close_column: str = 'close') -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D).
        
        %K = ((Current Close - Lowest Low) / (Highest High - Lowest Low)) * 100
        %D = SMA(%K, d_period)
        
        Args:
            data: OHLC data (DataFrame)
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D smoothing (default: 3)
            high_column: High price column name
            low_column: Low price column name
            close_column: Close price column name
            
        Returns:
            Tuple[pd.Series, pd.Series]: (%K, %D)
            
        Raises:
            ValidationError: If parameters are invalid
            DataError: If data is insufficient or columns missing
        """
        try:
            # Validate inputs
            if k_period <= 0 or d_period <= 0:
                raise ValidationError("Periods must be positive")
            
            # Validate DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValidationError("Stochastic oscillator requires OHLC DataFrame")
            
            required_columns = [high_column, low_column, close_column]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValidationError(f"Missing columns: {missing_columns}")
            
            if len(data) < k_period + d_period:
                raise DataError(f"Insufficient data. Required: {k_period + d_period}, Got: {len(data)}")
            
            # Get price series
            high_series = data[high_column]
            low_series = data[low_column]
            close_series = data[close_column]
            
            # Calculate rolling high and low
            highest_high = high_series.rolling(window=k_period).max()
            lowest_low = low_series.rolling(window=k_period).min()
            
            # Calculate %K
            k_percent = ((close_series - lowest_low) / (highest_high - lowest_low)) * 100
            k_percent = k_percent.fillna(50)  # Neutral value for NaN
            k_percent = k_percent.clip(0, 100)
            k_percent.name = f'Stoch_K_{k_period}'
            
            # Calculate %D (SMA of %K)
            d_percent = k_percent.rolling(window=d_period).mean()
            d_percent.name = f'Stoch_D_{d_period}'
            
            logging.debug(
                f"Calculated Stochastic({k_period}, {d_period}) for {len(data)} data points"
            )
            
            return k_percent, d_percent
            
        except Exception as e:
            if isinstance(e, (ValidationError, DataError)):
                raise
            raise DataError(f"Stochastic oscillator calculation failed: {str(e)}")
    
    @staticmethod
    def williams_r(data: Union[pd.DataFrame, np.ndarray], 
                   period: int = 14, high_column: str = 'high',
                   low_column: str = 'low', close_column: str = 'close') -> pd.Series:
        """
        Calculate Williams %R oscillator.
        
        %R = ((Highest High - Current Close) / (Highest High - Lowest Low)) * -100
        
        Args:
            data: OHLC data (DataFrame)
            period: Period for calculation (default: 14)
            high_column: High price column name
            low_column: Low price column name
            close_column: Close price column name
            
        Returns:
            pd.Series: Williams %R values (-100 to 0)
            
        Raises:
            ValidationError: If parameters are invalid
            DataError: If data is insufficient or columns missing
        """
        try:
            # Validate inputs
            if period <= 0:
                raise ValidationError("Period must be positive")
            
            # Validate DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValidationError("Williams %R requires OHLC DataFrame")
            
            required_columns = [high_column, low_column, close_column]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValidationError(f"Missing columns: {missing_columns}")
            
            if len(data) < period:
                raise DataError(f"Insufficient data. Required: {period}, Got: {len(data)}")
            
            # Get price series
            high_series = data[high_column]
            low_series = data[low_column]
            close_series = data[close_column]
            
            # Calculate rolling high and low
            highest_high = high_series.rolling(window=period).max()
            lowest_low = low_series.rolling(window=period).min()
            
            # Calculate Williams %R
            williams_r = ((highest_high - close_series) / (highest_high - lowest_low)) * -100
            williams_r = williams_r.fillna(-50)  # Neutral value for NaN
            williams_r = williams_r.clip(-100, 0)
            williams_r.name = f'Williams_R_{period}'
            
            logging.debug(f"Calculated Williams %R({period}) for {len(data)} data points")
            
            return williams_r
            
        except Exception as e:
            if isinstance(e, (ValidationError, DataError)):
                raise
            raise DataError(f"Williams %R calculation failed: {str(e)}")
    
    @classmethod
    def calculate_multiple_indicators(cls, data: Union[pd.Series, pd.DataFrame], 
                                    indicators: Dict[str, Dict[str, Any]],
                                    column: str = None) -> pd.DataFrame:
        """
        Calculate multiple indicators at once for efficiency.
        
        Args:
            data: Price data
            indicators: Dictionary of indicator configurations
                       {indicator_name: {param1: value1, param2: value2, ...}}
            column: Column name if DataFrame provided
            
        Returns:
            pd.DataFrame: DataFrame with all calculated indicators
            
        Example:
            >>> indicators_config = {
            ...     'sma_20': {'indicator': 'sma', 'period': 20},
            ...     'ema_12': {'indicator': 'ema', 'period': 12},
            ...     'rsi_14': {'indicator': 'rsi', 'period': 14},
            ... }
            >>> results = TechnicalIndicators.calculate_multiple_indicators(
            ...     df['close'], indicators_config
            ... )
        """
        try:
            results = pd.DataFrame(index=data.index if hasattr(data, 'index') else None)
            
            for name, config in indicators.items():
                indicator_type = config.pop('indicator')
                
                if indicator_type == 'sma':
                    results[name] = cls.sma(data, column=column, **config)
                elif indicator_type == 'ema':
                    results[name] = cls.ema(data, column=column, **config)
                elif indicator_type == 'rsi':
                    results[name] = cls.rsi(data, column=column, **config)
                elif indicator_type == 'macd':
                    macd_line, signal_line, histogram = cls.macd(data, column=column, **config)
                    results[f'{name}_line'] = macd_line
                    results[f'{name}_signal'] = signal_line
                    results[f'{name}_histogram'] = histogram
                elif indicator_type == 'bollinger_bands':
                    upper, middle, lower = cls.bollinger_bands(data, column=column, **config)
                    results[f'{name}_upper'] = upper
                    results[f'{name}_middle'] = middle
                    results[f'{name}_lower'] = lower
                else:
                    logging.warning(f"Unknown indicator type: {indicator_type}")
            
            logging.info(f"Calculated {len(indicators)} indicators successfully")
            return results
            
        except Exception as e:
            raise DataError(f"Multiple indicators calculation failed: {str(e)}")


# Utility functions for common indicator combinations
def trend_following_indicators(data: Union[pd.Series, pd.DataFrame], 
                             column: str = None) -> pd.DataFrame:
    """Calculate common trend-following indicators."""
    indicators = TechnicalIndicators()
    
    results = pd.DataFrame(index=data.index if hasattr(data, 'index') else None)
    results['SMA_20'] = indicators.sma(data, 20, column)
    results['SMA_50'] = indicators.sma(data, 50, column)
    results['EMA_12'] = indicators.ema(data, 12, column)
    results['EMA_26'] = indicators.ema(data, 26, column)
    
    macd_line, signal_line, histogram = indicators.macd(data, column=column)
    results['MACD_Line'] = macd_line
    results['MACD_Signal'] = signal_line
    results['MACD_Histogram'] = histogram
    
    return results


def momentum_indicators(data: Union[pd.Series, pd.DataFrame], 
                       column: str = None) -> pd.DataFrame:
    """Calculate common momentum indicators."""
    indicators = TechnicalIndicators()
    
    results = pd.DataFrame(index=data.index if hasattr(data, 'index') else None)
    results['RSI_14'] = indicators.rsi(data, 14, column)
    
    if isinstance(data, pd.DataFrame) and all(col in data.columns for col in ['high', 'low', 'close']):
        stoch_k, stoch_d = indicators.stochastic_oscillator(data)
        results['Stoch_K'] = stoch_k
        results['Stoch_D'] = stoch_d
        results['Williams_R'] = indicators.williams_r(data)
    
    return results


def volatility_indicators(data: Union[pd.Series, pd.DataFrame], 
                         column: str = None) -> pd.DataFrame:
    """Calculate common volatility indicators."""
    indicators = TechnicalIndicators()
    
    results = pd.DataFrame(index=data.index if hasattr(data, 'index') else None)
    
    upper, middle, lower = indicators.bollinger_bands(data, column=column)
    results['BB_Upper'] = upper
    results['BB_Middle'] = middle
    results['BB_Lower'] = lower
    
    # Calculate Bollinger Band width and %B
    results['BB_Width'] = (upper - lower) / middle * 100
    results['BB_PercentB'] = (data - lower) / (upper - lower) if column is None else (data[column] - lower) / (upper - lower)
    
    return results