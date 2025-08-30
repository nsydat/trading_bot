"""
Data Processor Module
====================

Handles processing, cleaning, and validation of market data.
Provides utilities for data transformation, normalization, and feature engineering.

Features:
- OHLCV data cleaning and validation
- Missing data handling and interpolation
- Data normalization and scaling
- Technical indicator calculation preparation
- Data quality checks and outlier detection
- Time series resampling and aggregation
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from core.utils.exceptions import DataError, ValidationError


@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    symbol: str
    timeframe: str
    total_rows: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    outliers: Dict[str, int]
    data_gaps: List[Tuple[datetime, datetime]]
    quality_score: float
    issues: List[str]
    recommendations: List[str]


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""
    fill_missing_method: str = 'forward'  # 'forward', 'backward', 'interpolate', 'drop'
    outlier_detection_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 3.0
    max_gap_fill_minutes: int = 60  # Maximum gap to fill in minutes
    normalize_prices: bool = False
    remove_duplicates: bool = True
    validate_ohlc_logic: bool = True
    min_volume_threshold: float = 0.0


class DataProcessor:
    """
    Main data processor class for cleaning and transforming market data.
    
    Provides comprehensive data processing capabilities including:
    - Data validation and quality checks
    - Missing data handling
    - Outlier detection and removal
    - Data normalization and transformation
    - Technical analysis preparation
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the data processor.
        
        Args:
            config (Optional[ProcessingConfig]): Processing configuration
        """
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'processed_datasets': 0,
            'total_rows_processed': 0,
            'missing_values_filled': 0,
            'outliers_removed': 0,
            'duplicates_removed': 0,
            'avg_quality_score': 0.0
        }
        
        self.logger.info("🔧 Data Processor initialized")
    
    async def process_ohlcv(self, df: pd.DataFrame, symbol: str = "", 
                          timeframe: str = "") -> pd.DataFrame:
        """
        Process OHLCV DataFrame with comprehensive cleaning and validation.
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            symbol (str): Trading symbol for logging
            timeframe (str): Timeframe for logging
            
        Returns:
            pd.DataFrame: Processed and cleaned OHLCV data
            
        Raises:
            DataError: If data processing fails
            ValidationError: If data validation fails
        """
        try:
            if df.empty:
                self.logger.warning(f"⚠️ Empty DataFrame provided for {symbol} {timeframe}")
                return df
            
            self.logger.debug(f"🔄 Processing {len(df)} rows of OHLCV data for {symbol} {timeframe}")
            
            # Make a copy to avoid modifying original
            processed_df = df.copy()
            
            # Step 1: Basic validation
            processed_df = self._validate_ohlcv_structure(processed_df)
            
            # Step 2: Remove duplicates
            if self.config.remove_duplicates:
                processed_df = self._remove_duplicates(processed_df, symbol)
            
            # Step 3: Sort by timestamp
            processed_df = self._sort_by_timestamp(processed_df)
            
            # Step 4: Validate OHLC logic
            if self.config.validate_ohlc_logic:
                processed_df = self._validate_ohlc_logic(processed_df, symbol)
            
            # Step 5: Handle missing values
            processed_df = await self._handle_missing_values(processed_df, symbol, timeframe)
            
            # Step 6: Detect and handle outliers
            processed_df = self._handle_outliers(processed_df, symbol)
            
            # Step 7: Normalize prices if requested
            if self.config.normalize_prices:
                processed_df = self._normalize_prices(processed_df)
            
            # Step 8: Add derived columns
            processed_df = self._add_derived_columns(processed_df)
            
            # Update statistics
            self.stats['processed_datasets'] += 1
            self.stats['total_rows_processed'] += len(processed_df)
            
            self.logger.debug(f"✅ Processed {len(processed_df)} rows for {symbol} {timeframe}")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"❌ Error processing OHLCV data for {symbol}: {e}")
            raise DataError(
                f"Failed to process OHLCV data for {symbol}",
                data_source="processor",
                symbol=symbol,
                timeframe=timeframe
            ) from e
    
    async def analyze_data_quality(self, df: pd.DataFrame, symbol: str = "",
                                 timeframe: str = "") -> DataQualityReport:
        """
        Analyze data quality and generate a comprehensive report.
        
        Args:
            df (pd.DataFrame): OHLCV data to analyze
            symbol (str): Trading symbol
            timeframe (str): Data timeframe
            
        Returns:
            DataQualityReport: Comprehensive quality assessment
        """
        try:
            if df.empty:
                return DataQualityReport(
                    symbol=symbol,
                    timeframe=timeframe,
                    total_rows=0,
                    missing_values={},
                    duplicate_rows=0,
                    outliers={},
                    data_gaps=[],
                    quality_score=0.0,
                    issues=["Dataset is empty"],
                    recommendations=["Provide valid OHLCV data"]
                )
            
            issues = []
            recommendations = []
            
            # Check missing values
            missing_values = df.isnull().sum().to_dict()
            total_missing = sum(missing_values.values())
            
            if total_missing > 0:
                issues.append(f"Found {total_missing} missing values")
                recommendations.append("Fill missing values using forward-fill or interpolation")
            
            # Check for duplicates
            duplicate_count = df.index.duplicated().sum()
            if duplicate_count > 0:
                issues.append(f"Found {duplicate_count} duplicate timestamps")
                recommendations.append("Remove duplicate entries")
            
            # Detect outliers
            outliers = self._detect_outliers(df)
            total_outliers = sum(outliers.values())
            
            if total_outliers > 0:
                issues.append(f"Found {total_outliers} potential outliers")
                recommendations.append("Review and possibly remove outliers")
            
            # Check for data gaps
            data_gaps = self._detect_data_gaps(df, timeframe)
            if data_gaps:
                issues.append(f"Found {len(data_gaps)} data gaps")
                recommendations.append("Fill data gaps or adjust analysis accordingly")
            
            # Validate OHLC logic
            ohlc_issues = self._validate_ohlc_relationships(df)
            if ohlc_issues:
                issues.extend(ohlc_issues)
                recommendations.append("Correct OHLC relationship violations")
            
            # Calculate quality score (0-100)
            quality_score = self._calculate_quality_score(
                df, missing_values, duplicate_count, outliers, data_gaps, ohlc_issues
            )
            
            return DataQualityReport(
                symbol=symbol,
                timeframe=timeframe,
                total_rows=len(df),
                missing_values=missing_values,
                duplicate_rows=duplicate_count,
                outliers=outliers,
                data_gaps=data_gaps,
                quality_score=quality_score,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing data quality: {e}")
            raise DataError(f"Failed to analyze data quality: {e}") from e
    
    def resample_data(self, df: pd.DataFrame, target_timeframe: str,
                     aggregation_method: str = 'ohlc') -> pd.DataFrame:
        """
        Resample OHLCV data to a different timeframe.
        
        Args:
            df (pd.DataFrame): Source OHLCV data
            target_timeframe (str): Target timeframe (e.g., '1H', '4H', '1D')
            aggregation_method (str): Method for aggregation ('ohlc', 'mean', 'last')
            
        Returns:
            pd.DataFrame: Resampled data
        """
        try:
            if df.empty:
                return df
            
            if aggregation_method == 'ohlc':
                # Standard OHLCV aggregation
                resampled = df.resample(target_timeframe).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
            elif aggregation_method == 'mean':
                resampled = df.resample(target_timeframe).mean()
                
            elif aggregation_method == 'last':
                resampled = df.resample(target_timeframe).last()
                
            else:
                raise ValidationError(f"Unknown aggregation method: {aggregation_method}")
            
            self.logger.debug(f"✅ Resampled {len(df)} -> {len(resampled)} rows to {target_timeframe}")
            return resampled
            
        except Exception as e:
            self.logger.error(f"❌ Error resampling data: {e}")
            raise DataError(f"Failed to resample data: {e}") from e
    
    def calculate_returns(self, df: pd.DataFrame, return_type: str = 'simple',
                         periods: int = 1) -> pd.Series:
        """
        Calculate returns from price data.
        
        Args:
            df (pd.DataFrame): OHLCV data
            return_type (str): 'simple' or 'log'
            periods (int): Number of periods for return calculation
            
        Returns:
            pd.Series: Calculated returns
        """
        try:
            if 'close' not in df.columns:
                raise ValidationError("DataFrame must contain 'close' column")
            
            if return_type == 'simple':
                returns = df['close'].pct_change(periods=periods)
            elif return_type == 'log':
                returns = np.log(df['close'] / df['close'].shift(periods))
            else:
                raise ValidationError(f"Unknown return type: {return_type}")
            
            return returns.dropna()
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating returns: {e}")
            raise DataError(f"Failed to calculate returns: {e}") from e
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical features to the DataFrame.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with additional technical features
        """
        try:
            enhanced_df = df.copy()
            
            # Price-based features
            enhanced_df['price_range'] = enhanced_df['high'] - enhanced_df['low']
            enhanced_df['body_size'] = abs(enhanced_df['close'] - enhanced_df['open'])
            enhanced_df['upper_shadow'] = enhanced_df['high'] - enhanced_df[['open', 'close']].max(axis=1)
            enhanced_df['lower_shadow'] = enhanced_df[['open', 'close']].min(axis=1) - enhanced_df['low']
            
            # Volume-based features
            if 'volume' in enhanced_df.columns:
                enhanced_df['volume_ma5'] = enhanced_df['volume'].rolling(window=5).mean()
                enhanced_df['volume_ratio'] = enhanced_df['volume'] / enhanced_df['volume_ma5']
                enhanced_df['price_volume'] = enhanced_df['close'] * enhanced_df['volume']
            
            # Basic moving averages
            for window in [5, 10, 20]:
                enhanced_df[f'sma_{window}'] = enhanced_df['close'].rolling(window=window).mean()
                enhanced_df[f'price_above_sma_{window}'] = (
                    enhanced_df['close'] > enhanced_df[f'sma_{window}']
                ).astype(int)
            
            # Volatility measures
            enhanced_df['volatility_5'] = enhanced_df['close'].rolling(window=5).std()
            enhanced_df['volatility_20'] = enhanced_df['close'].rolling(window=20).std()
            
            # Returns
            enhanced_df['returns'] = enhanced_df['close'].pct_change()
            enhanced_df['returns_5'] = enhanced_df['close'].pct_change(5)
            
            self.logger.debug(f"✅ Added {len(enhanced_df.columns) - len(df.columns)} technical features")
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"❌ Error adding technical features: {e}")
            raise DataError(f"Failed to add technical features: {e}") from e
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get data processing statistics.
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        return self.stats.copy()
    
    # Private helper methods
    
    def _validate_ohlcv_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that DataFrame has required OHLCV columns."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValidationError(
                f"Missing required columns: {missing_columns}",
                field_name="columns",
                field_value=list(df.columns)
            )
        
        # Ensure numeric types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Remove duplicate rows based on timestamp index."""
        initial_count = len(df)
        df_clean = df[~df.index.duplicated(keep='first')]
        removed_count = initial_count - len(df_clean)
        
        if removed_count > 0:
            self.logger.warning(f"⚠️ Removed {removed_count} duplicate rows for {symbol}")
            self.stats['duplicates_removed'] += removed_count
        
        return df_clean
    
    def _sort_by_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort DataFrame by timestamp index."""
        return df.sort_index()
    
    def _validate_ohlc_logic(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate OHLC relationships and fix obvious errors."""
        initial_count = len(df)
        
        # High should be >= max(open, close)
        # Low should be <= min(open, close)
        
        # Find violations
        high_violations = df['high'] < df[['open', 'close']].max(axis=1)
        low_violations = df['low'] > df[['open', 'close']].min(axis=1)
        
        total_violations = high_violations.sum() + low_violations.sum()
        
        if total_violations > 0:
            self.logger.warning(f"⚠️ Found {total_violations} OHLC logic violations for {symbol}")
            
            # Fix high violations
            df.loc[high_violations, 'high'] = df.loc[high_violations, ['open', 'close']].max(axis=1)
            
            # Fix low violations
            df.loc[low_violations, 'low'] = df.loc[low_violations, ['open', 'close']].min(axis=1)
        
        # Remove rows with invalid data (all prices are zero or negative)
        invalid_rows = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        df_clean = df[~invalid_rows]
        
        removed_count = len(df) - len(df_clean)
        if removed_count > 0:
            self.logger.warning(f"⚠️ Removed {removed_count} rows with invalid prices for {symbol}")
        
        return df_clean
    
    async def _handle_missing_values(self, df: pd.DataFrame, symbol: str, 
                                   timeframe: str) -> pd.DataFrame:
        """Handle missing values according to configuration."""
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            return df
        
        self.logger.debug(f"🔄 Handling {missing_count} missing values for {symbol}")
        
        if self.config.fill_missing_method == 'drop':
            df_clean = df.dropna()
        elif self.config.fill_missing_method == 'forward':
            df_clean = df.fillna(method='ffill')
        elif self.config.fill_missing_method == 'backward':
            df_clean = df.fillna(method='bfill')
        elif self.config.fill_missing_method == 'interpolate':
            df_clean = df.interpolate(method='linear')
        else:
            self.logger.warning(f"⚠️ Unknown fill method: {self.config.fill_missing_method}")
            df_clean = df.dropna()
        
        filled_count = missing_count - df_clean.isnull().sum().sum()
        if filled_count > 0:
            self.stats['missing_values_filled'] += filled_count
        
        return df_clean
    
    def _handle_outliers(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Detect and handle outliers."""
        if self.config.outlier_detection_method == 'none':
            return df
        
        outliers_detected = self._detect_outliers(df)
        total_outliers = sum(outliers_detected.values())
        
        if total_outliers == 0:
            return df
        
        self.logger.debug(f"🎯 Detected {total_outliers} outliers for {symbol}")
        
        # For now, we'll cap outliers rather than remove them to preserve data continuity
        df_clean = df.copy()
        
        for column in ['open', 'high', 'low', 'close']:
            if column in df_clean.columns and outliers_detected.get(column, 0) > 0:
                # Use IQR method to cap outliers
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean[column] = df_clean[column].clip(lower=lower_bound, upper=upper_bound)
        
        self.stats['outliers_removed'] += total_outliers
        return df_clean
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers in OHLCV data."""
        outliers = {}
        
        for column in ['open', 'high', 'low', 'close', 'volume']:
            if column not in df.columns:
                continue
                
            if self.config.outlier_detection_method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                
            elif self.config.outlier_detection_method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outlier_mask = z_scores > self.config.outlier_threshold
                
            else:
                outlier_mask = pd.Series([False] * len(df), index=df.index)
            
            outliers[column] = outlier_mask.sum()
        
        return outliers
    
    def _normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize prices to percentage changes from first value."""
        df_normalized = df.copy()
        
        price_columns = ['open', 'high', 'low', 'close']
        for column in price_columns:
            if column in df_normalized.columns:
                first_value = df_normalized[column].iloc[0]
                if first_value != 0:
                    df_normalized[f'{column}_normalized'] = (
                        df_normalized[column] / first_value - 1
                    ) * 100
        
        return df_normalized
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived columns."""
        enhanced_df = df.copy()
        
        # Typical Price
        enhanced_df['typical_price'] = (
            enhanced_df['high'] + enhanced_df['low'] + enhanced_df['close']
        ) / 3
        
        # Price change
        enhanced_df['price_change'] = enhanced_df['close'].diff()
        enhanced_df['price_change_pct'] = enhanced_df['close'].pct_change() * 100
        
        # True Range (for ATR calculation)
        enhanced_df['true_range'] = np.maximum(
            enhanced_df['high'] - enhanced_df['low'],
            np.maximum(
                abs(enhanced_df['high'] - enhanced_df['close'].shift(1)),
                abs(enhanced_df['low'] - enhanced_df['close'].shift(1))
            )
        )
        
        return enhanced_df
    
    def _detect_data_gaps(self, df: pd.DataFrame, timeframe: str) -> List[Tuple[datetime, datetime]]:
        """Detect gaps in time series data."""
        if len(df) < 2:
            return []
        
        # Determine expected frequency based on timeframe
        freq_mapping = {
            '1m': pd.Timedelta(minutes=1),
            '3m': pd.Timedelta(minutes=3),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '30m': pd.Timedelta(minutes=30),
            '1h': pd.Timedelta(hours=1),
            '2h': pd.Timedelta(hours=2),
            '4h': pd.Timedelta(hours=4),
            '6h': pd.Timedelta(hours=6),
            '8h': pd.Timedelta(hours=8),
            '12h': pd.Timedelta(hours=12),
            '1d': pd.Timedelta(days=1),
            '3d': pd.Timedelta(days=3),
            '1w': pd.Timedelta(weeks=1),
        }
        
        expected_freq = freq_mapping.get(timeframe.lower())
        if expected_freq is None:
            return []  # Unknown timeframe
        
        gaps = []
        timestamps = df.index
        
        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            if time_diff > expected_freq * 1.5:  # Allow 50% tolerance
                gaps.append((timestamps[i-1], timestamps[i]))
        
        return gaps
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> List[str]:
        """Validate OHLC relationships and return issues."""
        issues = []
        
        # Check if high >= max(open, close)
        high_violations = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
        if high_violations > 0:
            issues.append(f"{high_violations} rows where high < max(open, close)")
        
        # Check if low <= min(open, close)
        low_violations = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
        if low_violations > 0:
            issues.append(f"{low_violations} rows where low > min(open, close)")
        
        # Check for zero or negative prices
        zero_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        if zero_prices > 0:
            issues.append(f"{zero_prices} rows with zero or negative prices")
        
        # Check for negative volume
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"{negative_volume} rows with negative volume")
        
        return issues
    
    def _calculate_quality_score(self, df: pd.DataFrame, missing_values: Dict[str, int],
                               duplicate_count: int, outliers: Dict[str, int],
                               data_gaps: List, ohlc_issues: List[str]) -> float:
        """Calculate overall data quality score (0-100)."""
        total_rows = len(df)
        if total_rows == 0:
            return 0.0
        
        # Start with perfect score
        score = 100.0
        
        # Deduct points for missing values
        total_missing = sum(missing_values.values())
        missing_penalty = (total_missing / (total_rows * len(df.columns))) * 20
        score -= missing_penalty
        
        # Deduct points for duplicates
        duplicate_penalty = (duplicate_count / total_rows) * 15
        score -= duplicate_penalty
        
        # Deduct points for outliers
        total_outliers = sum(outliers.values())
        outlier_penalty = (total_outliers / (total_rows * 5)) * 10  # 5 = OHLCV columns
        score -= outlier_penalty
        
        # Deduct points for data gaps
        gap_penalty = min(len(data_gaps) * 2, 20)
        score -= gap_penalty
        
        # Deduct points for OHLC issues
        ohlc_penalty = min(len(ohlc_issues) * 5, 25)
        score -= ohlc_penalty
        
        return max(0.0, score)