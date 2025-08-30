import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime

from ..base_strategy import (
    BaseStrategy, 
    StrategyType, 
    Signal, 
    SignalType, 
    SignalStrength
)


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover trading strategy implementation.
    
    This strategy uses two exponential moving averages to identify trend changes
    and generate trading signals. It's one of the most popular and simple
    trend-following strategies.
    """
    
    # Strategy metadata
    strategy_type = StrategyType.TREND_FOLLOWING
    name = "EMA Crossover"
    description = "Exponential Moving Average crossover strategy for trend following"
    version = "1.0.0"
    
    def __init__(self, symbol: str = "BTCUSDT", timeframe: str = "15m", **kwargs):
        """
        Initialize EMA Crossover strategy.
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Analysis timeframe
            **kwargs: Additional parameters
                - fast_ema (int): Fast EMA period (default: 12)
                - slow_ema (int): Slow EMA period (default: 26)
                - volume_filter (bool): Use volume confirmation (default: True)
                - min_volume_ratio (float): Minimum volume ratio for signal (default: 1.2)
                - stop_loss_pct (float): Stop loss percentage (default: 2.0)
                - take_profit_pct (float): Take profit percentage (default: 4.0)
                - min_trend_strength (float): Minimum trend strength for signal (default: 0.5)
        """
        super().__init__(symbol, timeframe, **kwargs)
        
        # Strategy parameters with defaults
        self.parameters = {
            'fast_ema': kwargs.get('fast_ema', 12),
            'slow_ema': kwargs.get('slow_ema', 26),
            'volume_filter': kwargs.get('volume_filter', True),
            'min_volume_ratio': kwargs.get('min_volume_ratio', 1.2),
            'stop_loss_pct': kwargs.get('stop_loss_pct', 2.0),
            'take_profit_pct': kwargs.get('take_profit_pct', 4.0),
            'min_trend_strength': kwargs.get('min_trend_strength', 0.5)
        }
        
        # Validate parameters
        self._validate_parameters()
        
        # Internal state variables
        self.last_crossover_type: Optional[str] = None
        self.crossover_confirmation_bars = 0
        self.trend_direction: Optional[str] = None
        
        self.logger.info(
            f"🎯 EMA Crossover Strategy initialized: "
            f"Fast EMA={self.parameters['fast_ema']}, "
            f"Slow EMA={self.parameters['slow_ema']}"
        )
    
    def _validate_parameters(self):
        """Validate strategy parameters."""
        if self.parameters['fast_ema'] >= self.parameters['slow_ema']:
            raise ValueError("Fast EMA period must be less than slow EMA period")
        
        if self.parameters['fast_ema'] < 2:
            raise ValueError("Fast EMA period must be at least 2")
        
        if self.parameters['slow_ema'] < 5:
            raise ValueError("Slow EMA period must be at least 5")
        
        if not 0.1 <= self.parameters['stop_loss_pct'] <= 10.0:
            raise ValueError("Stop loss percentage must be between 0.1% and 10%")
        
        if not 0.1 <= self.parameters['take_profit_pct'] <= 20.0:
            raise ValueError("Take profit percentage must be between 0.1% and 20%")
    
    async def initialize(self) -> bool:
        """
        Initialize the EMA Crossover strategy.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("🔧 Initializing EMA Crossover Strategy...")
            
            # Validate that we have enough data for indicators
            min_periods = max(self.parameters['fast_ema'], self.parameters['slow_ema']) * 3
            
            if self.market_data is not None and len(self.market_data) < min_periods:
                self.logger.warning(
                    f"⚠️ Insufficient data: need at least {min_periods} periods, "
                    f"got {len(self.market_data)}"
                )
                return False
            
            self.is_initialized = True
            self.logger.info("✅ EMA Crossover Strategy initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize EMA Crossover Strategy: {e}")
            return False
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate EMA indicators and supporting metrics.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            
        Returns:
            Dict[str, pd.Series]: Dictionary of calculated indicators
        """
        indicators = {}
        
        try:
            # Calculate EMAs
            indicators['fast_ema'] = data['close'].ewm(
                span=self.parameters['fast_ema'], adjust=False
            ).mean()
            
            indicators['slow_ema'] = data['close'].ewm(
                span=self.parameters['slow_ema'], adjust=False
            ).mean()
            
            # Calculate EMA difference and percentage difference
            indicators['ema_diff'] = indicators['fast_ema'] - indicators['slow_ema']
            indicators['ema_diff_pct'] = (
                (indicators['fast_ema'] - indicators['slow_ema']) / indicators['slow_ema'] * 100
            )
            
            # Calculate crossover signals (1 for bullish, -1 for bearish, 0 for no crossover)
            ema_diff_shifted = indicators['ema_diff'].shift(1)
            indicators['crossover'] = np.where(
                (indicators['ema_diff'] > 0) & (ema_diff_shifted <= 0), 1,
                np.where(
                    (indicators['ema_diff'] < 0) & (ema_diff_shifted >= 0), -1, 0
                )
            )
            
            # Calculate trend strength (normalized EMA difference)
            price_range = data['high'].rolling(20).max() - data['low'].rolling(20).min()
            indicators['trend_strength'] = abs(indicators['ema_diff']) / price_range
            indicators['trend_strength'] = indicators['trend_strength'].fillna(0)
            
            # Volume indicators (if volume filter is enabled)
            if self.parameters['volume_filter'] and 'volume' in data.columns:
                indicators['volume_sma'] = data['volume'].rolling(20).mean()
                indicators['volume_ratio'] = data['volume'] / indicators['volume_sma']
                indicators['volume_ratio'] = indicators['volume_ratio'].fillna(1.0)
            
            # Price momentum (rate of change)
            indicators['price_momentum'] = data['close'].pct_change(5) * 100
            
            # Volatility (ATR-like measure)
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = true_range.rolling(14).mean()
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating EMA indicators: {e}")
            return {}
    
    async def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate trading signal based on EMA crossover logic.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns
            
        Returns:
            Optional[Signal]: Generated signal or None
        """
        try:
            if len(data) < max(self.parameters['fast_ema'], self.parameters['slow_ema']) + 10:
                return None
            
            # Get latest indicators
            indicators = self.calculate_indicators(data)
            if not indicators:
                return None
            
            # Get current values
            current_price = float(data['close'].iloc[-1])
            current_crossover = int(indicators['crossover'].iloc[-1])
            current_trend_strength = float(indicators['trend_strength'].iloc[-1])
            current_ema_diff_pct = float(indicators['ema_diff_pct'].iloc[-1])
            
            # Volume confirmation if enabled
            volume_confirmed = True
            if self.parameters['volume_filter'] and 'volume_ratio' in indicators:
                volume_ratio = float(indicators['volume_ratio'].iloc[-1])
                volume_confirmed = volume_ratio >= self.parameters['min_volume_ratio']
            
            # Check for crossover signals
            signal_type = None
            signal_strength = SignalStrength.WEAK
            confidence = 0.5
            metadata = {
                'fast_ema': float(indicators['fast_ema'].iloc[-1]),
                'slow_ema': float(indicators['slow_ema'].iloc[-1]),
                'ema_diff_pct': current_ema_diff_pct,
                'trend_strength': current_trend_strength,
                'volume_confirmed': volume_confirmed,
                'crossover_type': current_crossover
            }
            
            # Bullish crossover (golden cross)
            if current_crossover == 1:
                if self._validate_bullish_signal(indicators, volume_confirmed):
                    signal_type = SignalType.BUY
                    self.last_crossover_type = "bullish"
                    self.crossover_confirmation_bars = 0
                    self.trend_direction = "bullish"
                    
                    # Calculate signal strength and confidence
                    signal_strength, confidence = self._calculate_signal_metrics(
                        indicators, "bullish"
                    )
            
            # Bearish crossover (death cross)
            elif current_crossover == -1:
                if self._validate_bearish_signal(indicators, volume_confirmed):
                    # If we have a long position, generate sell signal
                    if self.current_position == "long":
                        signal_type = SignalType.SELL
                    
                    self.last_crossover_type = "bearish"
                    self.crossover_confirmation_bars = 0
                    self.trend_direction = "bearish"
                    
                    # Calculate signal strength and confidence
                    signal_strength, confidence = self._calculate_signal_metrics(
                        indicators, "bearish"
                    )
            
            # Generate signal if conditions are met
            if signal_type:
                # Calculate stop loss and take profit levels
                stop_loss, take_profit = self._calculate_risk_levels(
                    current_price, signal_type, indicators
                )
                
                # Calculate position size based on risk
                position_size = self._calculate_position_size_ratio(
                    current_price, stop_loss, signal_strength
                )
                
                signal = Signal(
                    type=signal_type,
                    strength=signal_strength,
                    price=current_price,
                    timestamp=data.index[-1],
                    confidence=confidence,
                    metadata=metadata,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size
                )
                
                self.logger.info(
                    f"📊 EMA Crossover Signal: {signal_type.value} at {current_price} "
                    f"(confidence: {confidence:.2f}, strength: {signal_strength.value})"
                )
                
                return signal
            
            # Update confirmation bars counter
            if self.last_crossover_type:
                self.crossover_confirmation_bars += 1
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error generating EMA crossover signal: {e}")
            return None
    
    def _validate_bullish_signal(self, indicators: Dict[str, pd.Series], 
                                volume_confirmed: bool) -> bool:
        """
        Validate bullish crossover signal with additional filters.
        
        Args:
            indicators (Dict[str, pd.Series]): Calculated indicators
            volume_confirmed (bool): Whether volume confirms the signal
            
        Returns:
            bool: True if signal is valid
        """
        try:
            current_trend_strength = float(indicators['trend_strength'].iloc[-1])
            
            # Check minimum trend strength
            if current_trend_strength < self.parameters['min_trend_strength']:
                self.logger.debug(f"🔍 Bullish signal rejected: trend strength too weak ({current_trend_strength:.3f})")
                return False
            
            # Check volume confirmation if enabled
            if self.parameters['volume_filter'] and not volume_confirmed:
                self.logger.debug("🔍 Bullish signal rejected: volume not confirmed")
                return False
            
            # Check that we're not already in a bullish trend
            if self.trend_direction == "bullish" and self.crossover_confirmation_bars < 5:
                self.logger.debug("🔍 Bullish signal rejected: already in bullish trend")
                return False
            
            # Additional momentum check
            if 'price_momentum' in indicators:
                momentum = float(indicators['price_momentum'].iloc[-1])
                if momentum < -2.0:  # Don't buy into strong downward momentum
                    self.logger.debug(f"🔍 Bullish signal rejected: negative momentum ({momentum:.2f}%)")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating bullish signal: {e}")
            return False
    
    def _validate_bearish_signal(self, indicators: Dict[str, pd.Series], 
                                volume_confirmed: bool) -> bool:
        """
        Validate bearish crossover signal with additional filters.
        
        Args:
            indicators (Dict[str, pd.Series]): Calculated indicators
            volume_confirmed (bool): Whether volume confirms the signal
            
        Returns:
            bool: True if signal is valid
        """
        try:
            current_trend_strength = float(indicators['trend_strength'].iloc[-1])
            
            # Check minimum trend strength
            if current_trend_strength < self.parameters['min_trend_strength']:
                self.logger.debug(f"🔍 Bearish signal rejected: trend strength too weak ({current_trend_strength:.3f})")
                return False
            
            # Check volume confirmation if enabled
            if self.parameters['volume_filter'] and not volume_confirmed:
                self.logger.debug("🔍 Bearish signal rejected: volume not confirmed")
                return False
            
            # Check that we're not already in a bearish trend
            if self.trend_direction == "bearish" and self.crossover_confirmation_bars < 5:
                self.logger.debug("🔍 Bearish signal rejected: already in bearish trend")
                return False
            
            # Additional momentum check
            if 'price_momentum' in indicators:
                momentum = float(indicators['price_momentum'].iloc[-1])
                if momentum > 2.0:  # Don't sell into strong upward momentum
                    self.logger.debug(f"🔍 Bearish signal rejected: positive momentum ({momentum:.2f}%)")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating bearish signal: {e}")
            return False
    
    def _calculate_signal_metrics(self, indicators: Dict[str, pd.Series], 
                                 signal_direction: str) -> tuple[SignalStrength, float]:
        """
        Calculate signal strength and confidence based on multiple factors.
        
        Args:
            indicators (Dict[str, pd.Series]): Calculated indicators
            signal_direction (str): "bullish" or "bearish"
            
        Returns:
            tuple[SignalStrength, float]: Signal strength and confidence
        """
        try:
            # Base confidence
            confidence = 0.6
            strength = SignalStrength.MODERATE
            
            # Trend strength factor
            trend_strength = float(indicators['trend_strength'].iloc[-1])
            confidence += min(trend_strength * 0.3, 0.25)  # Up to +0.25
            
            # EMA separation factor
            ema_diff_pct = abs(float(indicators['ema_diff_pct'].iloc[-1]))
            if ema_diff_pct > 1.0:  # Strong separation
                confidence += 0.1
                strength = SignalStrength.STRONG
            elif ema_diff_pct < 0.2:  # Weak separation
                confidence -= 0.15
                strength = SignalStrength.WEAK
            
            # Volume confirmation bonus
            if self.parameters['volume_filter'] and 'volume_ratio' in indicators:
                volume_ratio = float(indicators['volume_ratio'].iloc[-1])
                if volume_ratio > 1.5:  # High volume
                    confidence += 0.1
                elif volume_ratio < self.parameters['min_volume_ratio']:
                    confidence -= 0.2
            
            # Momentum alignment
            if 'price_momentum' in indicators:
                momentum = float(indicators['price_momentum'].iloc[-1])
                if signal_direction == "bullish" and momentum > 0.5:
                    confidence += 0.1
                elif signal_direction == "bearish" and momentum < -0.5:
                    confidence += 0.1
                elif (signal_direction == "bullish" and momentum < -1.0) or \
                     (signal_direction == "bearish" and momentum > 1.0):
                    confidence -= 0.15
            
            # Adjust strength based on final confidence
            if confidence > 0.8:
                strength = SignalStrength.VERY_STRONG
            elif confidence > 0.7:
                strength = SignalStrength.STRONG
            elif confidence < 0.4:
                strength = SignalStrength.WEAK
            
            # Clamp confidence between 0.1 and 0.95
            confidence = max(0.1, min(0.95, confidence))
            
            return strength, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating signal metrics: {e}")
            return SignalStrength.WEAK, 0.3
    
    def _calculate_risk_levels(self, current_price: float, signal_type: SignalType, 
                              indicators: Dict[str, pd.Series]) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            current_price (float): Current market price
            signal_type (SignalType): Type of signal
            indicators (Dict[str, pd.Series]): Calculated indicators
            
        Returns:
            tuple[Optional[float], Optional[float]]: Stop loss and take profit prices
        """
        try:
            stop_loss = None
            take_profit = None
            
            if signal_type == SignalType.BUY:
                # Stop loss below slow EMA or percentage-based
                slow_ema = float(indicators['slow_ema'].iloc[-1])
                percentage_stop = current_price * (1 - self.parameters['stop_loss_pct'] / 100)
                
                # Use the higher of the two (closer to current price, less risk)
                stop_loss = max(slow_ema * 0.995, percentage_stop)  # 0.5% buffer below EMA
                
                # Take profit
                take_profit = current_price * (1 + self.parameters['take_profit_pct'] / 100)
                
            elif signal_type == SignalType.SELL:
                # Stop loss above slow EMA or percentage-based
                slow_ema = float(indicators['slow_ema'].iloc[-1])
                percentage_stop = current_price * (1 + self.parameters['stop_loss_pct'] / 100)
                
                # Use the lower of the two (closer to current price, less risk)
                stop_loss = min(slow_ema * 1.005, percentage_stop)  # 0.5% buffer above EMA
                
                # Take profit
                take_profit = current_price * (1 - self.parameters['take_profit_pct'] / 100)
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk levels: {e}")
            return None, None
    
    def _calculate_position_size_ratio(self, current_price: float, stop_loss: Optional[float], 
                                      signal_strength: SignalStrength) -> Optional[float]:
        """
        Calculate position size as a ratio of available capital.
        
        Args:
            current_price (float): Current market price
            stop_loss (Optional[float]): Stop loss price
            signal_strength (SignalStrength): Strength of the signal
            
        Returns:
            Optional[float]: Position size ratio (0.0 to 1.0)
        """
        try:
            # Base position size based on signal strength
            base_sizes = {
                SignalStrength.WEAK: 0.25,
                SignalStrength.MODERATE: 0.5,
                SignalStrength.STRONG: 0.75,
                SignalStrength.VERY_STRONG: 1.0
            }
            
            base_size = base_sizes.get(signal_strength, 0.5)
            
            # Adjust for stop loss distance (risk adjustment)
            if stop_loss:
                risk_distance = abs(current_price - stop_loss) / current_price
                
                # Reduce position size for wider stops
                if risk_distance > 0.03:  # >3% stop
                    base_size *= 0.7
                elif risk_distance > 0.05:  # >5% stop
                    base_size *= 0.5
            
            # Maximum position size is 80% of capital
            return min(base_size * 0.8, 0.8)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0.25  # Conservative default
    
    def validate_signal(self, signal: Signal, data: pd.DataFrame) -> bool:
        """
        Final validation of generated signal.
        
        Args:
            signal (Signal): Signal to validate
            data (pd.DataFrame): Current market data
            
        Returns:
            bool: True if signal is valid
        """
        try:
            # Basic validation
            if not signal or signal.confidence < 0.3:
                self.logger.debug("🔍 Signal rejected: low confidence")
                return False
            
            # Price validation (ensure reasonable price)
            current_price = float(data['close'].iloc[-1])
            if abs(signal.price - current_price) / current_price > 0.01:  # 1% tolerance
                self.logger.debug("🔍 Signal rejected: price mismatch")
                return False
            
            # Risk validation
            if signal.stop_loss:
                risk_pct = abs(signal.price - signal.stop_loss) / signal.price * 100
                if risk_pct > 10:  # Maximum 10% risk per trade
                    self.logger.debug(f"🔍 Signal rejected: risk too high ({risk_pct:.1f}%)")
                    return False
            
            # Position size validation
            if signal.position_size and signal.position_size > 0.8:
                self.logger.debug("🔍 Signal rejected: position size too large")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating signal: {e}")
            return False
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """
        Get current strategy state and indicators.
        
        Returns:
            Dict[str, Any]: Strategy state information
        """
        base_state = self.get_status()
        
        # Add EMA-specific state
        ema_state = {
            'last_crossover_type': self.last_crossover_type,
            'crossover_confirmation_bars': self.crossover_confirmation_bars,
            'trend_direction': self.trend_direction,
            'fast_ema_period': self.parameters['fast_ema'],
            'slow_ema_period': self.parameters['slow_ema']
        }
        
        # Add current indicator values if available
        if self.indicators:
            try:
                ema_state.update({
                    'current_fast_ema': float(self.indicators['fast_ema'].iloc[-1]) if 'fast_ema' in self.indicators else None,
                    'current_slow_ema': float(self.indicators['slow_ema'].iloc[-1]) if 'slow_ema' in self.indicators else None,
                    'current_ema_diff_pct': float(self.indicators['ema_diff_pct'].iloc[-1]) if 'ema_diff_pct' in self.indicators else None,
                    'current_trend_strength': float(self.indicators['trend_strength'].iloc[-1]) if 'trend_strength' in self.indicators else None
                })
            except (KeyError, IndexError):
                pass
        
        base_state.update(ema_state)
        return base_state
    
    async def optimize_parameters(self, historical_data: pd.DataFrame, 
                          optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize strategy parameters using historical data.
        
        Args:
            historical_data (pd.DataFrame): Historical OHLCV data
            optimization_metric (str): Metric to optimize ('sharpe_ratio', 'total_return', 'win_rate')
            
        Returns:
            Dict[str, Any]: Optimization results with best parameters
        """
        self.logger.info(f"🔧 Optimizing EMA Crossover parameters using {optimization_metric}...")
        
        # Parameter ranges to test
        fast_ema_range = [5, 8, 10, 12, 15, 18, 21]
        slow_ema_range = [20, 26, 30, 35, 40, 50, 60]
        stop_loss_range = [1.0, 1.5, 2.0, 2.5, 3.0]
        take_profit_range = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        best_result = None
        best_metric_value = float('-inf') if optimization_metric != 'max_drawdown' else float('inf')
        best_params = {}
        
        total_combinations = len(fast_ema_range) * len(slow_ema_range) * len(stop_loss_range) * len(take_profit_range)
        tested_combinations = 0
        
        try:
            for fast_ema in fast_ema_range:
                for slow_ema in slow_ema_range:
                    if fast_ema >= slow_ema:
                        continue
                    
                    for stop_loss in stop_loss_range:
                        for take_profit in take_profit_range:
                            tested_combinations += 1
                            
                            # Create test instance with new parameters
                            test_params = self.parameters.copy()
                            test_params.update({
                                'fast_ema': fast_ema,
                                'slow_ema': slow_ema,
                                'stop_loss_pct': stop_loss,
                                'take_profit_pct': take_profit
                            })
                            
                            # Create temporary strategy instance
                            temp_strategy = EMACrossoverStrategy(
                                symbol=self.symbol,
                                timeframe=self.timeframe,
                                **test_params
                            )
                            
                            try:
                                # Run backtest
                                result = await temp_strategy.backtest(historical_data)
                                
                                # Check if this is the best result
                                current_metric_value = getattr(result, optimization_metric)
                                
                                is_better = False
                                if optimization_metric == 'max_drawdown':
                                    is_better = current_metric_value < best_metric_value
                                else:
                                    is_better = current_metric_value > best_metric_value
                                
                                if is_better and result.total_trades >= 10:  # Minimum trade requirement
                                    best_metric_value = current_metric_value
                                    best_result = result
                                    best_params = test_params.copy()
                                    
                                    self.logger.info(
                                        f"🎯 New best result: {optimization_metric}={current_metric_value:.3f} "
                                        f"(Fast EMA: {fast_ema}, Slow EMA: {slow_ema}, "
                                        f"Stop Loss: {stop_loss}%, Take Profit: {take_profit}%)"
                                    )
                            
                            except Exception as e:
                                self.logger.debug(f"Optimization test failed for params {test_params}: {e}")
                                continue
                            
                            # Progress logging
                            if tested_combinations % 50 == 0:
                                progress = tested_combinations / total_combinations * 100
                                self.logger.info(f"🔄 Optimization progress: {progress:.1f}% ({tested_combinations}/{total_combinations})")
            
            if best_result and best_params:
                self.logger.info(
                    f"✅ Optimization complete! Best {optimization_metric}: {best_metric_value:.3f}"
                )
                self.logger.info(f"📊 Best parameters: {best_params}")
                
                return {
                    'best_parameters': best_params,
                    'best_metric_value': best_metric_value,
                    'optimization_metric': optimization_metric,
                    'backtest_result': best_result,
                    'total_combinations_tested': tested_combinations
                }
            else:
                self.logger.warning("❌ Optimization failed: no valid results found")
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Optimization failed: {e}")
            return {}