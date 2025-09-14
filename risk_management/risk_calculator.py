"""
Risk Calculator
===============

Comprehensive risk calculation system for trading operations.
Handles risk metrics, risk-reward ratios, and trade risk assessment.

This module provides tools for:
- Calculating risk-reward ratios
- Stop loss and take profit price calculations
- Risk metrics computation
- Trade risk assessment
- Portfolio risk analysis

Author: dat-ns
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from datetime import datetime


class RiskLevel(Enum):
    """Risk level enumeration."""
    VERY_LOW = "very_low"
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TradeDirection(Enum):
    """Trade direction enumeration."""
    LONG = "long"
    SHORT = "short"


@dataclass
class RiskParameters:
    """Risk parameters for trade calculations."""
    max_account_risk: float = 0.02  # 2% default
    max_position_risk: float = 0.05  # 5% per position
    min_risk_reward_ratio: float = 2.0  # Minimum 2:1 ratio
    max_leverage: int = 10
    max_daily_loss: float = 0.10  # 10% daily loss limit
    max_drawdown: float = 0.20  # 20% max drawdown
    
    def __post_init__(self):
        """Validate risk parameters."""
        if not 0 < self.max_account_risk <= 0.1:
            raise ValueError("max_account_risk must be between 0 and 0.1 (10%)")
        if not 0 < self.max_position_risk <= 0.2:
            raise ValueError("max_position_risk must be between 0 and 0.2 (20%)")
        if self.min_risk_reward_ratio < 1.0:
            raise ValueError("min_risk_reward_ratio must be >= 1.0")


@dataclass
class RiskMetrics:
    """Risk metrics for a trade or position."""
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    risk_percentage: float
    max_loss: float
    max_gain: float
    breakeven_price: float
    risk_level: RiskLevel
    trade_direction: TradeDirection
    leverage: int = 1
    
    # Calculated fields
    price_risk_distance: float = field(init=False)
    price_reward_distance: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.price_risk_distance = abs(self.entry_price - self.stop_loss)
        self.price_reward_distance = abs(self.take_profit - self.entry_price)


@dataclass
class PositionRisk:
    """Risk assessment for a position."""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    risk_percentage: float
    max_loss: float
    position_value: float
    margin_required: float
    leverage: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeRisk:
    """Risk assessment for a potential trade."""
    symbol: str
    trade_direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    probability_success: Optional[float] = None
    expected_value: Optional[float] = None
    max_adverse_excursion: Optional[float] = None


class RiskCalculator:
    """
    Comprehensive risk calculator for trading operations.
    
    Handles all risk-related calculations including:
    - Risk-reward ratios
    - Stop loss and take profit calculations
    - Position risk assessment
    - Portfolio risk metrics
    """
    
    def __init__(self, risk_params: Optional[RiskParameters] = None):
        """
        Initialize risk calculator.
        
        Args:
            risk_params: Risk parameters configuration
        """
        self.risk_params = risk_params or RiskParameters()
        self.logger = logging.getLogger(__name__)
        
        # Risk calculation precision
        self.decimal_places = 8
        self.price_precision = 4
        
        self.logger.info("🛡️ Risk Calculator initialized")
    
    def calculate_stop_loss_price(
        self,
        entry_price: float,
        risk_amount: float,
        position_size: float,
        trade_direction: TradeDirection,
        leverage: int = 1
    ) -> float:
        """
        Calculate stop loss price based on risk amount.
        
        Args:
            entry_price: Entry price for the trade
            risk_amount: Maximum risk amount in base currency
            position_size: Position size in base asset
            trade_direction: Long or short position
            leverage: Trading leverage
            
        Returns:
            float: Stop loss price
        """
        try:
            # Calculate price distance for stop loss
            price_distance = risk_amount / (position_size * leverage)
            
            if trade_direction == TradeDirection.LONG:
                stop_loss = entry_price - price_distance
            else:  # SHORT
                stop_loss = entry_price + price_distance
            
            # Ensure stop loss is positive
            if stop_loss <= 0:
                raise ValueError("Calculated stop loss price is invalid (≤ 0)")
            
            return round(stop_loss, self.price_precision)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating stop loss: {e}")
            raise
    
    def calculate_take_profit_price(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float,
        trade_direction: TradeDirection
    ) -> float:
        """
        Calculate take profit price based on risk-reward ratio.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_reward_ratio: Desired risk-reward ratio
            trade_direction: Long or short position
            
        Returns:
            float: Take profit price
        """
        try:
            # Calculate risk distance
            risk_distance = abs(entry_price - stop_loss)
            
            # Calculate reward distance
            reward_distance = risk_distance * risk_reward_ratio
            
            if trade_direction == TradeDirection.LONG:
                take_profit = entry_price + reward_distance
            else:  # SHORT
                take_profit = entry_price - reward_distance
            
            # Ensure take profit is positive
            if take_profit <= 0:
                raise ValueError("Calculated take profit price is invalid (≤ 0)")
            
            return round(take_profit, self.price_precision)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating take profit: {e}")
            raise
    
    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> float:
        """
        Calculate risk-reward ratio for a trade.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            float: Risk-reward ratio
        """
        try:
            risk_distance = abs(entry_price - stop_loss)
            reward_distance = abs(take_profit - entry_price)
            
            if risk_distance == 0:
                raise ValueError("Risk distance cannot be zero")
            
            ratio = reward_distance / risk_distance
            return round(ratio, 2)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk-reward ratio: {e}")
            raise
    
    def calculate_position_risk(
        self,
        entry_price: float,
        stop_loss: float,
        position_size: float,
        leverage: int = 1
    ) -> float:
        """
        Calculate position risk amount.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            position_size: Position size
            leverage: Trading leverage
            
        Returns:
            float: Risk amount in base currency
        """
        try:
            price_distance = abs(entry_price - stop_loss)
            risk_amount = price_distance * position_size * leverage
            return round(risk_amount, self.decimal_places)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position risk: {e}")
            raise
    
    def calculate_breakeven_price(
        self,
        entry_price: float,
        fees_percentage: float = 0.001,  # 0.1% default fees
        trade_direction: TradeDirection = TradeDirection.LONG
    ) -> float:
        """
        Calculate breakeven price including fees.
        
        Args:
            entry_price: Entry price
            fees_percentage: Trading fees percentage
            trade_direction: Trade direction
            
        Returns:
            float: Breakeven price
        """
        try:
            total_fees = fees_percentage * 2  # Entry + Exit fees
            
            if trade_direction == TradeDirection.LONG:
                breakeven = entry_price * (1 + total_fees)
            else:  # SHORT
                breakeven = entry_price * (1 - total_fees)
            
            return round(breakeven, self.price_precision)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating breakeven price: {e}")
            raise
    
    def assess_risk_level(
        self,
        risk_percentage: float,
        risk_reward_ratio: float,
        leverage: int = 1
    ) -> RiskLevel:
        """
        Assess overall risk level for a trade.
        
        Args:
            risk_percentage: Risk as percentage of account
            risk_reward_ratio: Risk-reward ratio
            leverage: Trading leverage
            
        Returns:
            RiskLevel: Assessed risk level
        """
        try:
            # Calculate risk score
            risk_score = 0
            
            # Risk percentage scoring
            if risk_percentage > 0.05:  # > 5%
                risk_score += 3
            elif risk_percentage > 0.03:  # > 3%
                risk_score += 2
            elif risk_percentage > 0.01:  # > 1%
                risk_score += 1
            
            # Risk-reward ratio scoring
            if risk_reward_ratio < 1.5:
                risk_score += 3
            elif risk_reward_ratio < 2.0:
                risk_score += 2
            elif risk_reward_ratio < 3.0:
                risk_score += 1
            
            # Leverage scoring
            if leverage > 10:
                risk_score += 3
            elif leverage > 5:
                risk_score += 2
            elif leverage > 2:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 7:
                return RiskLevel.VERY_HIGH
            elif risk_score >= 5:
                return RiskLevel.HIGH
            elif risk_score >= 3:
                return RiskLevel.MEDIUM
            elif risk_score >= 1:
                return RiskLevel.LOW
            else:
                return RiskLevel.VERY_LOW
                
        except Exception as e:
            self.logger.error(f"❌ Error assessing risk level: {e}")
            return RiskLevel.HIGH  # Default to high risk on error
    
    def calculate_comprehensive_risk_metrics(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        trade_direction: TradeDirection,
        account_balance: float,
        leverage: int = 1
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a trade.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size: Position size
            trade_direction: Trade direction
            account_balance: Account balance
            leverage: Trading leverage
            
        Returns:
            RiskMetrics: Comprehensive risk metrics
        """
        try:
            # Calculate risk and reward amounts
            risk_amount = self.calculate_position_risk(
                entry_price, stop_loss, position_size, leverage
            )
            
            reward_distance = abs(take_profit - entry_price)
            reward_amount = reward_distance * position_size * leverage
            
            # Calculate ratios and percentages
            risk_reward_ratio = self.calculate_risk_reward_ratio(
                entry_price, stop_loss, take_profit
            )
            risk_percentage = (risk_amount / account_balance) * 100
            
            # Calculate max loss and gain
            max_loss = risk_amount
            max_gain = reward_amount
            
            # Calculate breakeven price
            breakeven_price = self.calculate_breakeven_price(
                entry_price, trade_direction=trade_direction
            )
            
            # Assess risk level
            risk_level = self.assess_risk_level(
                risk_percentage / 100, risk_reward_ratio, leverage
            )
            
            return RiskMetrics(
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_amount=risk_amount,
                reward_amount=reward_amount,
                risk_reward_ratio=risk_reward_ratio,
                risk_percentage=risk_percentage,
                max_loss=max_loss,
                max_gain=max_gain,
                breakeven_price=breakeven_price,
                risk_level=risk_level,
                trade_direction=trade_direction,
                leverage=leverage
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating comprehensive risk metrics: {e}")
            raise
    
    def validate_trade_risk(
        self,
        risk_metrics: RiskMetrics,
        account_balance: float
    ) -> Dict[str, Any]:
        """
        Validate trade risk against risk parameters.
        
        Args:
            risk_metrics: Risk metrics to validate
            account_balance: Current account balance
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            validation = {
                'is_valid': True,
                'violations': [],
                'warnings': [],
                'risk_score': 0
            }
            
            # Check account risk limit
            account_risk_pct = risk_metrics.risk_percentage / 100
            if account_risk_pct > self.risk_params.max_account_risk:
                validation['is_valid'] = False
                validation['violations'].append(
                    f"Account risk ({account_risk_pct:.2%}) exceeds limit "
                    f"({self.risk_params.max_account_risk:.2%})"
                )
            
            # Check risk-reward ratio
            if risk_metrics.risk_reward_ratio < self.risk_params.min_risk_reward_ratio:
                validation['warnings'].append(
                    f"Risk-reward ratio ({risk_metrics.risk_reward_ratio:.2f}) "
                    f"below recommended minimum ({self.risk_params.min_risk_reward_ratio:.2f})"
                )
            
            # Check leverage
            if risk_metrics.leverage > self.risk_params.max_leverage:
                validation['is_valid'] = False
                validation['violations'].append(
                    f"Leverage ({risk_metrics.leverage}) exceeds limit "
                    f"({self.risk_params.max_leverage})"
                )
            
            # Calculate overall risk score
            if risk_metrics.risk_level == RiskLevel.VERY_HIGH:
                validation['risk_score'] = 5
            elif risk_metrics.risk_level == RiskLevel.HIGH:
                validation['risk_score'] = 4
            elif risk_metrics.risk_level == RiskLevel.MEDIUM:
                validation['risk_score'] = 3
            elif risk_metrics.risk_level == RiskLevel.LOW:
                validation['risk_score'] = 2
            else:
                validation['risk_score'] = 1
            
            return validation
            
        except Exception as e:
            self.logger.error(f"❌ Error validating trade risk: {e}")
            return {
                'is_valid': False,
                'violations': [f"Validation error: {e}"],
                'warnings': [],
                'risk_score': 5
            }
    
    def calculate_portfolio_risk(
        self,
        positions: List[PositionRisk],
        account_balance: float
    ) -> Dict[str, Any]:
        """
        Calculate overall portfolio risk metrics.
        
        Args:
            positions: List of current positions
            account_balance: Total account balance
            
        Returns:
            Dict[str, Any]: Portfolio risk metrics
        """
        try:
            if not positions:
                return {
                    'total_risk': 0.0,
                    'risk_percentage': 0.0,
                    'position_count': 0,
                    'max_individual_risk': 0.0,
                    'diversification_score': 1.0,
                    'overall_risk_level': RiskLevel.VERY_LOW
                }
            
            # Calculate total portfolio risk
            total_risk = sum(pos.max_loss for pos in positions)
            risk_percentage = (total_risk / account_balance) * 100
            
            # Find maximum individual position risk
            max_individual_risk = max(pos.risk_percentage for pos in positions)
            
            # Simple diversification score (can be enhanced)
            unique_symbols = len(set(pos.symbol for pos in positions))
            diversification_score = min(1.0, unique_symbols / 5)  # Normalize to 5 assets
            
            # Assess overall portfolio risk level
            if risk_percentage > 15:
                overall_risk_level = RiskLevel.VERY_HIGH
            elif risk_percentage > 10:
                overall_risk_level = RiskLevel.HIGH
            elif risk_percentage > 5:
                overall_risk_level = RiskLevel.MEDIUM
            elif risk_percentage > 2:
                overall_risk_level = RiskLevel.LOW
            else:
                overall_risk_level = RiskLevel.VERY_LOW
            
            return {
                'total_risk': round(total_risk, 2),
                'risk_percentage': round(risk_percentage, 2),
                'position_count': len(positions),
                'max_individual_risk': round(max_individual_risk, 2),
                'diversification_score': round(diversification_score, 2),
                'overall_risk_level': overall_risk_level,
                'positions': positions
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating portfolio risk: {e}")
            raise