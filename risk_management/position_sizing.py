"""
Position Sizing Module
=====================

Advanced position sizing algorithms for risk-based trading.
Implements various position sizing methods including fixed percentage,
Kelly criterion, and volatility-based sizing.

This module provides:
- Multiple position sizing strategies
- Account balance and equity curve tracking
- Risk-based position calculations
- Support for different asset types and leverage
- Comprehensive position size validation

Author: dat-ns
Version: 1.0.0
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from datetime import datetime

from .risk_calculator import TradeDirection, RiskParameters


class SizingMethod(Enum):
    """Position sizing method enumeration."""
    FIXED_PERCENTAGE = "fixed_percentage"
    FIXED_AMOUNT = "fixed_amount"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_BASED = "volatility_based"
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"


class AssetType(Enum):
    """Asset type enumeration for position sizing."""
    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"
    FOREX = "forex"


@dataclass
class AccountInfo:
    """Account information for position sizing."""
    total_balance: float
    available_balance: float
    equity: float
    margin_used: float
    free_margin: float
    margin_ratio: float
    unrealized_pnl: float
    realized_pnl: float
    total_positions: int
    max_positions: int = 10
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate account information."""
        if self.total_balance <= 0:
            raise ValueError("Total balance must be positive")
        if self.available_balance < 0:
            raise ValueError("Available balance cannot be negative")


@dataclass
class PositionSize:
    """Position size calculation result."""
    symbol: str
    sizing_method: SizingMethod
    base_size: float  # Size in base asset
    quote_size: float  # Size in quote asset
    notional_value: float  # Total notional value
    leverage: int
    margin_required: float
    risk_amount: float
    max_loss_percentage: float
    position_percentage: float  # Percentage of total balance
    contracts: Optional[int] = None  # For futures/options
    lot_size: Optional[float] = None  # For forex
    
    # Validation flags
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate position size."""
        if self.base_size <= 0:
            self.is_valid = False
            self.errors.append("Base size must be positive")
        
        if self.leverage <= 0:
            self.is_valid = False
            self.errors.append("Leverage must be positive")
        
        if self.margin_required < 0:
            self.is_valid = False
            self.errors.append("Margin required cannot be negative")


class PositionSizer:
    """
    Advanced position sizing calculator.
    
    Implements multiple position sizing strategies:
    - Fixed percentage of account
    - Fixed dollar amount
    - Kelly criterion
    - Volatility-based sizing
    - Equal weight allocation
    - Risk parity allocation
    """
    
    def __init__(self, risk_params: Optional[RiskParameters] = None):
        """
        Initialize position sizer.
        
        Args:
            risk_params: Risk parameters for calculations
        """
        self.risk_params = risk_params or RiskParameters()
        self.logger = logging.getLogger(__name__)
        
        # Position sizing parameters
        self.min_position_size = 10.0  # Minimum position size in USD
        self.max_position_percentage = 0.25  # Maximum 25% per position
        self.decimal_places = 8
        
        # Asset-specific parameters
        self.asset_multipliers = {
            AssetType.SPOT: 1.0,
            AssetType.FUTURES: 1.0,
            AssetType.OPTIONS: 0.1,  # More conservative for options
            AssetType.FOREX: 1.0
        }
        
        self.logger.info("📏 Position Sizer initialized")
    
    def calculate_fixed_percentage_size(
        self,
        account_info: AccountInfo,
        entry_price: float,
        stop_loss: float,
        risk_percentage: float,
        trade_direction: TradeDirection,
        leverage: int = 1,
        symbol: str = "BTCUSDT"
    ) -> PositionSize:
        """
        Calculate position size using fixed percentage method.
        
        This is the most common method (2% rule) where you risk
        a fixed percentage of your account balance on each trade.
        
        Args:
            account_info: Account information
            entry_price: Trade entry price
            stop_loss: Stop loss price
            risk_percentage: Risk percentage (e.g., 0.02 for 2%)
            trade_direction: Long or short
            leverage: Trading leverage
            symbol: Trading symbol
            
        Returns:
            PositionSize: Calculated position size
        """
        try:
            # Calculate risk amount
            risk_amount = account_info.available_balance * risk_percentage
            
            # Calculate price distance to stop loss
            price_distance = abs(entry_price - stop_loss)
            if price_distance == 0:
                raise ValueError("Stop loss cannot equal entry price")
            
            # Calculate position size
            base_size = risk_amount / (price_distance * leverage)
            quote_size = base_size * entry_price
            notional_value = quote_size * leverage
            margin_required = notional_value / leverage
            
            # Calculate position percentage
            position_percentage = (margin_required / account_info.total_balance) * 100
            
            # Create position size object
            position_size = PositionSize(
                symbol=symbol,
                sizing_method=SizingMethod.FIXED_PERCENTAGE,
                base_size=round(base_size, self.decimal_places),
                quote_size=round(quote_size, 2),
                notional_value=round(notional_value, 2),
                leverage=leverage,
                margin_required=round(margin_required, 2),
                risk_amount=round(risk_amount, 2),
                max_loss_percentage=risk_percentage * 100,
                position_percentage=round(position_percentage, 2)
            )
            
            # Validate position size
            self._validate_position_size(position_size, account_info)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating fixed percentage size: {e}")
            raise
    
    def calculate_fixed_amount_size(
        self,
        account_info: AccountInfo,
        entry_price: float,
        risk_amount: float,
        stop_loss: float,
        leverage: int = 1,
        symbol: str = "BTCUSDT"
    ) -> PositionSize:
        """
        Calculate position size using fixed dollar amount method.
        
        Args:
            account_info: Account information
            entry_price: Trade entry price
            risk_amount: Fixed risk amount in USD
            stop_loss: Stop loss price
            leverage: Trading leverage
            symbol: Trading symbol
            
        Returns:
            PositionSize: Calculated position size
        """
        try:
            # Validate risk amount
            if risk_amount > account_info.available_balance:
                raise ValueError("Risk amount exceeds available balance")
            
            # Calculate price distance
            price_distance = abs(entry_price - stop_loss)
            if price_distance == 0:
                raise ValueError("Stop loss cannot equal entry price")
            
            # Calculate position size
            base_size = risk_amount / (price_distance * leverage)
            quote_size = base_size * entry_price
            notional_value = quote_size * leverage
            margin_required = notional_value / leverage
            
            # Calculate percentages
            risk_percentage = (risk_amount / account_info.total_balance) * 100
            position_percentage = (margin_required / account_info.total_balance) * 100
            
            position_size = PositionSize(
                symbol=symbol,
                sizing_method=SizingMethod.FIXED_AMOUNT,
                base_size=round(base_size, self.decimal_places),
                quote_size=round(quote_size, 2),
                notional_value=round(notional_value, 2),
                leverage=leverage,
                margin_required=round(margin_required, 2),
                risk_amount=round(risk_amount, 2),
                max_loss_percentage=round(risk_percentage, 2),
                position_percentage=round(position_percentage, 2)
            )
            
            self._validate_position_size(position_size, account_info)
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating fixed amount size: {e}")
            raise
    
    def calculate_kelly_criterion_size(
        self,
        account_info: AccountInfo,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        win_probability: float,
        leverage: int = 1,
        symbol: str = "BTCUSDT"
    ) -> PositionSize:
        """
        Calculate position size using Kelly criterion.
        
        Kelly formula: f = (bp - q) / b
        Where:
        f = fraction of capital to wager
        b = odds received on the wager
        p = probability of winning
        q = probability of losing = 1 - p
        
        Args:
            account_info: Account information
            entry_price: Trade entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            win_probability: Probability of winning (0.0 to 1.0)
            leverage: Trading leverage
            symbol: Trading symbol
            
        Returns:
            PositionSize: Calculated position size
        """
        try:
            # Validate probability
            if not 0 < win_probability < 1:
                raise ValueError("Win probability must be between 0 and 1")
            
            # Calculate win/loss ratios
            risk_distance = abs(entry_price - stop_loss)
            reward_distance = abs(take_profit - entry_price)
            
            if risk_distance == 0 or reward_distance == 0:
                raise ValueError("Invalid price levels for Kelly calculation")
            
            # Kelly calculation
            win_loss_ratio = reward_distance / risk_distance  # b in Kelly formula
            lose_probability = 1 - win_probability  # q in Kelly formula
            
            # Kelly fraction: f = (bp - q) / b
            kelly_fraction = ((win_loss_ratio * win_probability) - lose_probability) / win_loss_ratio
            
            # Apply Kelly fraction limits (never risk more than 25% using Kelly)
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
            
            # Apply fractional Kelly (often use 25% of Kelly to reduce volatility)
            conservative_kelly = kelly_fraction * 0.25
            
            # Calculate position size based on Kelly fraction
            risk_amount = account_info.available_balance * conservative_kelly
            base_size = risk_amount / (risk_distance * leverage)
            quote_size = base_size * entry_price
            notional_value = quote_size * leverage
            margin_required = notional_value / leverage
            
            # Calculate percentages
            risk_percentage = (risk_amount / account_info.total_balance) * 100
            position_percentage = (margin_required / account_info.total_balance) * 100
            
            position_size = PositionSize(
                symbol=symbol,
                sizing_method=SizingMethod.KELLY_CRITERION,
                base_size=round(base_size, self.decimal_places),
                quote_size=round(quote_size, 2),
                notional_value=round(notional_value, 2),
                leverage=leverage,
                margin_required=round(margin_required, 2),
                risk_amount=round(risk_amount, 2),
                max_loss_percentage=round(risk_percentage, 2),
                position_percentage=round(position_percentage, 2)
            )
            
            # Add Kelly-specific warnings
            if kelly_fraction > 0.1:
                position_size.warnings.append(
                    f"Kelly suggests high allocation ({kelly_fraction:.2%}), using conservative fraction"
                )
            
            self._validate_position_size(position_size, account_info)
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating Kelly criterion size: {e}")
            raise
    
    def calculate_volatility_based_size(
        self,
        account_info: AccountInfo,
        entry_price: float,
        volatility: float,  # ATR or volatility measure
        risk_percentage: float,
        volatility_multiplier: float = 2.0,
        leverage: int = 1,
        symbol: str = "BTCUSDT"
    ) -> PositionSize:
        """
        Calculate position size based on market volatility.
        
        Uses volatility (like ATR) to determine appropriate position size.
        Higher volatility = smaller position size.
        
        Args:
            account_info: Account information
            entry_price: Trade entry price
            volatility: Market volatility measure (e.g., ATR)
            risk_percentage: Risk percentage
            volatility_multiplier: Multiplier for volatility-based stop
            leverage: Trading leverage
            symbol: Trading symbol
            
        Returns:
            PositionSize: Calculated position size
        """
        try:
            if volatility <= 0:
                raise ValueError("Volatility must be positive")
            
            # Calculate volatility-based stop distance
            stop_distance = volatility * volatility_multiplier
            
            # Calculate risk amount
            risk_amount = account_info.available_balance * risk_percentage
            
            # Calculate position size
            base_size = risk_amount / (stop_distance * leverage)
            quote_size = base_size * entry_price
            notional_value = quote_size * leverage
            margin_required = notional_value / leverage
            
            # Calculate percentages
            position_percentage = (margin_required / account_info.total_balance) * 100
            
            position_size = PositionSize(
                symbol=symbol,
                sizing_method=SizingMethod.VOLATILITY_BASED,
                base_size=round(base_size, self.decimal_places),
                quote_size=round(quote_size, 2),
                notional_value=round(notional_value, 2),
                leverage=leverage,
                margin_required=round(margin_required, 2),
                risk_amount=round(risk_amount, 2),
                max_loss_percentage=risk_percentage * 100,
                position_percentage=round(position_percentage, 2)
            )
            
            self._validate_position_size(position_size, account_info)
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating volatility-based size: {e}")
            raise
    
    def calculate_equal_weight_size(
        self,
        account_info: AccountInfo,
        entry_price: float,
        num_positions: int,
        allocation_percentage: float = 0.9,  # Use 90% of available balance
        leverage: int = 1,
        symbol: str = "BTCUSDT"
    ) -> PositionSize:
        """
        Calculate equal-weighted position size.
        
        Divides available capital equally among all positions.
        
        Args:
            account_info: Account information
            entry_price: Trade entry price
            num_positions: Total number of positions
            allocation_percentage: Percentage of balance to allocate
            leverage: Trading leverage
            symbol: Trading symbol
            
        Returns:
            PositionSize: Calculated position size
        """
        try:
            if num_positions <= 0:
                raise ValueError("Number of positions must be positive")
            
            # Calculate allocation per position
            total_allocation = account_info.available_balance * allocation_percentage
            position_allocation = total_allocation / num_positions
            
            # Calculate position size
            margin_required = position_allocation
            notional_value = margin_required * leverage
            base_size = notional_value / entry_price
            quote_size = base_size * entry_price
            
            # Estimate risk (assume 2% stop loss for equal weight)
            estimated_risk = position_allocation * 0.02
            
            # Calculate percentages
            position_percentage = (margin_required / account_info.total_balance) * 100
            risk_percentage = (estimated_risk / account_info.total_balance) * 100
            
            position_size = PositionSize(
                symbol=symbol,
                sizing_method=SizingMethod.EQUAL_WEIGHT,
                base_size=round(base_size, self.decimal_places),
                quote_size=round(quote_size, 2),
                notional_value=round(notional_value, 2),
                leverage=leverage,
                margin_required=round(margin_required, 2),
                risk_amount=round(estimated_risk, 2),
                max_loss_percentage=round(risk_percentage, 2),
                position_percentage=round(position_percentage, 2)
            )
            
            self._validate_position_size(position_size, account_info)
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating equal weight size: {e}")
            raise
    
    def calculate_futures_position_size(
        self,
        account_info: AccountInfo,
        entry_price: float,
        stop_loss: float,
        risk_percentage: float,
        contract_size: float,
        tick_size: float,
        leverage: int = 1,
        symbol: str = "BTCUSDT"
    ) -> PositionSize:
        """
        Calculate position size for futures contracts.
        
        Handles contract-based sizing with proper rounding to tick sizes.
        
        Args:
            account_info: Account information
            entry_price: Trade entry price
            stop_loss: Stop loss price
            risk_percentage: Risk percentage
            contract_size: Size of one contract
            tick_size: Minimum price movement
            leverage: Trading leverage
            symbol: Trading symbol
            
        Returns:
            PositionSize: Calculated position size
        """
        try:
            # Calculate ideal position size using fixed percentage method
            base_position = self.calculate_fixed_percentage_size(
                account_info, entry_price, stop_loss, risk_percentage,
                TradeDirection.LONG, leverage, symbol
            )
            
            # Convert to number of contracts
            contracts = round(base_position.base_size / contract_size)
            contracts = max(1, contracts)  # Minimum 1 contract
            
            # Recalculate actual position size based on contracts
            actual_base_size = contracts * contract_size
            actual_quote_size = actual_base_size * entry_price
            actual_notional = actual_quote_size * leverage
            actual_margin = actual_notional / leverage
            
            # Recalculate risk based on actual position size
            price_distance = abs(entry_price - stop_loss)
            actual_risk = price_distance * actual_base_size * leverage
            
            # Calculate percentages
            position_percentage = (actual_margin / account_info.total_balance) * 100
            risk_percentage_actual = (actual_risk / account_info.total_balance) * 100
            
            position_size = PositionSize(
                symbol=symbol,
                sizing_method=SizingMethod.FIXED_PERCENTAGE,
                base_size=round(actual_base_size, self.decimal_places),
                quote_size=round(actual_quote_size, 2),
                notional_value=round(actual_notional, 2),
                leverage=leverage,
                margin_required=round(actual_margin, 2),
                risk_amount=round(actual_risk, 2),
                max_loss_percentage=round(risk_percentage_actual, 2),
                position_percentage=round(position_percentage, 2),
                contracts=contracts
            )
            
            # Add futures-specific warnings
            if abs(actual_risk - base_position.risk_amount) > base_position.risk_amount * 0.1:
                position_size.warnings.append(
                    f"Risk adjusted due to contract sizing: "
                    f"${actual_risk:.2f} vs target ${base_position.risk_amount:.2f}"
                )
            
            self._validate_position_size(position_size, account_info)
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating futures position size: {e}")
            raise
    
    def optimize_position_size_for_drawdown(
        self,
        base_position_size: PositionSize,
        current_drawdown: float,
        max_drawdown: float = 0.20,
        scaling_factor: float = 0.5
    ) -> PositionSize:
        """
        Optimize position size based on current drawdown.
        
        Reduces position size as drawdown increases to preserve capital.
        
        Args:
            base_position_size: Base position size calculation
            current_drawdown: Current portfolio drawdown (0.0 to 1.0)
            max_drawdown: Maximum acceptable drawdown
            scaling_factor: How aggressively to scale down
            
        Returns:
            PositionSize: Adjusted position size
        """
        try:
            if current_drawdown < 0:
                current_drawdown = 0
            
            # Calculate drawdown scaling factor
            if current_drawdown >= max_drawdown:
                # Stop trading if max drawdown reached
                scale_factor = 0.1  # Reduce to 10% of normal size
            else:
                # Linear scaling based on drawdown
                drawdown_ratio = current_drawdown / max_drawdown
                scale_factor = 1.0 - (drawdown_ratio * scaling_factor)
                scale_factor = max(0.1, scale_factor)  # Minimum 10% size
            
            # Apply scaling to position size
            adjusted_position = PositionSize(
                symbol=base_position_size.symbol,
                sizing_method=base_position_size.sizing_method,
                base_size=base_position_size.base_size * scale_factor,
                quote_size=base_position_size.quote_size * scale_factor,
                notional_value=base_position_size.notional_value * scale_factor,
                leverage=base_position_size.leverage,
                margin_required=base_position_size.margin_required * scale_factor,
                risk_amount=base_position_size.risk_amount * scale_factor,
                max_loss_percentage=base_position_size.max_loss_percentage * scale_factor,
                position_percentage=base_position_size.position_percentage * scale_factor,
                contracts=int(base_position_size.contracts * scale_factor) if base_position_size.contracts else None,
                lot_size=base_position_size.lot_size
            )
            
            # Add drawdown adjustment warning
            if scale_factor < 1.0:
                adjusted_position.warnings.append(
                    f"Position size reduced by {(1-scale_factor)*100:.1f}% due to "
                    f"drawdown ({current_drawdown*100:.1f}%)"
                )
            
            return adjusted_position
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing position size for drawdown: {e}")
            raise
    
    def calculate_multi_asset_allocation(
        self,
        account_info: AccountInfo,
        assets: List[Dict[str, Any]],
        allocation_method: SizingMethod = SizingMethod.EQUAL_WEIGHT,
        total_allocation: float = 0.9
    ) -> Dict[str, PositionSize]:
        """
        Calculate position sizes for multiple assets.
        
        Args:
            account_info: Account information
            assets: List of asset dictionaries with pricing info
            allocation_method: Method to use for allocation
            total_allocation: Total percentage of balance to allocate
            
        Returns:
            Dict[str, PositionSize]: Position sizes by symbol
        """
        try:
            position_sizes = {}
            
            if allocation_method == SizingMethod.EQUAL_WEIGHT:
                # Equal weight allocation
                allocation_per_asset = total_allocation / len(assets)
                
                for asset in assets:
                    symbol = asset['symbol']
                    entry_price = asset['entry_price']
                    
                    # Calculate equal weight position
                    margin_per_asset = account_info.available_balance * allocation_per_asset
                    leverage = asset.get('leverage', 1)
                    notional_value = margin_per_asset * leverage
                    base_size = notional_value / entry_price
                    
                    position_sizes[symbol] = PositionSize(
                        symbol=symbol,
                        sizing_method=SizingMethod.EQUAL_WEIGHT,
                        base_size=round(base_size, self.decimal_places),
                        quote_size=round(base_size * entry_price, 2),
                        notional_value=round(notional_value, 2),
                        leverage=leverage,
                        margin_required=round(margin_per_asset, 2),
                        risk_amount=round(margin_per_asset * 0.02, 2),  # Assume 2% risk
                        max_loss_percentage=2.0,
                        position_percentage=round(allocation_per_asset * 100, 2)
                    )
            
            elif allocation_method == SizingMethod.RISK_PARITY:
                # Risk parity allocation (equal risk per asset)
                total_risk_budget = account_info.available_balance * total_allocation * 0.02
                risk_per_asset = total_risk_budget / len(assets)
                
                for asset in assets:
                    symbol = asset['symbol']
                    entry_price = asset['entry_price']
                    volatility = asset.get('volatility', entry_price * 0.02)  # Default 2% volatility
                    leverage = asset.get('leverage', 1)
                    
                    # Size based on equal risk allocation
                    base_size = risk_per_asset / (volatility * leverage)
                    quote_size = base_size * entry_price
                    notional_value = quote_size * leverage
                    margin_required = notional_value / leverage
                    
                    position_sizes[symbol] = PositionSize(
                        symbol=symbol,
                        sizing_method=SizingMethod.RISK_PARITY,
                        base_size=round(base_size, self.decimal_places),
                        quote_size=round(quote_size, 2),
                        notional_value=round(notional_value, 2),
                        leverage=leverage,
                        margin_required=round(margin_required, 2),
                        risk_amount=round(risk_per_asset, 2),
                        max_loss_percentage=round(risk_per_asset / account_info.total_balance * 100, 2),
                        position_percentage=round(margin_required / account_info.total_balance * 100, 2)
                    )
            
            return position_sizes
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating multi-asset allocation: {e}")
            raise
    
    def _validate_position_size(self, position_size: PositionSize, account_info: AccountInfo):
        """
        Validate calculated position size against account limits and risk parameters.
        
        Args:
            position_size: Position size to validate
            account_info: Account information
        """
        try:
            # Check minimum position size
            if position_size.quote_size < self.min_position_size:
                position_size.warnings.append(
                    f"Position size (${position_size.quote_size:.2f}) below minimum "
                    f"(${self.min_position_size:.2f})"
                )
            
            # Check maximum position percentage
            max_position_pct = self.max_position_percentage * 100
            if position_size.position_percentage > max_position_pct:
                position_size.is_valid = False
                position_size.errors.append(
                    f"Position percentage ({position_size.position_percentage:.2f}%) "
                    f"exceeds maximum ({max_position_pct:.2f}%)"
                )
            
            # Check available balance
            if position_size.margin_required > account_info.available_balance:
                position_size.is_valid = False
                position_size.errors.append(
                    f"Required margin (${position_size.margin_required:.2f}) "
                    f"exceeds available balance (${account_info.available_balance:.2f})"
                )
            
            # Check risk percentage
            max_risk_pct = self.risk_params.max_account_risk * 100
            if position_size.max_loss_percentage > max_risk_pct:
                position_size.warnings.append(
                    f"Risk percentage ({position_size.max_loss_percentage:.2f}%) "
                    f"exceeds recommended maximum ({max_risk_pct:.2f}%)"
                )
            
            # Check leverage limits
            if position_size.leverage > self.risk_params.max_leverage:
                position_size.is_valid = False
                position_size.errors.append(
                    f"Leverage ({position_size.leverage}) exceeds maximum "
                    f"({self.risk_params.max_leverage})"
                )
            
        except Exception as e:
            self.logger.error(f"❌ Error validating position size: {e}")
            position_size.is_valid = False
            position_size.errors.append(f"Validation error: {e}")
    
    def get_sizing_recommendations(
        self,
        account_info: AccountInfo,
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get position sizing recommendations based on account and market conditions.
        
        Args:
            account_info: Account information
            market_conditions: Current market conditions
            
        Returns:
            Dict[str, Any]: Sizing recommendations
        """
        try:
            recommendations = {
                'recommended_risk_per_trade': 0.02,  # Default 2%
                'max_positions': 5,
                'preferred_leverage': 1,
                'sizing_method': SizingMethod.FIXED_PERCENTAGE,
                'warnings': [],
                'adjustments': []
            }
            
            # Adjust based on account size
            if account_info.total_balance < 1000:
                recommendations['warnings'].append("Small account size - consider lower risk per trade")
                recommendations['recommended_risk_per_trade'] = 0.01  # 1% for small accounts
            elif account_info.total_balance > 100000:
                recommendations['max_positions'] = 10  # More positions for larger accounts
            
            # Adjust based on current drawdown
            current_drawdown = market_conditions.get('portfolio_drawdown', 0)
            if current_drawdown > 0.1:  # 10% drawdown
                recommendations['recommended_risk_per_trade'] = 0.01
                recommendations['adjustments'].append("Reduced risk due to drawdown")
            
            # Adjust based on market volatility
            market_volatility = market_conditions.get('market_volatility', 'normal')
            if market_volatility == 'high':
                recommendations['preferred_leverage'] = 1
                recommendations['sizing_method'] = SizingMethod.VOLATILITY_BASED
                recommendations['adjustments'].append("Using volatility-based sizing due to high market volatility")
            
            # Adjust based on win rate if available
            recent_win_rate = market_conditions.get('recent_win_rate')
            if recent_win_rate is not None:
                if recent_win_rate > 0.6:
                    recommendations['sizing_method'] = SizingMethod.KELLY_CRITERION
                    recommendations['adjustments'].append("Consider Kelly criterion due to high win rate")
                elif recent_win_rate < 0.4:
                    recommendations['recommended_risk_per_trade'] = 0.01
                    recommendations['adjustments'].append("Reduced risk due to low recent win rate")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error getting sizing recommendations: {e}")
            return {
                'recommended_risk_per_trade': 0.01,
                'max_positions': 3,
                'preferred_leverage': 1,
                'sizing_method': SizingMethod.FIXED_PERCENTAGE,
                'warnings': [f"Error in recommendations: {e}"],
                'adjustments': []
            }