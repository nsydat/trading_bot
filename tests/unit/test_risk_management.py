"""
Unit Tests for Risk Management
==============================

Comprehensive test suite for risk management components including:
- Risk calculator tests
- Position sizing tests
- Portfolio risk metrics
- Stop loss and take profit calculations
- Risk validation and edge cases

Author: dat-ns
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime, timedelta

# Import risk management components
from risk_management.risk_calculator import (
    RiskCalculator, 
    RiskMetrics, 
    RiskParameters, 
    PositionRisk,
    TradeRisk
)
from risk_management.position_sizing import (
    PositionSizer,
    PositionSize,
    SizingMethod,
    AccountInfo
)
from core.models.market_data import OHLCV, PriceData
from core.utils.exceptions import RiskManagementError


class TestRiskParameters:
    """Test cases for RiskParameters data class."""
    
    def test_risk_parameters_creation(self):
        """Test RiskParameters object creation and validation."""
        params = RiskParameters(
            max_risk_per_trade=0.02,
            max_portfolio_risk=0.10,
            max_correlation=0.7,
            max_positions=5,
            stop_loss_pct=0.03,
            take_profit_ratio=2.0
        )
        
        assert params.max_risk_per_trade == 0.02
        assert params.max_portfolio_risk == 0.10
        assert params.max_correlation == 0.7
        assert params.max_positions == 5
        assert params.stop_loss_pct == 0.03
        assert params.take_profit_ratio == 2.0
    
    def test_risk_parameters_validation(self):
        """Test RiskParameters validation rules."""
        # Test invalid risk percentages
        with pytest.raises(ValueError):
            RiskParameters(max_risk_per_trade=1.5)  # > 100%
        
        with pytest.raises(ValueError):
            RiskParameters(max_risk_per_trade=-0.01)  # Negative
        
        # Test invalid ratios
        with pytest.raises(ValueError):
            RiskParameters(take_profit_ratio=0.5)  # Less than 1.0
        
        # Test invalid positions
        with pytest.raises(ValueError):
            RiskParameters(max_positions=0)  # Must be positive
    
    def test_risk_parameters_defaults(self):
        """Test RiskParameters with default values."""
        params = RiskParameters()
        
        # Should have reasonable defaults
        assert 0 < params.max_risk_per_trade <= 0.05  # Max 5% per trade
        assert 0 < params.max_portfolio_risk <= 0.20  # Max 20% portfolio risk
        assert 0 < params.max_correlation < 1.0
        assert params.max_positions > 0
        assert params.take_profit_ratio >= 1.0


class TestPositionRisk:
    """Test cases for PositionRisk calculations."""
    
    def setup_method(self):
        """Setup test environment."""
        self.sample_position = {
            'symbol': 'BTCUSDT',
            'side': 'LONG',
            'size': Decimal('0.1'),
            'entry_price': Decimal('50000.00'),
            'current_price': Decimal('51000.00'),
            'stop_loss': Decimal('48500.00'),
            'take_profit': Decimal('53000.00')
        }
    
    def test_position_risk_calculation(self):
        """Test basic position risk calculation."""
        risk_calc = RiskCalculator()
        
        position_risk = risk_calc.calculate_position_risk(
            entry_price=self.sample_position['entry_price'],
            current_price=self.sample_position['current_price'],
            stop_loss=self.sample_position['stop_loss'],
            position_size=self.sample_position['size']
        )
        
        assert isinstance(position_risk, PositionRisk)
        assert position_risk.max_loss > 0  # Should have some risk
        assert position_risk.unrealized_pnl != 0  # Should have some P&L
        assert position_risk.risk_reward_ratio > 0
    
    def test_long_position_risk(self):
        """Test risk calculation for long positions."""
        risk_calc = RiskCalculator()
        
        position_risk = risk_calc.calculate_position_risk(
            entry_price=Decimal('50000'),
            current_price=Decimal('51000'),
            stop_loss=Decimal('48500'),
            position_size=Decimal('0.1'),
            side='LONG'
        )
        
        # Long position: profit when price goes up
        assert position_risk.unrealized_pnl > 0  # Currently in profit
        
        # Max loss should be entry - stop_loss
        expected_max_loss = (50000 - 48500) * 0.1
        assert abs(float(position_risk.max_loss) - expected_max_loss) < 1.0
    
    def test_short_position_risk(self):
        """Test risk calculation for short positions."""
        risk_calc = RiskCalculator()
        
        position_risk = risk_calc.calculate_position_risk(
            entry_price=Decimal('50000'),
            current_price=Decimal('49000'),  # Price went down
            stop_loss=Decimal('51500'),      # Stop loss above entry
            position_size=Decimal('0.1'),
            side='SHORT'
        )
        
        # Short position: profit when price goes down
        assert position_risk.unrealized_pnl > 0  # Currently in profit
        
        # Max loss should be stop_loss - entry
        expected_max_loss = (51500 - 50000) * 0.1
        assert abs(float(position_risk.max_loss) - expected_max_loss) < 1.0
    
    def test_position_risk_edge_cases(self):
        """Test position risk edge cases."""
        risk_calc = RiskCalculator()
        
        # Test position at break-even
        risk_breakeven = risk_calc.calculate_position_risk(
            entry_price=Decimal('50000'),
            current_price=Decimal('50000'),
            stop_loss=Decimal('48500'),
            position_size=Decimal('0.1')
        )
        assert risk_breakeven.unrealized_pnl == 0
        
        # Test very small position
        risk_small = risk_calc.calculate_position_risk(
            entry_price=Decimal('50000'),
            current_price=Decimal('51000'),
            stop_loss=Decimal('49000'),
            position_size=Decimal('0.001')
        )
        assert risk_small.max_loss < 100  # Should be small
        
        # Test position without stop loss
        with pytest.raises(ValueError):
            risk_calc.calculate_position_risk(
                entry_price=Decimal('50000'),
                current_price=Decimal('51000'),
                stop_loss=None,
                position_size=Decimal('0.1')
            )


class TestTradeRisk:
    """Test cases for TradeRisk calculations."""
    
    def test_trade_risk_calculation(self):
        """Test basic trade risk calculation before entry."""
        risk_calc = RiskCalculator()
        
        trade_risk = risk_calc.calculate_trade_risk(
            entry_price=Decimal('50000'),
            stop_loss=Decimal('48500'),
            take_profit=Decimal('53000'),
            position_size=Decimal('0.1'),
            account_balance=10000.0
        )
        
        assert isinstance(trade_risk, TradeRisk)
        assert trade_risk.risk_amount > 0
        assert trade_risk.reward_amount > 0
        assert trade_risk.risk_reward_ratio > 0
        assert 0 <= trade_risk.risk_percentage <= 1.0
    
    def test_trade_risk_reward_calculation(self):
        """Test risk-reward ratio calculation."""
        risk_calc = RiskCalculator()
        
        # Good risk-reward trade (1:2 ratio)
        trade_risk = risk_calc.calculate_trade_risk(
            entry_price=Decimal('50000'),
            stop_loss=Decimal('48000'),   # 2000 risk
            take_profit=Decimal('54000'), # 4000 reward
            position_size=Decimal('0.1')
        )
        
        # Should be approximately 2:1 reward:risk
        assert 1.8 <= trade_risk.risk_reward_ratio <= 2.2
    
    def test_trade_risk_percentage_calculation(self):
        """Test risk percentage calculation."""
        risk_calc = RiskCalculator()
        
        trade_risk = risk_calc.calculate_trade_risk(
            entry_price=Decimal('50000'),
            stop_loss=Decimal('49000'),  # 1000 risk per unit
            position_size=Decimal('0.1'), # 100 total risk
            account_balance=10000.0
        )
        
        # Risk should be 1% of account (100 / 10000)
        assert abs(trade_risk.risk_percentage - 0.01) < 0.001
    
    def test_trade_risk_validation(self):
        """Test trade risk validation rules."""
        risk_calc = RiskCalculator()
        
        # Test invalid stop loss (above entry for long)
        with pytest.raises(ValueError):
            risk_calc.calculate_trade_risk(
                entry_price=Decimal('50000'),
                stop_loss=Decimal('51000'),  # Above entry for long
                position_size=Decimal('0.1'),
                side='LONG'
            )
        
        # Test invalid take profit (below entry for long)
        with pytest.raises(ValueError):
            risk_calc.calculate_trade_risk(
                entry_price=Decimal('50000'),
                stop_loss=Decimal('48000'),
                take_profit=Decimal('49000'),  # Below entry for long
                position_size=Decimal('0.1'),
                side='LONG'
            )


class TestRiskMetrics:
    """Test cases for portfolio risk metrics."""
    
    def setup_method(self):
        """Setup test portfolio."""
        self.sample_portfolio = [
            {
                'symbol': 'BTCUSDT',
                'size': Decimal('0.1'),
                'entry_price': Decimal('50000'),
                'current_price': Decimal('51000'),
                'unrealized_pnl': 100.0
            },
            {
                'symbol': 'ETHUSDT', 
                'size': Decimal('2.0'),
                'entry_price': Decimal('3000'),
                'current_price': Decimal('3150'),
                'unrealized_pnl': 300.0
            }
        ]
    
    def test_portfolio_risk_metrics(self):
        """Test portfolio-level risk metrics calculation."""
        risk_calc = RiskCalculator()
        
        metrics = risk_calc.calculate_portfolio_risk(
            positions=self.sample_portfolio,
            account_balance=10000.0
        )
        
        assert isinstance(metrics, RiskMetrics)
        assert metrics.total_exposure > 0
        assert metrics.portfolio_risk_percentage >= 0
        assert len(metrics.position_risks) == len(self.sample_portfolio)
    
    def test_portfolio_correlation_analysis(self):
        """Test portfolio correlation analysis."""
        if hasattr(RiskCalculator, 'calculate_portfolio_correlation'):
            risk_calc = RiskCalculator()
            
            # Mock price data for correlation analysis
            btc_prices = [50000, 51000, 49500, 52000, 50500]
            eth_prices = [3000, 3100, 2950, 3200, 3050]
            
            correlation = risk_calc.calculate_portfolio_correlation([
                {'symbol': 'BTCUSDT', 'prices': btc_prices},
                {'symbol': 'ETHUSDT', 'prices': eth_prices}
            ])
            
            assert -1.0 <= correlation <= 1.0
    
    def test_value_at_risk_calculation(self):
        """Test Value at Risk (VaR) calculation."""
        if hasattr(RiskCalculator, 'calculate_value_at_risk'):
            risk_calc = RiskCalculator()
            
            # Historical returns data
            returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.025]
            
            var_95 = risk_calc.calculate_value_at_risk(
                returns=returns,
                confidence_level=0.95,
                portfolio_value=10000.0
            )
            
            assert var_95 < 0  # VaR should be negative (loss)
            assert abs(var_95) < 10000  # Should be less than portfolio value
    
    def test_maximum_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        if hasattr(RiskCalculator, 'calculate_max_drawdown'):
            risk_calc = RiskCalculator()
            
            # Portfolio value over time
            portfolio_values = [10000, 10200, 9800, 9500, 9700, 10100, 10300]
            
            max_drawdown = risk_calc.calculate_max_drawdown(portfolio_values)
            
            # Expected drawdown: (10200 - 9500) / 10200 = 6.86%
            assert max_drawdown > 0  # Drawdown is positive percentage
            assert max_drawdown < 1.0  # Less than 100%


class TestPositionSizer:
    """Test cases for PositionSizer class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.account_info = AccountInfo(
            balance=10000.0,
            equity=10000.0,
            margin_used=0.0,
            free_margin=10000.0,
            currency='USDT'
        )
        
        self.position_sizer = PositionSizer()
    
    def test_position_sizer_initialization(self):
        """Test PositionSizer initialization."""
        assert isinstance(self.position_sizer, PositionSizer)
        # Test default sizing method if applicable
    
    def test_fixed_amount_sizing(self):
        """Test fixed amount position sizing."""
        position_size = self.position_sizer.calculate_size(
            method=SizingMethod.FIXED_AMOUNT,
            account_info=self.account_info,
            entry_price=Decimal('50000'),
            stop_loss=Decimal('48500'),
            fixed_amount=1000.0  # $1000 position
        )
        
        assert isinstance(position_size, PositionSize)
        # Position size should be 1000 / 50000 = 0.02 BTC
        expected_size = 1000.0 / 50000.0
        assert abs(float(position_size.quantity) - expected_size) < 0.001
        assert position_size.method == SizingMethod.FIXED_AMOUNT
    
    def test_fixed_percentage_sizing(self):
        """Test fixed percentage position sizing.""" 
        position_size = self.position_sizer.calculate_size(
            method=SizingMethod.FIXED_PERCENTAGE,
            account_info=self.account_info,
            entry_price=Decimal('50000'),
            percentage=0.1  # 10% of account
        )
        
        # Should use 10% of $10,000 = $1,000
        expected_value = 1000.0
        actual_value = float(position_size.quantity) * 50000.0
        assert abs(actual_value - expected_value) < 1.0
    
    def test_risk_based_sizing(self):
        """Test risk-based position sizing."""
        position_size = self.position_sizer.calculate_size(
            method=SizingMethod.RISK_BASED,
            account_info=self.account_info,
            entry_price=Decimal('50000'),
            stop_loss=Decimal('48000'),  # $2000 risk per unit
            risk_percentage=0.02  # Risk 2% of account
        )
        
        # Risk amount = $10,000 * 0.02 = $200
        # Position size = $200 / $2000 = 0.1 BTC
        expected_size = 200.0 / 2000.0
        assert abs(float(position_size.quantity) - expected_size) < 0.001
        assert position_size.risk_amount <= 200.0
    
    def test_kelly_criterion_sizing(self):
        """Test Kelly Criterion position sizing."""
        if SizingMethod.KELLY_CRITERION in SizingMethod.__members__.values():
            position_size = self.position_sizer.calculate_size(
                method=SizingMethod.KELLY_CRITERION,
                account_info=self.account_info,
                entry_price=Decimal('50000'),
                win_probability=0.6,
                avg_win=1000.0,
                avg_loss=500.0
            )
            
            # Kelly % = (bp - q) / b where b=odds, p=win prob, q=loss prob
            # Should result in reasonable position size
            assert position_size.quantity > 0
            assert float(position_size.quantity) * 50000.0 <= 10000.0  # Not exceed account
    
    def test_position_size_validation(self):
        """Test position size validation rules."""