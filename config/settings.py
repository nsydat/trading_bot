"""
Configuration Settings
=====================

Central configuration management for the trading bot.
Handles environment variables, validation, and default values.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

from core.utils.exceptions import ConfigurationError


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///data/trading_bot.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class BinanceConfig:
    """Binance API configuration settings."""
    api_key: str = ""
    secret_key: str = ""
    testnet: bool = True
    base_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate Binance configuration after initialization."""
        # Only require API keys if not in testnet mode
        if not self.testnet:
            if not self.api_key or not self.secret_key:
                raise ConfigurationError(
                    "Binance API key and secret are required for live trading"
                )


@dataclass
class TradingConfig:
    """Trading configuration settings."""
    default_symbol: str = "BTCUSDT"
    default_timeframe: str = "15m"
    max_positions: int = 3
    risk_per_trade: float = 0.01
    max_daily_trades: int = 20
    trading_hours_start: str = "00:00"
    trading_hours_end: str = "23:59"
    
    def __post_init__(self):
        """Validate trading configuration."""
        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            raise ConfigurationError(
                "Risk per trade must be between 0 and 0.1 (0% to 10%)"
            )
        
        if self.max_positions <= 0:
            raise ConfigurationError("Max positions must be greater than 0")


@dataclass
class MLConfig:
    """Machine Learning configuration settings."""
    retrain_interval: str = "24h"
    model_path: str = "data/models/"
    feature_lookback_period: int = 100
    prediction_threshold: float = 0.6
    
    def __post_init__(self):
        """Create model directory if it doesn't exist."""
        Path(self.model_path).mkdir(parents=True, exist_ok=True)


@dataclass
class NotificationConfig:
    """Notification configuration settings."""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    email_enabled: bool = False
    email_smtp_server: str = ""
    email_from: str = ""
    email_to: str = ""


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    file_path: str = "logs/trading_bot.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    def __post_init__(self):
        """Create logs directory if it doesn't exist."""
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class DashboardConfig:
    """Dashboard configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8501
    debug: bool = False


@dataclass
class Settings:
    """Main settings class containing all configuration sections."""
    
    # Configuration sections
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    
    # Convenience properties for backward compatibility
    @property
    def BINANCE_API_KEY(self) -> str:
        return self.binance.api_key
    
    @property
    def BINANCE_SECRET_KEY(self) -> str:
        return self.binance.secret_key
    
    @property
    def BINANCE_TESTNET(self) -> bool:
        return self.binance.testnet
    
    @property
    def DEFAULT_SYMBOL(self) -> str:
        return self.trading.default_symbol
    
    @property
    def DEFAULT_TIMEFRAME(self) -> str:
        return self.trading.default_timeframe
    
    @property
    def MAX_POSITIONS(self) -> int:
        return self.trading.max_positions
    
    @property
    def RISK_PER_TRADE(self) -> float:
        return self.trading.risk_per_trade


class SettingsLoader:
    """Settings loader with environment variable support."""
    
    @staticmethod
    def load_env_file(env_file: str = ".env") -> None:
        """Load environment variables from file."""
        if Path(env_file).exists():
            load_dotenv(env_file)
            logging.info(f"📝 Loaded environment variables from {env_file}")
        else:
            logging.warning(f"⚠️ Environment file {env_file} not found")
    
    @staticmethod
    def get_env(key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get environment variable with type conversion.
        
        Args:
            key (str): Environment variable name
            default (Any): Default value if not found
            required (bool): Whether the variable is required
            
        Returns:
            Any: Environment variable value
            
        Raises:
            ConfigurationError: If required variable is missing
        """
        value = os.getenv(key, default)
        
        if required and value is None:
            raise ConfigurationError(f"Required environment variable {key} is not set")
        
        # Type conversion based on default value type
        if value is not None and default is not None:
            if isinstance(default, bool):
                return value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(default, int):
                try:
                    return int(value)
                except ValueError:
                    raise ConfigurationError(f"Invalid integer value for {key}: {value}")
            elif isinstance(default, float):
                try:
                    return float(value)
                except ValueError:
                    raise ConfigurationError(f"Invalid float value for {key}: {value}")
        
        return value
    
    @classmethod
    def load_settings(cls, env_file: str = ".env") -> Settings:
        """
        Load all settings from environment variables.
        
        Args:
            env_file (str): Path to environment file
            
        Returns:
            Settings: Loaded configuration settings
        """
        # Load environment file
        cls.load_env_file(env_file)
        
        # Load Binance configuration
        binance_config = BinanceConfig(
            api_key=cls.get_env("BINANCE_API_KEY", ""),
            secret_key=cls.get_env("BINANCE_SECRET_KEY", ""),
            testnet=cls.get_env("BINANCE_TESTNET", True)
        )
        
        # Load trading configuration
        trading_config = TradingConfig(
            default_symbol=cls.get_env("DEFAULT_SYMBOL", "BTCUSDT"),
            default_timeframe=cls.get_env("DEFAULT_TIMEFRAME", "15m"),
            max_positions=cls.get_env("MAX_POSITIONS", 3),
            risk_per_trade=cls.get_env("RISK_PER_TRADE", 0.01)
        )
        
        # Load database configuration
        database_config = DatabaseConfig(
            url=cls.get_env("DATABASE_URL", "sqlite:///data/trading_bot.db")
        )
        
        # Load ML configuration
        ml_config = MLConfig(
            retrain_interval=cls.get_env("ML_RETRAIN_INTERVAL", "24h"),
            model_path=cls.get_env("ML_MODEL_PATH", "data/models/")
        )
        
        # Load notification configuration
        notifications_config = NotificationConfig(
            telegram_bot_token=cls.get_env("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=cls.get_env("TELEGRAM_CHAT_ID", ""),
            discord_webhook_url=cls.get_env("DISCORD_WEBHOOK_URL", "")
        )
        
        # Load logging configuration
        logging_config = LoggingConfig(
            level=cls.get_env("LOG_LEVEL", "INFO"),
            file_path=cls.get_env("LOG_FILE", "logs/trading_bot.log")
        )
        
        # Load dashboard configuration
        dashboard_config = DashboardConfig(
            host=cls.get_env("DASHBOARD_HOST", "0.0.0.0"),
            port=cls.get_env("DASHBOARD_PORT", 8501)
        )
        
        return Settings(
            binance=binance_config,
            trading=trading_config,
            database=database_config,
            ml=ml_config,
            notifications=notifications_config,
            logging=logging_config,
            dashboard=dashboard_config
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get global settings instance (singleton pattern).
    
    Args:
        reload (bool): Whether to reload settings from environment
        
    Returns:
        Settings: Global settings instance
    """
    global _settings
    
    if _settings is None or reload:
        try:
            _settings = SettingsLoader.load_settings()
            logging.info("✅ Settings loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load settings: {e}"
            logging.error(f"❌ {error_msg}")
            raise ConfigurationError(error_msg) from e
    
    return _settings