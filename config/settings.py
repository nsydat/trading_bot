"""
Configuration Settings - Complete Implementation
==============================================

Comprehensive configuration management for the trading bot with
security best practices, validation, and environment support.

Features:
- Secure API key handling
- Environment-based configuration
- Comprehensive validation
- Multiple deployment environments
- Default fallbacks

Author: dat-ns
Version: 1.0.0
"""

import os
import logging
import warnings
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import base64
import json

from core.utils.exceptions import ConfigurationError


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///data/trading_bot.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    connection_timeout: int = 30
    
    def __post_init__(self):
        """Validate database configuration."""
        if not self.url:
            raise ConfigurationError("Database URL is required")
        
        # Create database directory for SQLite
        if self.url.startswith('sqlite:'):
            db_path = Path(self.url.replace('sqlite:///', ''))
            db_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass 
class BinanceConfig:
    """Binance API configuration with security features."""
    api_key: str = ""
    secret_key: str = ""
    testnet: bool = True
    base_url: Optional[str] = None
    recv_window: int = 5000
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Rate limiting
    requests_per_second: int = 10
    orders_per_second: int = 5
    orders_per_day: int = 200000
    
    # Security settings
    encrypt_keys: bool = False
    key_file: str = "data/.api_keys"
    
    def __post_init__(self):
        """Validate and secure Binance configuration."""
        # Validate API keys for live trading
        if not self.testnet:
            if not self.api_key or not self.secret_key:
                raise ConfigurationError(
                    "Binance API credentials are required for live trading. "
                    "Set BINANCE_TESTNET=true for testnet mode."
                )
            
            # Warn about live trading
            warnings.warn(
                "🚨 LIVE TRADING MODE ENABLED - Real money at risk!",
                UserWarning,
                stacklevel=2
            )
        
        # Validate key format (basic check)
        if self.api_key and len(self.api_key) < 30:
            raise ConfigurationError("Invalid API key format")
        
        if self.secret_key and len(self.secret_key) < 30:
            raise ConfigurationError("Invalid secret key format")
        
        # Set appropriate base URL
        if not self.base_url:
            if self.testnet:
                self.base_url = "https://testnet.binance.vision"
            else:
                self.base_url = "https://api.binance.com"
        
        # Handle key encryption
        if self.encrypt_keys and self.api_key and self.secret_key:
            self._handle_key_encryption()
    
    def _handle_key_encryption(self):
        """Handle API key encryption for security."""
        try:
            key_file_path = Path(self.key_file)
            key_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # For now, just store encoded (not truly encrypted without key management)
            # In production, implement proper key management system
            encoded_keys = {
                'api_key': base64.b64encode(self.api_key.encode()).decode(),
                'secret_key': base64.b64encode(self.secret_key.encode()).decode(),
                'encrypted': True
            }
            
            with open(key_file_path, 'w') as f:
                json.dump(encoded_keys, f)
            
            # Clear from memory
            self.api_key = "ENCRYPTED"
            self.secret_key = "ENCRYPTED"
            
        except Exception as e:
            logging.warning(f"⚠️ Key encryption failed: {e}")
    
    def get_decrypted_keys(self) -> tuple[str, str]:
        """Get decrypted API keys."""
        if not self.encrypt_keys:
            return self.api_key, self.secret_key
        
        try:
            key_file_path = Path(self.key_file)
            if not key_file_path.exists():
                raise ConfigurationError("Encrypted keys file not found")
            
            with open(key_file_path, 'r') as f:
                data = json.load(f)
            
            api_key = base64.b64decode(data['api_key']).decode()
            secret_key = base64.b64decode(data['secret_key']).decode()
            
            return api_key, secret_key
            
        except Exception as e:
            raise ConfigurationError(f"Failed to decrypt keys: {e}")


@dataclass
class TradingConfig:
    """Advanced trading configuration."""
    # Basic settings
    default_symbol: str = "BTCUSDT"
    default_timeframe: str = "15m"
    
    # Position management
    max_positions: int = 3
    max_position_size: float = 0.95  # Max 95% of balance per position
    min_position_size: float = 0.001  # Minimum position size
    
    # Risk management
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_drawdown: float = 0.10   # 10% max drawdown
    stop_loss_percent: float = 0.02  # 2% stop loss
    take_profit_percent: float = 0.04  # 4% take profit
    
    # Trading rules
    max_daily_trades: int = 50
    max_trades_per_hour: int = 10
    min_time_between_trades: int = 60  # seconds
    
    # Trading hours (24h format)
    trading_hours_start: str = "00:00"
    trading_hours_end: str = "23:59"
    trading_timezone: str = "UTC"
    
    # Market conditions
    min_volume_threshold: float = 1000000  # Minimum daily volume
    max_spread_percent: float = 0.5  # Max 0.5% spread
    volatility_threshold: float = 0.05  # 5% volatility threshold
    
    # Strategy settings
    strategy_name: str = "ema_crossover"
    signal_confidence_threshold: float = 0.6
    enable_multiple_strategies: bool = False
    
    # Order settings
    order_timeout: int = 300  # 5 minutes
    slippage_tolerance: float = 0.001  # 0.1% slippage
    use_market_orders: bool = True
    
    # Position management
    close_positions_on_stop: bool = False
    trailing_stop_enabled: bool = False
    trailing_stop_percent: float = 0.01
    
    def __post_init__(self):
        """Validate trading configuration."""
        # Risk validation
        if not 0 < self.risk_per_trade <= 0.1:
            raise ConfigurationError("Risk per trade must be between 0.1% and 10%")
        
        if not 0 < self.max_daily_loss <= 0.2:
            raise ConfigurationError("Max daily loss must be between 0.1% and 20%")
        
        if not 0 < self.max_drawdown <= 0.5:
            raise ConfigurationError("Max drawdown must be between 0.1% and 50%")
        
        # Position validation
        if self.max_positions <= 0 or self.max_positions > 20:
            raise ConfigurationError("Max positions must be between 1 and 20")
        
        if not 0 < self.max_position_size <= 1.0:
            raise ConfigurationError("Max position size must be between 0.1% and 100%")
        
        # Symbol validation
        if not self.default_symbol or len(self.default_symbol) < 6:
            raise ConfigurationError("Invalid trading symbol format")
        
        # Timeframe validation
        valid_timeframes = [
            '1m', '3m', '5m', '15m', '30m', '1h', '2h', 
            '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
        ]
        if self.default_timeframe not in valid_timeframes:
            raise ConfigurationError(f"Invalid timeframe. Valid options: {valid_timeframes}")
        
        # Trading hours validation
        try:
            start_hour, start_min = map(int, self.trading_hours_start.split(':'))
            end_hour, end_min = map(int, self.trading_hours_end.split(':'))
            
            if not (0 <= start_hour <= 23 and 0 <= start_min <= 59):
                raise ValueError("Invalid start time")
            
            if not (0 <= end_hour <= 23 and 0 <= end_min <= 59):
                raise ValueError("Invalid end time")
                
        except ValueError as e:
            raise ConfigurationError(f"Invalid trading hours format: {e}")


@dataclass
class MLConfig:
    """Machine Learning configuration."""
    # Model settings
    model_type: str = "ensemble"  # ensemble, xgboost, lstm, transformer
    model_path: str = "data/models/"
    model_update_interval: str = "24h"
    
    # Training settings
    feature_lookback_period: int = 100
    prediction_horizon: int = 1
    training_data_size: int = 10000
    validation_split: float = 0.2
    
    # Prediction settings
    prediction_threshold: float = 0.6
    prediction_confidence_min: float = 0.7
    ensemble_models: List[str] = field(default_factory=lambda: ["xgboost", "lstm"])
    
    # Feature engineering
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma_20", "ema_12", "ema_26", "rsi_14", "macd", "bollinger_bands",
        "atr_14", "adx_14", "stoch_14", "cci_20"
    ])
    
    # Advanced features
    use_market_regime_detection: bool = True
    use_sentiment_analysis: bool = False
    use_alternative_data: bool = False
    
    def __post_init__(self):
        """Create model directory and validate ML configuration."""
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        
        # Validate settings
        if not 0.5 <= self.prediction_threshold <= 1.0:
            raise ConfigurationError("Prediction threshold must be between 0.5 and 1.0")
        
        if not 0.1 <= self.validation_split <= 0.5:
            raise ConfigurationError("Validation split must be between 0.1 and 0.5")


@dataclass
class NotificationConfig:
    """Comprehensive notification configuration."""
    # Telegram settings
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_parse_mode: str = "Markdown"
    
    # Discord settings
    discord_enabled: bool = False
    discord_webhook_url: str = ""
    discord_username: str = "Trading Bot"
    
    # Email settings
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    # Slack settings
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#trading"
    
    # Push notifications
    pushover_enabled: bool = False
    pushover_user_key: str = ""
    pushover_api_token: str = ""
    
    # Notification levels
    notify_on_startup: bool = True
    notify_on_shutdown: bool = True
    notify_on_trade: bool = True
    notify_on_error: bool = True
    notify_on_profit: bool = True
    notify_on_loss: bool = True
    
    # Rate limiting
    max_notifications_per_hour: int = 60
    error_notification_cooldown: int = 300  # 5 minutes
    
    def __post_init__(self):
        """Validate notification configuration."""
        enabled_services = sum([
            self.telegram_enabled, self.discord_enabled, 
            self.email_enabled, self.slack_enabled, self.pushover_enabled
        ])
        
        if enabled_services == 0:
            logging.warning("⚠️ No notification services enabled")
        
        # Validate Telegram settings
        if self.telegram_enabled:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                raise ConfigurationError("Telegram bot token and chat ID required")
        
        # Validate email settings
        if self.email_enabled:
            if not all([self.email_username, self.email_password, self.email_from]):
                raise ConfigurationError("Email credentials required")
            
            if not self.email_to:
                raise ConfigurationError("Email recipient list required")


@dataclass
class LoggingConfig:
    """Advanced logging configuration."""
    # Basic settings
    level: str = "INFO"
    file_path: str = "logs/trading_bot.log"
    
    # File rotation
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    backup_count: int = 10
    
    # Format settings
    detailed_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    simple_format: str = "%(asctime)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Advanced logging
    enable_json_logs: bool = False
    json_log_file: str = "logs/trading_bot.json"
    
    # Performance logging
    log_performance_metrics: bool = True
    performance_log_interval: int = 300  # 5 minutes
    
    # Error logging
    error_log_file: str = "logs/errors.log"
    capture_warnings: bool = True
    
    # Trade logging
    trade_log_file: str = "logs/trades.log"
    log_all_trades: bool = True
    
    # Debug settings
    log_api_calls: bool = False
    log_strategy_details: bool = False
    log_risk_calculations: bool = False
    
    def __post_init__(self):
        """Create logging directories and validate configuration."""
        # Create all log directories
        log_dirs = [
            Path(self.file_path).parent,
            Path(self.json_log_file).parent if self.enable_json_logs else None,
            Path(self.error_log_file).parent,
            Path(self.trade_log_file).parent
        ]
        
        for log_dir in log_dirs:
            if log_dir:
                log_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.level.upper() not in valid_levels:
            raise ConfigurationError(f"Invalid log level. Valid options: {valid_levels}")


@dataclass
class DashboardConfig:
    """Dashboard and monitoring configuration."""
    # Web dashboard
    enabled: bool = True
    host: str = "127.0.0.1"  # Secure default
    port: int = 8501
    debug: bool = False
    
    # Authentication
    enable_auth: bool = True
    username: str = "admin"
    password: str = ""  # Should be set via environment
    
    # Features
    enable_live_charts: bool = True
    chart_update_interval: int = 5  # seconds
    enable_trade_history: bool = True
    enable_performance_metrics: bool = True
    
    # Data retention
    max_chart_data_points: int = 1000
    metrics_retention_days: int = 30
    
    def __post_init__(self):
        """Validate dashboard configuration."""
        if self.enabled and self.enable_auth and not self.password:
            raise ConfigurationError("Dashboard password required when authentication enabled")
        
        if not 1024 <= self.port <= 65535:
            raise ConfigurationError("Dashboard port must be between 1024 and 65535")


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    # API security
    enable_api_key_rotation: bool = False
    api_key_rotation_days: int = 30
    
    # Access control
    allowed_ips: List[str] = field(default_factory=lambda: ["127.0.0.1"])
    enable_ip_whitelist: bool = False
    
    # Encryption
    encrypt_sensitive_data: bool = True
    encryption_key_file: str = "data/.encryption_key"
    
    # Audit logging
    enable_audit_log: bool = True
    audit_log_file: str = "logs/audit.log"
    
    # Session security
    session_timeout: int = 3600  # 1 hour
    max_failed_attempts: int = 5
    lockout_duration: int = 1800  # 30 minutes
    
    def __post_init__(self):
        """Initialize security settings."""
        if self.encrypt_sensitive_data:
            self._ensure_encryption_key()
        
        if self.enable_audit_log:
            Path(self.audit_log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def _ensure_encryption_key(self):
        """Ensure encryption key exists."""
        key_file = Path(self.encryption_key_file)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not key_file.exists():
            # Generate new key
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            key_file.chmod(0o600)  # Read/write for owner only


@dataclass
class Settings:
    """Main settings class with all configuration sections."""
    
    # Configuration sections
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Environment info
    environment: str = "development"
    debug: bool = False
    dry_run: bool = False
    
    # Version info
    version: str = "1.0.0"
    config_version: str = "1.0"
    
    # Convenience properties for backward compatibility
    @property
    def BINANCE_API_KEY(self) -> str:
        api_key, _ = self.binance.get_decrypted_keys()
        return api_key
    
    @property
    def BINANCE_SECRET_KEY(self) -> str:
        _, secret_key = self.binance.get_decrypted_keys()
        return secret_key
    
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
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Set environment-specific defaults
        if self.environment == "production":
            self.debug = False
            self.logging.level = "INFO"
        elif self.environment == "development":
            self.debug = True
            self.logging.level = "DEBUG"
        
        # Validate environment
        valid_environments = ["development", "testing", "staging", "production"]
        if self.environment not in valid_environments:
            raise ConfigurationError(f"Invalid environment. Valid options: {valid_environments}")


class SettingsLoader:
    """Advanced settings loader with multiple environment support."""
    
    @staticmethod
    def load_env_file(env_file: str = None) -> None:
        """Load environment variables from file with fallbacks."""
        if env_file is None:
            # Try multiple environment files in order
            env_files = [
                f".env.{os.getenv('ENVIRONMENT', 'development')}",
                ".env.local",
                ".env"
            ]
        else:
            env_files = [env_file]
        
        loaded_file = None
        for env_file in env_files:
            env_path = Path(env_file)
            if env_path.exists():
                load_dotenv(env_path)
                loaded_file = env_file
                logging.info(f"📝 Loaded environment file: {env_file}")
                break
        
        if not loaded_file:
            logging.warning("⚠️ No environment file found, using system environment variables")
    
    @staticmethod
    def get_env(key: str, default: Any = None, required: bool = False, 
                data_type: type = str) -> Any:
        """
        Get environment variable with advanced type conversion and validation.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            data_type: Expected data type
            
        Returns:
            Converted environment variable value
        """
        value = os.getenv(key, default)
        
        if required and value is None:
            raise ConfigurationError(f"Required environment variable {key} is not set")
        
        if value is None:
            return default
        
        # Type conversion
        try:
            if data_type == bool:
                return str(value).lower() in ('true', '1', 'yes', 'on', 'enabled')
            elif data_type == int:
                return int(value)
            elif data_type == float:
                return float(value)
            elif data_type == list:
                return [item.strip() for item in str(value).split(',') if item.strip()]
            else:
                return str(value)
                
        except (ValueError, TypeError) as e:
            raise ConfigurationError(f"Invalid {data_type.__name__} value for {key}: {value}") from e
    
    @classmethod
    def load_settings(cls, env_file: str = None) -> Settings:
        """
        Load comprehensive settings from environment variables.
        
        Args:
            env_file: Path to environment file (optional)
            
        Returns:
            Fully configured Settings object
        """
        # Load environment file
        cls.load_env_file(env_file)
        
        # Determine environment
        environment = cls.get_env("ENVIRONMENT", "development")
        
        # Load Binance configuration
        binance_config = BinanceConfig(
            api_key=cls.get_env("BINANCE_API_KEY", ""),
            secret_key=cls.get_env("BINANCE_SECRET_KEY", ""),
            testnet=cls.get_env("BINANCE_TESTNET", True, data_type=bool),
            recv_window=cls.get_env("BINANCE_RECV_WINDOW", 5000, data_type=int),
            timeout=cls.get_env("BINANCE_TIMEOUT", 30, data_type=int),
            encrypt_keys=cls.get_env("ENCRYPT_API_KEYS", False, data_type=bool)
        )
        
        # Load trading configuration
        trading_config = TradingConfig(
            default_symbol=cls.get_env("DEFAULT_SYMBOL", "BTCUSDT"),
            default_timeframe=cls.get_env("DEFAULT_TIMEFRAME", "15m"),
            max_positions=cls.get_env("MAX_POSITIONS", 3, data_type=int),
            risk_per_trade=cls.get_env("RISK_PER_TRADE", 0.02, data_type=float),
            max_daily_loss=cls.get_env("MAX_DAILY_LOSS", 0.05, data_type=float),
            max_drawdown=cls.get_env("MAX_DRAWDOWN", 0.10, data_type=float),
            strategy_name=cls.get_env("STRATEGY_NAME", "ema_crossover"),
            close_positions_on_stop=cls.get_env("CLOSE_POSITIONS_ON_STOP", False, data_type=bool)
        )
        
        # Load database configuration
        database_config = DatabaseConfig(
            url=cls.get_env("DATABASE_URL", "sqlite:///data/trading_bot.db"),
            echo=cls.get_env("DATABASE_ECHO", False, data_type=bool)
        )
        
        # Load ML configuration
        ml_config = MLConfig(
            model_type=cls.get_env("ML_MODEL_TYPE", "ensemble"),
            model_path=cls.get_env("ML_MODEL_PATH", "data/models/"),
            prediction_threshold=cls.get_env("ML_PREDICTION_THRESHOLD", 0.6, data_type=float),
            use_market_regime_detection=cls.get_env("USE_MARKET_REGIME", True, data_type=bool)
        )
        
        # Load notification configuration
        notifications_config = NotificationConfig(
            telegram_enabled=cls.get_env("TELEGRAM_ENABLED", False, data_type=bool),
            telegram_bot_token=cls.get_env("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=cls.get_env("TELEGRAM_CHAT_ID", ""),
            discord_enabled=cls.get_env("DISCORD_ENABLED", False, data_type=bool),
            discord_webhook_url=cls.get_env("DISCORD_WEBHOOK_URL", ""),
            email_enabled=cls.get_env("EMAIL_ENABLED", False, data_type=bool),
            email_username=cls.get_env("EMAIL_USERNAME", ""),
            email_password=cls.get_env("EMAIL_PASSWORD", ""),
            email_to=cls.get_env("EMAIL_TO", [], data_type=list)
        )
        
        # Load logging configuration
        logging_config = LoggingConfig(
            level=cls.get_env("LOG_LEVEL", "INFO").upper(),
            file_path=cls.get_env("LOG_FILE", "logs/trading_bot.log"),
            enable_json_logs=cls.get_env("ENABLE_JSON_LOGS", False, data_type=bool),
            log_performance_metrics=cls.get_env("LOG_PERFORMANCE", True, data_type=bool),
            log_api_calls=cls.get_env("LOG_API_CALLS", False, data_type=bool)
        )
        
        # Load dashboard configuration
        dashboard_config = DashboardConfig(
            enabled=cls.get_env("DASHBOARD_ENABLED", True, data_type=bool),
            host=cls.get_env("DASHBOARD_HOST", "127.0.0.1"),
            port=cls.get_env("DASHBOARD_PORT", 8501, data_type=int),
            enable_auth=cls.get_env("DASHBOARD_AUTH", True, data_type=bool),
            password=cls.get_env("DASHBOARD_PASSWORD", "")
        )
        
        # Load security configuration
        security_config = SecurityConfig(
            encrypt_sensitive_data=cls.get_env("ENCRYPT_DATA", True, data_type=bool),
            enable_ip_whitelist=cls.get_env("ENABLE_IP_WHITELIST", False, data_type=bool),
            allowed_ips=cls.get_env("ALLOWED_IPS", ["127.0.0.1"], data_type=list),
            enable_audit_log=cls.get_env("ENABLE_AUDIT_LOG", True, data_type=bool)
        )
        
        # Create main settings object
        settings = Settings(
            binance=binance_config,
            trading=trading_config,
            database=database_config,
            ml=ml_config,
            notifications=notifications_config,
            logging=logging_config,
            dashboard=dashboard_config,
            security=security_config,
            environment=environment,
            debug=cls.get_env("DEBUG", environment == "development", data_type=bool),
            dry_run=cls.get_env("DRY_RUN", False, data_type=bool)
        )
        
        return settings


class ConfigurationValidator:
    """Advanced configuration validation utilities."""
    
    @staticmethod
    def validate_production_config(settings: Settings) -> List[str]:
        """
        Validate configuration for production deployment.
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        # Security checks
        if settings.environment == "production":
            if settings.debug:
                issues.append("DEBUG mode should be disabled in production")
            
            if settings.binance.testnet:
                issues.append("Testnet mode enabled in production environment")
            
            if settings.dashboard.enabled and not settings.dashboard.enable_auth:
                issues.append("Dashboard authentication should be enabled in production")
            
            if not settings.security.encrypt_sensitive_data:
                issues.append("Sensitive data encryption should be enabled in production")
        
        # Trading safety checks
        if settings.trading.risk_per_trade > 0.05:
            issues.append(f"High risk per trade: {settings.trading.risk_per_trade*100:.1f}%")
        
        if settings.trading.max_daily_loss > 0.1:
            issues.append(f"High daily loss limit: {settings.trading.max_daily_loss*100:.1f}%")
        
        # Notification checks
        notification_enabled = any([
            settings.notifications.telegram_enabled,
            settings.notifications.discord_enabled,
            settings.notifications.email_enabled
        ])
        
        if not notification_enabled:
            issues.append("No notification services configured")
        
        return issues
    
    @staticmethod
    def generate_example_env_file(file_path: str = ".env.example"):
        """Generate example environment file with all settings."""
        env_content = """# Trading Bot Configuration Example
# Copy this file to .env and modify the values

# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================
ENVIRONMENT=development
DEBUG=true
DRY_RUN=false

# =============================================================================
# BINANCE API SETTINGS
# =============================================================================
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
BINANCE_TESTNET=true
BINANCE_TIMEOUT=30
ENCRYPT_API_KEYS=false

# =============================================================================
# TRADING SETTINGS
# =============================================================================
DEFAULT_SYMBOL=BTCUSDT
DEFAULT_TIMEFRAME=15m
MAX_POSITIONS=3
RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05
MAX_DRAWDOWN=0.10
STRATEGY_NAME=ema_crossover
CLOSE_POSITIONS_ON_STOP=false

# =============================================================================
# DATABASE SETTINGS
# =============================================================================
DATABASE_URL=sqlite:///data/trading_bot.db
DATABASE_ECHO=false

# =============================================================================
# MACHINE LEARNING SETTINGS
# =============================================================================
ML_MODEL_TYPE=ensemble
ML_MODEL_PATH=data/models/
ML_PREDICTION_THRESHOLD=0.6
USE_MARKET_REGIME=true

# =============================================================================
# NOTIFICATION SETTINGS
# =============================================================================
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

DISCORD_ENABLED=false
DISCORD_WEBHOOK_URL=your_discord_webhook_url

EMAIL_ENABLED=false
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_TO=recipient1@example.com,recipient2@example.com

# =============================================================================
# LOGGING SETTINGS
# =============================================================================
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log
ENABLE_JSON_LOGS=false
LOG_PERFORMANCE=true
LOG_API_CALLS=false

# =============================================================================
# DASHBOARD SETTINGS
# =============================================================================
DASHBOARD_ENABLED=true
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=8501
DASHBOARD_AUTH=true
DASHBOARD_PASSWORD=secure_password_here

# =============================================================================
# SECURITY SETTINGS
# =============================================================================
ENCRYPT_DATA=true
ENABLE_IP_WHITELIST=false
ALLOWED_IPS=127.0.0.1,192.168.1.100
ENABLE_AUDIT_LOG=true
"""
        
        with open(file_path, 'w') as f:
            f.write(env_content)
        
        logging.info(f"📝 Generated example environment file: {file_path}")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(reload: bool = False, env_file: str = None) -> Settings:
    """
    Get global settings instance with caching and validation.
    
    Args:
        reload: Force reload settings from environment
        env_file: Specific environment file to load
        
    Returns:
        Global settings instance
    """
    global _settings
    
    if _settings is None or reload:
        try:
            _settings = SettingsLoader.load_settings(env_file)
            
            # Validate production configuration
            if _settings.environment == "production":
                issues = ConfigurationValidator.validate_production_config(_settings)
                if issues:
                    logging.warning("⚠️ Production configuration issues:")
                    for issue in issues:
                        logging.warning(f"  - {issue}")
            
            logging.info(f"✅ Settings loaded successfully (environment: {_settings.environment})")
            
        except Exception as e:
            error_msg = f"Failed to load settings: {e}"
            logging.error(f"❌ {error_msg}")
            raise ConfigurationError(error_msg) from e
    
    return _settings


def create_example_config():
    """Create example configuration files for setup."""
    ConfigurationValidator.generate_example_env_file()
    logging.info("✅ Example configuration files created")


# CLI utility for configuration management
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "create-example":
            create_example_config()
        elif sys.argv[1] == "validate":
            try:
                settings = get_settings()
                issues = ConfigurationValidator.validate_production_config(settings)
                
                if issues:
                    print("⚠️ Configuration Issues:")
                    for issue in issues:
                        print(f"  - {issue}")
                else:
                    print("✅ Configuration validation passed")
                    
            except Exception as e:
                print(f"❌ Configuration validation failed: {e}")
                sys.exit(1)
        else:
            print("Usage: python settings.py [create-example|validate]")
    else:
        # Test configuration loading
        try:
            settings = get_settings()
            print("✅ Settings loaded successfully")
            print(f"Environment: {settings.environment}")
            print(f"Trading Symbol: {settings.trading.default_symbol}")
            print(f"Testnet Mode: {settings.binance.testnet}")
        except Exception as e:
            print(f"❌ Failed to load settings: {e}")
            sys.exit(1)