#!/usr/bin/env python3
"""
Trading Bot Main Entry Point
============================

Complete automated trading bot system with integrated components.
Handles initialization, startup, shutdown, and error recovery.

Features:
- Complete component integration
- Robust error handling and recovery
- Secure API key management
- Performance monitoring
- Graceful shutdown procedures
- Configuration-driven setup

Author: dat-ns
Version: 1.0.0
"""

import sys
import signal
import logging
import asyncio
import traceback
from pathlib import Path
from typing import Optional
from datetime import datetime
import warnings
import json

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from bot_controller import TradingBotController, BotState
from config.settings import get_settings, SettingsLoader
from core.utils.exceptions import (
    TradingBotException,
    ConfigurationError,
    InitializationError,
    ExchangeConnectionError
)


class TradingBotApp:
    """
    Main application class for the trading bot.
    
    Handles the complete application lifecycle including:
    - Configuration validation and loading
    - Component initialization and dependency management
    - Signal handling for graceful shutdown
    - Error recovery and restart logic
    - Performance monitoring and health checks
    """
    
    def __init__(self):
        """Initialize the trading bot application."""
        self.bot_controller: Optional[TradingBotController] = None
        self.logger: Optional[logging.Logger] = None
        self.is_running = False
        self.restart_requested = False
        self.shutdown_initiated = False
        
        # Performance metrics
        self.start_time: Optional[datetime] = None
        self.total_runtime = 0
        self.restart_count = 0
        self.max_restarts = 5
        
        # Setup signal handlers early
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Windows compatibility
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame):
        """
        Handle system signals for graceful shutdown.
        
        Args:
            signum (int): Signal number
            frame: Current stack frame
        """
        signal_names = {2: 'SIGINT', 15: 'SIGTERM'}
        if hasattr(signal, 'SIGBREAK') and signum == signal.SIGBREAK:
            signal_names[signal.SIGBREAK] = 'SIGBREAK'
        
        signal_name = signal_names.get(signum, f'Signal {signum}')
        
        if self.logger:
            self.logger.info(f"📨 Received {signal_name}, initiating graceful shutdown...")
        else:
            print(f"📨 Received {signal_name}, initiating graceful shutdown...")
        
        self.shutdown()
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup comprehensive logging configuration.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        try:
            # Create logs directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Get log level from settings
            try:
                settings = get_settings()
                log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
                log_file = settings.logging.file_path
            except:
                log_level = logging.INFO
                log_file = "logs/trading_bot.log"
            
            # Create formatters
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            simple_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            
            # Setup file handler with rotation
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            
            # Setup console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(simple_formatter)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            root_logger.handlers.clear()
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            
            # Create application logger
            logger = logging.getLogger(__name__)
            logger.info("✅ Logging system initialized successfully")
            
            return logger
            
        except Exception as e:
            # Fallback logging setup
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler('trading_bot_fallback.log', encoding='utf-8')
                ]
            )
            
            logger = logging.getLogger(__name__)
            logger.warning(f"⚠️ Using fallback logging due to error: {e}")
            return logger
    
    def _validate_environment(self) -> bool:
        """
        Validate the runtime environment.
        
        Returns:
            bool: True if environment is valid
        """
        try:
            self.logger.info("🔍 Validating runtime environment...")
            
            # Check Python version
            if sys.version_info < (3, 8):
                self.logger.error("❌ Python 3.8+ required")
                return False
            
            # Check required directories
            required_dirs = ['logs', 'data', 'data/models']
            for dir_path in required_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"📁 Ensured directory exists: {dir_path}")
            
            # Check disk space (minimum 100MB)
            import shutil
            free_space = shutil.disk_usage('.').free / (1024**2)  # MB
            if free_space < 100:
                self.logger.warning(f"⚠️ Low disk space: {free_space:.1f}MB available")
            
            # Check memory (minimum 512MB available)
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**2)  # MB
            if available_memory < 512:
                self.logger.warning(f"⚠️ Low memory: {available_memory:.1f}MB available")
            
            self.logger.info("✅ Environment validation passed")
            return True
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Optional dependency missing: {e}")
            return True  # Continue without system checks
        except Exception as e:
            self.logger.error(f"❌ Environment validation failed: {e}")
            return False
    
    def _load_configuration(self) -> bool:
        """
        Load and validate configuration settings.
        
        Returns:
            bool: True if configuration loaded successfully
        """
        try:
            self.logger.info("📝 Loading configuration settings...")
            
            # Check for environment file
            env_files = ['.env', '.env.local', '.env.production']
            env_file_found = None
            
            for env_file in env_files:
                if Path(env_file).exists():
                    env_file_found = env_file
                    self.logger.info(f"📄 Using environment file: {env_file}")
                    break
            
            if not env_file_found:
                self.logger.warning("⚠️ No environment file found, using defaults")
            
            # Load settings with error handling
            settings = get_settings(reload=True)
            
            # Validate critical settings
            self._validate_critical_settings(settings)
            
            # Log configuration summary (without sensitive data)
            self._log_configuration_summary(settings)
            
            self.logger.info("✅ Configuration loaded successfully")
            return True
            
        except ConfigurationError as e:
            self.logger.error(f"❌ Configuration error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            return False
    
    def _validate_critical_settings(self, settings):
        """Validate critical settings for production use."""
        
        # API Key validation for live trading
        if not settings.binance.testnet:
            if not settings.binance.api_key or not settings.binance.secret_key:
                raise ConfigurationError(
                    "Binance API credentials required for live trading. "
                    "Set BINANCE_TESTNET=true for testnet mode."
                )
            
            # Warn about live trading
            self.logger.warning("⚠️ LIVE TRADING MODE ENABLED - Real money at risk!")
            response = input("Are you sure you want to continue with live trading? (yes/no): ")
            if response.lower() != 'yes':
                raise ConfigurationError("Live trading cancelled by user")
        
        # Risk management validation
        if settings.trading.risk_per_trade > 0.05:  # 5%
            self.logger.warning(f"⚠️ High risk per trade: {settings.trading.risk_per_trade*100:.1f}%")
        
        # Timeframe validation
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        if settings.trading.default_timeframe not in valid_timeframes:
            raise ConfigurationError(f"Invalid timeframe: {settings.trading.default_timeframe}")
    
    def _log_configuration_summary(self, settings):
        """Log configuration summary without sensitive information."""
        summary = {
            'Trading Mode': 'TESTNET' if settings.binance.testnet else 'LIVE',
            'Symbol': settings.trading.default_symbol,
            'Timeframe': settings.trading.default_timeframe,
            'Max Positions': settings.trading.max_positions,
            'Risk Per Trade': f"{settings.trading.risk_per_trade*100:.1f}%",
            'Log Level': settings.logging.level
        }
        
        self.logger.info("📊 Configuration Summary:")
        for key, value in summary.items():
            self.logger.info(f"  {key}: {value}")
    
    async def initialize(self) -> bool:
        """
        Initialize the trading bot with all components.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("🔧 Initializing Trading Bot System...")
            
            # Validate environment
            if not self._validate_environment():
                return False
            
            # Load configuration
            if not self._load_configuration():
                return False
            
            # Get validated settings
            settings = get_settings()
            
            # Initialize bot controller
            self.logger.info("🤖 Creating Bot Controller...")
            self.bot_controller = TradingBotController(settings)
            
            # Initialize bot components with timeout
            self.logger.info("⚙️ Initializing bot components...")
            try:
                await asyncio.wait_for(
                    self.bot_controller.initialize(),
                    timeout=60.0  # 1 minute timeout
                )
            except asyncio.TimeoutError:
                raise InitializationError("Component initialization timed out after 60 seconds")
            
            # Validate initialization
            status = await self.bot_controller.get_status()
            if status['bot_info']['state'] != BotState.STOPPED.value:
                raise InitializationError(f"Bot in unexpected state: {status['bot_info']['state']}")
            
            # Log system information
            await self._log_system_info()
            
            self.logger.info("✅ Trading Bot initialized successfully")
            return True
            
        except InitializationError as e:
            self.logger.error(f"❌ Initialization error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during initialization: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def _log_system_info(self):
        """Log system and bot information."""
        try:
            import platform
            import psutil
            
            system_info = {
                'Platform': f"{platform.system()} {platform.release()}",
                'Python': f"{platform.python_version()}",
                'CPU Cores': psutil.cpu_count(),
                'Memory': f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
                'Available Memory': f"{psutil.virtual_memory().available / (1024**2):.0f}MB"
            }
            
            self.logger.info("💻 System Information:")
            for key, value in system_info.items():
                self.logger.info(f"  {key}: {value}")
                
        except ImportError:
            self.logger.debug("System info logging skipped (psutil not available)")
        except Exception as e:
            self.logger.warning(f"Failed to log system info: {e}")
    
    async def run(self) -> int:
        """
        Main application run loop with error recovery.
        
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        exit_code = 1
        
        try:
            # Setup logging first
            self.logger = self._setup_logging()
            self.logger.info("🚀 Starting Trading Bot Application...")
            self.logger.info(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Main execution loop with restart capability
            while self.restart_count <= self.max_restarts and not self.shutdown_initiated:
                try:
                    # Initialize the bot
                    if not await self.initialize():
                        self.logger.error("❌ Initialization failed")
                        break
                    
                    # Record successful start
                    self.start_time = datetime.now()
                    self.is_running = True
                    
                    self.logger.info("🎯 Trading Bot started successfully")
                    if self.restart_count > 0:
                        self.logger.info(f"🔄 Restart #{self.restart_count} successful")
                    
                    # Start the bot controller
                    await self.bot_controller.start()
                    
                    # If we reach here, bot stopped normally
                    exit_code = 0
                    break
                    
                except KeyboardInterrupt:
                    self.logger.info("👋 Keyboard interrupt received")
                    exit_code = 0
                    break
                    
                except TradingBotException as e:
                    self.logger.error(f"❌ Trading bot error: {e}")
                    
                    # Attempt restart if under limit
                    if self._should_restart(e):
                        await self._attempt_restart()
                    else:
                        break
                        
                except Exception as e:
                    self.logger.error(f"❌ Unexpected error: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Attempt restart for unexpected errors
                    if self._should_restart(e):
                        await self._attempt_restart()
                    else:
                        break
            
            # Check if max restarts exceeded
            if self.restart_count > self.max_restarts:
                self.logger.error(f"❌ Maximum restart attempts ({self.max_restarts}) exceeded")
                exit_code = 1
            
            return exit_code
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Critical error in main loop: {e}")
            else:
                print(f"❌ Critical error: {e}")
            return 1
            
        finally:
            await self._cleanup()
    
    def _should_restart(self, error) -> bool:
        """
        Determine if bot should attempt restart based on error type.
        
        Args:
            error: The error that occurred
            
        Returns:
            bool: True if restart should be attempted
        """
        if self.shutdown_initiated or self.restart_count >= self.max_restarts:
            return False
        
        # Don't restart for configuration errors
        if isinstance(error, ConfigurationError):
            return False
        
        # Don't restart for user interrupts
        if isinstance(error, KeyboardInterrupt):
            return False
        
        # Restart for network/connection issues
        if isinstance(error, (ExchangeConnectionError, ConnectionError)):
            return True
        
        # Restart for other trading bot exceptions
        if isinstance(error, TradingBotException):
            return True
        
        # Don't restart for unknown errors by default
        return False
    
    async def _attempt_restart(self):
        """Attempt to restart the bot after a delay."""
        self.restart_count += 1
        restart_delay = min(30, 5 * self.restart_count)  # Exponential backoff, max 30s
        
        self.logger.warning(f"🔄 Attempting restart #{self.restart_count} in {restart_delay} seconds...")
        
        # Cleanup current instance
        await self._cleanup_bot_only()
        
        # Wait before restart
        await asyncio.sleep(restart_delay)
        
        self.logger.info(f"🔄 Starting restart attempt #{self.restart_count}...")
    
    def shutdown(self):
        """Initiate graceful shutdown of the trading bot."""
        if self.shutdown_initiated:
            return
        
        self.shutdown_initiated = True
        
        if self.logger:
            self.logger.info("🛑 Shutdown initiated...")
        
        # Stop main loop
        self.is_running = False
    
    async def _cleanup_bot_only(self):
        """Cleanup only bot controller resources."""
        if self.bot_controller:
            try:
                await self.bot_controller.stop()
                self.bot_controller = None
            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ Error stopping bot controller: {e}")
    
    async def _cleanup(self):
        """Comprehensive cleanup of all resources."""
        if self.logger:
            self.logger.info("🧹 Performing final cleanup...")
        
        # Stop bot controller
        await self._cleanup_bot_only()
        
        # Calculate runtime
        if self.start_time:
            runtime = datetime.now() - self.start_time
            self.total_runtime += runtime.total_seconds()
            
            if self.logger:
                self.logger.info(f"⏱️ Session runtime: {self._format_duration(runtime.total_seconds())}")
                self.logger.info(f"⏱️ Total runtime: {self._format_duration(self.total_runtime)}")
        
        # Log final statistics
        if self.logger:
            self.logger.info(f"🔄 Total restarts: {self.restart_count}")
            self.logger.info("👋 Trading Bot shutdown complete")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


async def main() -> int:
    """
    Main function to run the trading bot application.
    
    Returns:
        int: Exit code (0 = success, 1 = error)
    """
    app = TradingBotApp()
    return await app.run()


def cli_main():
    """Command line interface entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Automated Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with default settings
  python main.py --testnet          # Force testnet mode
  python main.py --config custom.env # Use custom config file
  python main.py --log-level DEBUG   # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='.env',
        help='Configuration file path (default: .env)'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Override log level'
    )
    
    parser.add_argument(
        '--testnet', '-t',
        action='store_true',
        help='Force testnet mode'
    )
    
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Run without executing trades (simulation only)'
    )
    
    parser.add_argument(
        '--backtest', '-b',
        action='store_true',
        help='Run in backtest mode instead of live trading'
    )
    
    parser.add_argument(
        '--strategy', 
        choices=['ema_crossover', 'simple_ma', 'rsi'],
        help='Strategy to use for backtesting'
    )
    
    parser.add_argument(
        '--backtest-days', type=int, default=365,
        help='Number of days to backtest (default: 365)'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Trading Bot v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Set environment variables from CLI args
    import os
    if args.testnet:
        os.environ['BINANCE_TESTNET'] = 'true'
    
    if args.log_level:
        os.environ['LOG_LEVEL'] = args.log_level
    
    if args.dry_run:
        os.environ['DRY_RUN'] = 'true'
    
    # Update config file path
    if args.config != '.env':
        # Load custom config file
        from pathlib import Path
        if not Path(args.config).exists():
            print(f"❌ Configuration file not found: {args.config}")
            return 1
    
    # Run backtest mode if requested
    if args.backtest:
        try:
            from run_backtest import BacktestRunner
            runner = BacktestRunner()
            
            if args.strategy:
                print(f"🔄 Running backtest with {args.strategy} strategy...")
                results = await runner.run_backtest(
                    strategy_name=args.strategy,
                    symbol="BTCUSDT",
                    days=args.backtest_days,
                    initial_capital=10000.0
                )
                runner.print_results(results)
                return 0
            else:
                print("📊 Available strategies:")
                runner.list_strategies()
                print("\n💡 Use --strategy <name> to run a specific strategy")
                return 0
                
        except Exception as e:
            print(f"❌ Backtest error: {e}")
            return 1
    
    # Run the application
    try:
        exit_code = asyncio.run(main())
        return exit_code
    except KeyboardInterrupt:
        print("\n👋 Trading Bot stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = cli_main()
    sys.exit(exit_code)