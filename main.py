#!/usr/bin/env python3
"""
Trading Bot Main Entry Point
============================

Main entry point for the automated trading bot system.
Handles initialization, startup, shutdown, and error recovery.

Author: dat-ns
Version: 1.0.0
"""

import sys
import signal
import logging
import asyncio
from pathlib import Path
from typing import Optional

from bot_controller import TradingBotController
from config.settings import get_settings
from core.utils.exceptions import (
    TradingBotException,
    ConfigurationError,
    ExchangeConnectionError
)


class TradingBotApp:
    """
    Main application class for the trading bot.
    
    Handles the application lifecycle including:
    - Configuration loading
    - Bot initialization
    - Graceful shutdown
    - Signal handling
    """
    
    def __init__(self):
        """Initialize the trading bot application."""
        self.bot_controller: Optional[TradingBotController] = None
        self.logger = self._setup_logging()
        self.is_running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging configuration.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        # Fix UnicodeEncodeError for emojis in logs by using UTF-8 encoding
        file_handler = logging.FileHandler('logs/trading_bot.log', encoding='utf-8')
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[file_handler, stream_handler]
        )
        
        return logging.getLogger(__name__)
    
    def _signal_handler(self, signum: int, frame):
        """
        Handle system signals for graceful shutdown.
        
        Args:
            signum (int): Signal number
            frame: Current stack frame
        """
        signal_names = {2: 'SIGINT', 15: 'SIGTERM'}
        signal_name = signal_names.get(signum, f'Signal {signum}')
        
        self.logger.info(f"📨 Received {signal_name}, initiating graceful shutdown...")
        self.shutdown()
    
    async def initialize(self) -> bool:
        """
        Initialize the trading bot with configuration and dependencies.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("🔧 Initializing Trading Bot...")
            
            # Load configuration
            self.logger.info("📝 Loading configuration...")
            settings = get_settings()
            
            # Initialize bot controller
            self.logger.info("🤖 Initializing Bot Controller...")
            self.bot_controller = TradingBotController(settings)
            
            # Initialize bot components
            await self.bot_controller.initialize()
            
            self.logger.info("✅ Trading Bot initialized successfully")
            return True
            
        except ConfigurationError as e:
            self.logger.error(f"❌ Configuration error: {e}")
            return False
        except ExchangeConnectionError as e:
            self.logger.error(f"❌ Exchange connection error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during initialization: {e}")
            return False
    
    async def run(self) -> int:
        """
        Main application run loop.
        
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        try:
            # Initialize the bot
            if not await self.initialize():
                return 1
            
            self.is_running = True
            self.logger.info("🚀 Trading Bot started successfully")
            self.logger.info("🔄 Bot is now running... Press Ctrl+C to stop")
            
            # Start the bot controller
            await self.bot_controller.start()
            
            # Keep the main loop running
            while self.is_running:
                await asyncio.sleep(1)
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("👋 Keyboard interrupt received")
            return 0
        except TradingBotException as e:
            self.logger.error(f"❌ Trading bot error: {e}")
            return 1
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}")
            return 1
        finally:
            await self._cleanup()
    
    def shutdown(self):
        """Initiate graceful shutdown of the trading bot."""
        if not self.is_running:
            return
            
        self.logger.info("🛑 Shutting down Trading Bot...")
        self.is_running = False
    
    async def _cleanup(self):
        """Cleanup resources and stop bot components."""
        if self.bot_controller:
            try:
                await self.bot_controller.stop()
                self.logger.info("✅ Bot Controller stopped successfully")
            except Exception as e:
                self.logger.error(f"❌ Error stopping Bot Controller: {e}")
        
        self.logger.info("👋 Trading Bot shutdown complete")


async def main() -> int:
    """
    Main function to run the trading bot application.
    
    Returns:
        int: Exit code
    """
    app = TradingBotApp()
    return await app.run()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Trading Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)