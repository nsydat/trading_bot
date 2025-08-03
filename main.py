#!/usr/bin/env python3
"""
Entry point chính cho Trading Bot
"""

from bot_controller import TradingBotController
from config.settings import SETTINGS
import logging

def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🤖 Starting Trading Bot...")
    
    # Initialize bot controller
    bot = TradingBotController(SETTINGS)
    
    try:
        # Start the bot
        bot.start()
    except KeyboardInterrupt:
        logger.info("👋 Shutting down Trading Bot...")
        bot.stop()
    except Exception as e:
        logger.error(f"❌ Bot crashed: {e}")
        bot.emergency_stop()

if __name__ == "__main__":
    main()
