from aiogram import BaseMiddleware
from help_src.tg_bot.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        logger.info("User %s sent: %s", event.from_user.id, event.text)
        return await handler(event, data)
