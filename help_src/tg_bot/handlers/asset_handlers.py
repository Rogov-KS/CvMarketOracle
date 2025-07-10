from aiogram import Router, types, Bot
from aiogram.filters import Command

from help_src.tg_bot.services.assets_list import get_formatted_assets_list
from help_src.tg_bot.config import Config
from help_src.tg_bot.logger import get_logger

logger = get_logger(__name__)

router = Router()


@router.message(Command(commands=["assets_list"]))
async def assets_list_command(
    message: types.Message, config: Config, bot: Bot
):
    """Обработчик команды /assets_list."""
    tinkoff_token = config.t_token
    if not tinkoff_token:
        await message.answer(
            "Ошибка: Токен Tinkoff API не настроен для выполнения запроса."
        )
        return

    loading_msg = await message.answer(
        "<i>Загружаю список активов...</i>", parse_mode="HTML"
    )

    try:
        assets_text = await get_formatted_assets_list(token=tinkoff_token)
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=loading_msg.message_id,
            text=assets_text,
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(
            "Ошибка в assets_list_command при вызове "
            "get_formatted_assets_list: %s", e
        )
        error_text = (
            "Произошла неожиданная ошибка при загрузке списка активов. "
            "Попробуйте позже."
        )
        await bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=loading_msg.message_id,
            text=error_text,
            parse_mode="HTML"
        )