from aiogram import Router, types
from aiogram.filters import CommandStart


router = Router()


@router.message(CommandStart())
async def send_welcome(message: types.Message):
    """Обработчик команды /start."""
    await message.reply(
        """Привет! Я бот-помощник для анализа финансовых данных.
Чтобы получить совет по компании, используй команды бота из меню (синее слева от строки ввода сообщений)."""
    )


@router.message()
async def echo(message: types.Message):
    """Обработчик всех остальных сообщений (эхо)."""
    try:
        # Отправляем копию полученного сообщения
        await message.send_copy(chat_id=message.chat.id)
    except TypeError:
        # Если тип сообщения не поддерживается для копирования
        await message.answer("Этот тип сообщения я не могу скопировать :(")
