from aiogram import Bot
from aiogram.types import BotCommand


async def set_main_menu(bot: Bot):
    commands = [
        BotCommand(
            command="/start",
            description="Запустить бота / Обновить меню"
        ),
        BotCommand(
            command="/assets_list",
            description="Показать список активов"
        ),
        BotCommand(
            command="/give_advice",
            description="Получить совет по компании"
        ),
    ]
    await bot.set_my_commands(commands)


async def set_advice_menu(bot: Bot):
    commands = [
        BotCommand(
            command="/start",
            description="Запустить бота / Обновить меню"
        ),
        BotCommand(
            command="/give_advice",
            description="Получить совет по компании"
        ),
        BotCommand(
            command="/pattern_advice",
            description="Получить паттерн-совет по компании"
        ),
        BotCommand(
            command="/semantic_advice",
            description="Получить семантический совет по компании"
        ),
        BotCommand(
            command="/forecast_advice",
            description="Получить прогнозный совет по компании"
        )
    ]
    await bot.set_my_commands(commands)
