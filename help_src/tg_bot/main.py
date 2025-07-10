from aiogram import Bot, Dispatcher
import asyncio
import logging
import sys
import os
from yandex_cloud_ml_sdk import YCloudML

sys.path.append(os.getcwd())

from help_src.tg_bot.middlewares import LoggingMiddleware
from help_src.tg_bot.config import config
from help_src.tg_bot.handlers import main_router
from help_src.tg_bot.logger import setup_logging, get_logger
from help_src.tg_bot.bot_commands import set_advice_menu
from market_oracle_lib.model import LSTMModel
from help_src.tg_bot.services.tsf_predict_service import load_catboost_models, load_scalers


logger = get_logger(__name__)

if config.tg_token is None:
    logging.critical("Не найден TG_TOKEN в переменных окружения!")
    sys.exit(1)

if config.t_token is None:
    logging.warning(
        "Не найден TINKOFF_TOKEN в переменных окружения! "
        "Функционал списка активов может не работать."
    )


# # If use LSTMModel
# checkpoint_path = os.path.join("./checkpoints", "my_test_checkpoint",
#                                "epoch=00-val_loss=172044.7344.ckpt")
# model = LSTMModel.load_from_checkpoint(
#     checkpoint_path,
#     input_size=23,
#     hidden_size=128,
#     output_size=1,
# )
# model.eval()
# model.to('cpu')

# If use CatBoostRegressor
models = load_catboost_models(models_dir_path="./checkpoints/catboosts")
scalers = load_scalers(scalers_dir_path="./checkpoints/scalers")

# Yandex Cloud ML
ya_sdk = YCloudML(
    folder_id=config.ya_folder_id,
    auth=config.ya_token,
)

# Инициализация бота и диспетчера
dp = Dispatcher(tinkoff_token=config.t_token)
bot = Bot(token=config.tg_token)

# Подключаем middleware
dp.message.middleware(LoggingMiddleware())

# Подключаем роутер с хэндлерами
dp.include_router(main_router)
dp["config"] = config
dp["tsf_model"] = None
dp["scalers"] = scalers
dp["models"] = models
dp["ya_sdk"] = ya_sdk

# Запуск бота
async def main() -> None:
    setup_logging()
    # Устанавливаем команды главного меню
    # await set_main_menu(bot)
    await set_advice_menu(bot)
    # Пропускаем ожидающие обновления
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
