import logging
from pathlib import Path


def setup_logging():
    """Настройка базовой конфигурации логирования"""
    logs_dir = Path("help_src/tg_bot/logs")
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{logs_dir}/bot_log.txt", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # Пример кастомной настройки для конкретных логгеров
    aiohttp_logger = logging.getLogger("aiohttp")
    aiohttp_logger.setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Получение логгера с указанным именем"""
    return logging.getLogger(name)
