from typing import List


# FEATURE_COLS: List[str] = ["Open", "High", "Low", "Close", "Volume", "Symbol"]
# FEATURE_COLS: List[str] = ["Open", "High", "Low", "Close", "Volume"]
FEATURE_COLS: List[str] = ["Open", "High", "Low", "Close", "Volume",
                           "return_1d", "return_1w", "return_2w", "return_1m", "return_3m", "return_6m", "return_1y",
                           "SMA_5", "SMA_10", "SMA_20", "SMA_50", "SMA_200",
                           "EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_200",
                           "MACD"]
BASE_TARGET_COL: str = "Close"
TARGET_COL: str = "target"
TARGET_HORIZONT: int = 7

TRAIN_END_DATE: str = "2024-09-01"
# VAL_END_DATE: str = "2025-01-01"
TEST_START_DATE: str = "2025-01-01"

# Список популярных компаний
US_DEFAULT_SYMBOLS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "META",   # Meta
    "NVDA",   # NVIDIA
    "TSLA",   # Tesla
    "JPM",    # JPMorgan Chase
    "V",      # Visa
    "WMT",    # Walmart
    "PG",     # Procter & Gamble
    "JNJ",    # Johnson & Johnson
    "MA",     # Mastercard
    "HD",     # Home Depot
    "BAC",    # Bank of America
    "XOM",    # Exxon Mobil
    "PFE",    # Pfizer
    "KO",     # Coca-Cola
    "PEP"     # PepsiCo
]

# Список популярных российских компаний
RU_DEFAULT_SYMBOLS = [
    "SBER",  # Сбербанк
    "GAZP",  # Газпром
    "LKOH",  # ЛУКОЙЛ
    "ROSN",  # Роснефть
    "MGNT",  # Магнит
    "VTBR",  # ВТБ
    "ALRS",  # РЖД
    "MTSS",  # МТС
    "YNDX",  # Яндекс
    "POLY",  # Полиметалл
    "AFLT",  # Аэрофлот
    "MOEX",  # Московская биржа
    "TATN",  # Татнефть
    "NVTK",  # НОВАТЭК
    "CHMF",  # Северсталь
    "GMKN",  # Норникель
    "SNGS",  # Сургутнефтегаз
    "IRAO",  # Интер РАО
    "PHOR",  # ФосАгро
    "RTKM"   # Ростелеком
]


RUS_PROD_TICKERS_12 = [
    "SBER",
    "LKOH",
    "ROSN",
    "NVTK",
    "GAZP",
    "SIBN",
    "PLZL",
    "GMKN",
    "YDEX",
    "TATN",
    "T",
    "PHOR",
    # "CHMF",
    # "X5",
]