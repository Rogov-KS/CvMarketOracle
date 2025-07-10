import torch
from torch.utils.data import Dataset, DataLoader
import investpy
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
from market_oracle_lib.dataloaders.dataset import StockDataset


# Список популярных российских компаний
DEFAULT_SYMBOLS = [
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

def get_symbol_data(symbol: str,
                    start_date: str = "01/01/2020",
                    end_date: str = "01/01/2024") -> pd.DataFrame:
                    # start_date: str = "2020-01-01",
                    # end_date: str = "2024-01-01") -> pd.DataFrame:
    """Получение данных для одного символа с MOEX."""
    try:
        # Преобразование дат в формат, который ожидает investpy
        # start = datetime.strptime(start_date, "%Y-%m-%d")
        # end = datetime.strptime(end_date, "%Y-%m-%d")

        # Получение данных
        df = investpy.get_stock_historical_data(
            stock=symbol,
            country='russia',
            from_date=start_date,
            to_date=end_date
        )

        # Добавляем символ как колонку
        df['Symbol'] = symbol

        return df
    except Exception as e:
        print(f"Ошибка при получении данных для {symbol}: {e}")
        return pd.DataFrame()

def get_multiple_symbols_data(
    symbols: List[str] = DEFAULT_SYMBOLS,
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01"
) -> pd.DataFrame:
    """Получение данных для множества символов."""
    all_data = []
    for symbol in symbols:
        df = get_symbol_data(symbol, start_date, end_date)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        raise ValueError("Не удалось получить данные ни для одного символа")

    combined_df = pd.concat(all_data)
    return combined_df

# def prepare_data(
#     df: pd.DataFrame,
#     sequence_length: int = 10,
#     target_column: str = "Close",
#     features: Optional[List[str]] = None
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Подготовка данных для модели."""
#     if features is None:
#         features = ["Open", "High", "Low", "Close", "Volume"]

#     # Нормализация данных
#     scaler = {}
#     for feature in features:
#         scaler[feature] = (df[feature].mean(), df[feature].std())
#         df[feature] = (df[feature] - scaler[feature][0]) / scaler[feature][1]

#     # Создание последовательностей
#     X, y = [], []
#     for symbol in df['Symbol'].unique():
#         symbol_data = df[df['Symbol'] == symbol]
#         for i in range(len(symbol_data) - sequence_length):
#             X.append(symbol_data[features].iloc[i:(i + sequence_length)].values)
#             y.append(symbol_data[target_column].iloc[i + sequence_length])

#     return np.array(X), np.array(y)

# def timeseries_split(
#     X: np.ndarray,
#     y: np.ndarray,
#     train_size: float = 0.8,
#     val_size: float = 0.1
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Разделение временных рядов на тренировочную, валидационную и тестовую выборки."""
#     # Для временных рядов важно сохранять порядок данных
#     train_idx = int(len(X) * train_size)
#     val_idx = int(len(X) * (train_size + val_size))

#     X_train = X[:train_idx]
#     y_train = y[:train_idx]

#     X_val = X[train_idx:val_idx]
#     y_val = y[train_idx:val_idx]

#     X_test = X[val_idx:]
#     y_test = y[val_idx:]

#     return X_train, y_train, X_val, y_val, X_test, y_test


# def create_data_loaders(
#     symbols: List[str] = DEFAULT_SYMBOLS,
#     sequence_length: int = 10,
#     batch_size: int = 32,
#     train_size: float = 0.8,
#     val_size: float = 0.1,
#     start_date: str = "2020-01-01",
#     end_date: str = "2024-01-01"
# ) -> Tuple[DataLoader, DataLoader, DataLoader]:
#     """Создание DataLoader'ов для обучения, валидации и тестирования."""
#     print("Creating data loaders...")
#     # Получение данных
#     df = get_multiple_symbols_data(symbols, start_date, end_date)
#     print("Data loaded successfully")
#     # Подготовка данных
#     X, y = prepare_data(df, sequence_length)
#     print("Data prepared successfully")
#     # Разделение данных
#     X_train, y_train, X_val, y_val, X_test, y_test = timeseries_split(
#         X, y, train_size, val_size
#     )
#     print("Data split successfully")
#     # Создание датасетов
#     train_dataset = StockDataset(X_train, y_train)
#     val_dataset = StockDataset(X_val, y_val)
#     test_dataset = StockDataset(X_test, y_test)
#     print("Data datasets created successfully")
#     # Создание DataLoader'ов
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=False  # Не перемешиваем данные для временных рядов
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False
#     )
#     print("Data loaders created successfully")

#     return train_loader, val_loader, test_loader

# if __name__ == "__main__":
#     train_loader, val_loader, test_loader = create_data_loaders()
#     print(train_loader)
#     print(val_loader)
#     print(test_loader)