# ================================
# const.py


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


# ===========================================================
# model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        learning_rate: float = 0.001,
        loss_fn: nn.Module = nn.MSELoss(),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # print("\n", lstm_out.shape)
        # predictions = self.fc(lstm_out[:, -1, :])
        predictions = self.fc(lstm_out)
        return predictions

    # def _step(self, batch, batch_idx, stage):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = nn.MSELoss()(y_hat, y)
    #     self.log(f"{stage}_loss", loss,
    #              prog_bar=True, on_step=True, on_epoch=True)
    #     return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(f"train_loss", loss,
                 prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Calculate real quantile - % of values where prediction was higher than actual
        higher_than_actual = (y_hat > y).float().mean()
        lower_than_actual = (y_hat < y).float().mean()

        self.log(f"{stage}_loss", loss,
                 prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_higher_ratio", higher_than_actual,
                 prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_lower_ratio", lower_than_actual,
                 prog_bar=True, on_step=True, on_epoch=True)

        return_dict = {
                       f"{stage}_loss": loss,
                       f"{stage}_higher_ratio": higher_than_actual,
                       f"{stage}_lower_ratio": lower_than_actual
                      }
        return return_dict

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# ========================================
# t_bank.py

import pandas as pd
from typing import Dict, Optional, List, Generator
from datetime import datetime, timedelta, timezone
from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.schemas import CandleSource
from tinkoff.invest.utils import quotation_to_decimal
from tinkoff.invest.constants import INVEST_GRPC_API
from tinkoff.invest.schemas import (
    GetTechAnalysisRequest,
    GetAssetFundamentalsRequest,
    IndicatorType,
    IndicatorInterval,
    TypeOfPrice,
)

# Словарь с именами индикаторов
indicator_names = {
    IndicatorType.INDICATOR_TYPE_SMA: "SMA",
    IndicatorType.INDICATOR_TYPE_EMA: "EMA",
    IndicatorType.INDICATOR_TYPE_RSI: "RSI",
    IndicatorType.INDICATOR_TYPE_MACD: "MACD",
    IndicatorType.INDICATOR_TYPE_BB: "Bollinger Bands"
}

interval_mapping: Dict[str, CandleInterval] = {
    "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
    "1d": CandleInterval.CANDLE_INTERVAL_DAY,
}


def _get_instrument_methods(client: Client):
    """Возвращает список методов для получения инструментов."""
    return [
        ('акции', client.instruments.shares),
        ('облигации', client.instruments.bonds),
        ('фонды', client.instruments.etfs),
        ('фьючерсы', client.instruments.futures),
        ('валюты', client.instruments.currencies)
    ]


def _get_all_assets(token: str,
                    is_rus: bool = True) -> Generator[str, None, None]:
    """
    Получает все доступные акции из API Тинькофф Инвестиций.
    """
    with Client(token, target=INVEST_GRPC_API) as client:
        assets = client.instruments.shares().instruments
        # return assets;
        for asset in assets:
            if not is_rus or asset.country_of_risk == "RU":
                yield asset


def find_instrument_by_ticker(ticker: str, token: str) -> Optional[Dict]:
    """
    Ищет инструмент по тикеру во всех доступных типах инструментов

    Args:
        ticker: Тикер инструмента
        token: Токен API Тинькофф

    Returns:
        Информация об инструменте или None, если инструмент не найден
    """
    with Client(token, target=INVEST_GRPC_API) as client:
        # Получаем список методов из новой функции
        instrument_methods = _get_instrument_methods(client)

        for instrument_type, method in instrument_methods:
            try:
                # Убедимся, что метод вызывается
                response = method()
                # Итерируемся по инструментам в ответе
                for instrument in response.instruments:
                    if instrument.ticker == ticker:
                        return {
                            'type': instrument_type,
                            'figi': instrument.figi,
                            'name': instrument.name,
                            'instrument': instrument
                        }
            except Exception as e:
                print(f"Ошибка при получении {instrument_type}: {e}")

        return None


def find_asset_by_ticker(ticker: str, token: str) -> Optional[Dict]:
    """
    Ищет инструмент по тикеру во всех доступных типах инструментов
    """
    with Client(token, target=INVEST_GRPC_API) as client:
        return find_instrument_by_ticker(ticker, token)


def get_dates_format(
    start_date: str | None, end_date: str | None
) -> tuple[datetime, datetime]:
    if start_date is None:
        # Устанавливаем очень раннюю дату по умолчанию
        start_date = datetime.fromisoformat("1991-01-01T00:00:00+00:00")
    if end_date is None:
        end_date = datetime.now(timezone.utc)

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    assert (isinstance(start_date, datetime) and isinstance(end_date, datetime)), "start_date and end_date must be datetime"

    return start_date, end_date


def get_symbol_data(
    symbol: str,
    token: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "1d",
    **kwargs
) -> pd.DataFrame:
    """
    Получение данных для одного символа с Tinkoff Invest API.

    Args:
        symbol: Тикер инструмента
        token: Токен API Тинькофф
        start_date: Дата начала. Либо datetime, либо str в формате "YYYY-MM-DD"
        end_date: Дата конца. Либо datetime, либо str в формате "YYYY-MM-DD"
        interval: Интервал
    """
    assert token is not None, "token is required"

    start_date, end_date = get_dates_format(start_date, end_date)

    try:
        # Сначала получаем информацию об инструменте по тикеру
        instrument_info = find_instrument_by_ticker(symbol, token)
        # print(f"instrument_info: {instrument_info}")

        if not instrument_info:
            print(f"Инструмент с тикером {symbol} не найден")
            return pd.DataFrame()

        figi = instrument_info['figi']

        with Client(token, target=INVEST_GRPC_API) as client:
            # Получаем исторические данные по FIGI
            data = []
            for candle in client.get_all_candles(
                instrument_id=figi,
                from_=start_date,
                to=end_date,
                interval=interval_mapping[interval],
                # Явно указываем источник свечей
                candle_source_type=CandleSource.CANDLE_SOURCE_UNSPECIFIED,
                **kwargs
            ):
                data.append({
                    'Date': candle.time,
                    'Open': float(quotation_to_decimal(candle.open)),
                    'High': float(quotation_to_decimal(candle.high)),
                    'Low': float(quotation_to_decimal(candle.low)),
                    'Close': float(quotation_to_decimal(candle.close)),
                    'Volume': candle.volume,
                })

            df = pd.DataFrame(data)
            df['Symbol'] = symbol
            return df
    except Exception as e:
        print(f"Ошибка при получении данных для {symbol}: {e}")
        return pd.DataFrame()


def get_indicators(
    instrument_info: Dict,
    token: str,
    indicator_type: IndicatorType = IndicatorType.INDICATOR_TYPE_SMA,
    interval: IndicatorInterval = IndicatorInterval.INDICATOR_INTERVAL_ONE_DAY,
    length: int = 20,
    days_back: int = 365
) -> List[Dict]:
    """
    Получение технических индикаторов для инструмента через API Тинькофф Инвестиций.

    Args:
        instrument_info: Информация об инструменте (полученная через find_instrument_by_ticker)
        token: Токен API Тинькофф
        indicator_type: Тип индикатора (по умолчанию SMA)
        interval: Интервал (по умолчанию дневной)
        length: Период для расчета индикатора (по умолчанию 20)
        days_back: Количество дней назад для получения данных (по умолчанию 365)

    Returns:
        Список словарей с техническими индикаторами
    """
    assert token is not None, "token is required"

    with Client(token) as client:
        market_data_service = client.market_data
        req = GetTechAnalysisRequest(
            indicator_type=indicator_type,
            instrument_uid=instrument_info["instrument"].uid,
            # Задаем период для индикатора
            from_=datetime.now(timezone.utc) - timedelta(days=days_back),
            to=datetime.now(timezone.utc),
            interval=interval,
            # Используем цену закрытия для индикаторов
            type_of_price=TypeOfPrice.TYPE_OF_PRICE_CLOSE,
            length=length,
        )
        res = market_data_service.get_tech_analysis(request=req)

        # Преобразуем результаты в список словарей
        indicators = []
        for indicator in res.technical_indicators:
            indicators.append({
                'time': indicator.timestamp,
                'value': float(quotation_to_decimal(indicator.signal)),
                'type': indicator_names[indicator_type]
            })

        return indicators


def get_asset_market_cap(asset_uid: str, token: str) -> float:
    """
    Получение рыночной капитализации актива по его asset_uid.

    Args:
        asset_uid: asset_uid актива
        token: Токен API Тинькофф

    Returns:
        Рыночная капитализация актива
    """
    with Client(token) as client:
        req = GetAssetFundamentalsRequest(assets=[asset_uid])
        response = client.instruments.get_asset_fundamentals(req)
        if len(response.fundamentals) != 1:
            return None
        return response.fundamentals[0].market_capitalization


def get_top_capitalized_assets(token: str,
                               n: int = 10,
                               is_rus: bool = True,
                               batch_size: int = 100,
                               do_return_all: bool = False) -> List[Dict]:
    """
    Получение списка топ-активов по рыночной капитализации.

    Args:
        token: Токен API Тинькофф
        is_rus: Флаг, указывающий, учитывать ли только российские активы
        n: Количество топ-активов для получения
        batch_size: Размер батча для обработки запросов

    Returns:
        Список словарей с информацией о топ-активах
    """
    assets_lst = list(_get_all_assets(token, is_rus))
    # return assets_lst
    result = []

    # Разбиваем список на батчи
    for i in range(0, len(assets_lst), batch_size):
    # for i in range(0, batch_size * 3, batch_size):
        batch = assets_lst[i:i + batch_size]
        batch_results = []

        for asset in batch:
            try:
                market_cap = get_asset_market_cap(asset_uid=asset.asset_uid, token=token)
                # print(f"market_cap: {market_cap} | asset: {asset.asset_uid}")
                if market_cap is not None:
                    batch_results.append((asset.ticker, asset.asset_uid, market_cap))
            except Exception as e:
                pass
                # print(f"Ошибка при получении капитализации для {asset.ticker}: {e}")

        result.extend(batch_results)

    # Сортируем по капитализации (второй элемент кортежа)
    result.sort(key=lambda x: x[2] if x[2] is not None else 0, reverse=True)

    if n > len(result):
        n = len(result)

    if do_return_all:
        return result
    else:
        return result[:n]



# ===========================================
# data_funcs.py


from copy import deepcopy
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional, Callable, Union, Literal, Dict
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# from market_oracle_lib.data.make_features import (
#     calculate_shifted_features,
#     calculate_joined_features,
#     add_shift_col_by_join,
#     calculate_time_features,
# )
from tqdm import tqdm
from IPython.display import clear_output, display
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from market_oracle_lib.consts import (
#     FEATURE_COLS,
#     BASE_TARGET_COL,
#     TARGET_COL,
#     US_DEFAULT_SYMBOLS,
#     TARGET_HORIZONT,
#     TRAIN_END_DATE,
#     TEST_START_DATE,
# )


# class StockDataset(Dataset):
#     """
#     Dataset for stock data.
#     """
#     def __init__(
#         self,
#         X: np.ndarray,
#         y: np.ndarray,
#     ):
#         self.X = torch.FloatTensor(X)
#         self.y = torch.FloatTensor(y)

#     def __len__(self) -> int:
#         return len(self.X)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         return self.X[idx], self.y[idx]


def prepare_data(df: pd.DataFrame,
                 feature_cols: Optional[List[str]] = FEATURE_COLS,
                 target_col: str = TARGET_COL,
                 window_size: int = 10,
                 calculate_additional_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Подготовка данных из DataFrame для обучения модели

    Args:
        df: DataFrame с данными
        feature_cols: Список колонок с признаками
        target_col: Название колонки с целевой переменной
        window_size: Размер окна для временного ряда
        calculate_additional_features: Рассчитывать ли дополнительные технические индикаторы

    Returns:
        X: DataFrame с признаками
        y: Series с целевыми значениями
    """
    # Выбираем только нужные колонки из DataFrame
    if feature_cols is None:
        feature_cols = ["Open", "High", "Low", "Close", "Volume", "Symbol"]

    # Если требуется, рассчитываем дополнительные признаки
    if calculate_additional_features:
        df = calculate_shifted_features(df)

    # Целевая переменная - обычно это доходность на следующий день (или период)
    # if target_col == 'return_1d' and 'return_1d' not in df.columns:
    #     df['return_1d'] = df['Close'].pct_change(1).shift(-1)  # Сдвигаем на один день вперед

    # Отбрасываем первые строки с NaN значениями
    df = df.dropna()

    # TODO: НАДО сделать one-hot encoding для Symbol
    df = df.drop(["Date", "Symbol"], axis=1)

    # # Исключаем целевую колонку из признаков
    # feature_cols = [col for col in df.columns if col != target_col]

    # X = df[feature_cols]
    # y = df[target_col]

    return df


def get_multiple_symbols_data(
    single_symbol_data_getter_fn: Callable,
    symbols: List[str] = US_DEFAULT_SYMBOLS,
    **kwargs
) -> pd.DataFrame:
    """Получение данных для множества символов."""
    all_data = []
    for symbol in tqdm(symbols):
        df = single_symbol_data_getter_fn(symbol, **kwargs)
        if not df.empty:
            all_data.append(df)
    clear_output()

    if not all_data:
        raise ValueError("Не удалось получить данные ни для одного символа")

    combined_df = pd.concat(all_data)
    return combined_df


def timeseries_split(df: pd.DataFrame,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделение временного ряда на обучающую, валидационную и тестовую выборки

    Args:
        df: DataFrame с данными
        train_ratio: Доля данных для обучения
        val_ratio: Доля данных для валидации

    Returns:
        train_df: DataFrame с данными для обучения
        val_df: DataFrame с данными для валидации
        test_df: DataFrame с данными для тестирования
    """
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    return train_df, val_df, test_df


def scale_col(df: pd.DataFrame,
              cols: List[str],
              scalers: Dict[str, Union[MinMaxScaler, StandardScaler]],
              prefix: str = "_scaled") -> pd.DataFrame:
    """
    Масштабирование колонки
    """
    df_copy = df.copy(deep=False)
    new_df = {}

    for symbol, group in df_copy.groupby("Symbol"):
        for col in cols:
            group[prefix + col] = scalers[symbol].transform(group[col].values.reshape(-1, 1)).flatten()

        group['Symbol'] = symbol
        new_df[symbol] = group

    return pd.concat(new_df.values())


def add_scale_target(df: pd.DataFrame,
                 target_scaler: Literal["min-max", "standard"] = "standard",
                 target_col: str = TARGET_COL,
                 new_target_name: str = "scaled_target",
                 train_end_date: str = TRAIN_END_DATE) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Union[MinMaxScaler, StandardScaler]]]:
    """
    Масштабирование целевой переменной для каждого символа отдельно
    """
    if "Symbol" not in df.columns:
        raise ValueError("Столбец 'Symbol' не найден в DataFrame")

    scalers = {}
    train_end_date_dt = pd.Timestamp(train_end_date, tz='utc')
    df_copy = df.copy(deep=False)
    new_df = {}

    # def scaller_aply_func(group_df):
    #     display(group_df)
    #     train_df = group_df[group_df["Date"] <= train_end_date_dt]

    #     scaler.fit(train_df[target_col].values.reshape(-1, 1))

    #     scaled_values = scaler.transform(group_df[target_col].values.reshape(-1, 1)).flatten()

    #     return scaled_values

    # df_copy[new_target_name] = df_copy.groupby("Symbol").apply(scaller_aply_func)

    for symbol, group in df_copy.groupby("Symbol"):
        if target_scaler == "min-max":
            scaler = MinMaxScaler()
        elif target_scaler == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Неизвестный тип скейлера: {target_scaler}")

        # display(group)
        train_df = group[group["Date"] <= train_end_date_dt]

        # check for min_train_size
        if len(train_df) < 100:
            continue

        # display(train_df)

        if train_df.empty:
            continue

        scaler.fit(train_df[target_col].values.reshape(-1, 1))
        group[new_target_name] = scaler.transform(group[target_col].values.reshape(-1, 1)).flatten()
        # display(group)

        group['Symbol'] = symbol

        scalers[symbol] = scaler
        new_df[symbol] = group.sort_values(by=["Date"])

    new_df = pd.concat(new_df.values())
    return new_df, scalers


# def add_scaled_target(df: pd.DataFrame,
#                       target_scaler: Literal["min-max", "standard"] = "standard",
#                       target_col: str = TARGET_COL,
#                       new_target_name: str = "scaled_target") -> Tuple[pd.DataFrame, Dict[str, Union[MinMaxScaler, StandardScaler]]]:
#     """
#     Масштабирование целевой переменной для каждого символа отдельно
#     с использованием groupby
#     """
#     if "Symbol" not in df.columns:
#         raise ValueError("Столбец 'Symbol' не найден в DataFrame")

#     scalers = {}

#     # Используем groupby для обработки каждого символа
#     for symbol, group in df.groupby("Symbol"):
#         if target_scaler == "min-max":
#             scaler = MinMaxScaler()
#         elif target_scaler == "standard":
#             scaler = StandardScaler()
#         else:
#             raise ValueError(f"Неизвестный тип скейлера: {target_scaler}")

#         values = group[target_col].values.reshape(-1, 1)
#         scaler.fit(values)
#         scalers[symbol] = scaler

#     def scaler_func(x):
#         nonlocal scalers
#         scaler = scalers[x.name]
#         return scaler.transform(x.values.reshape(-1, 1)).flatten()

#     df[new_target_name] = df.groupby("Symbol")[target_col].transform(scaler_func)

#     return df, scalers


# def add_shift_col_by_join(df: pd.DataFrame,
#                           targets_horizont: int = TARGET_HORIZONT,
#                           target_col: str = TARGET_COL,
#                           new_target_name: str = TARGET_COL,
#                           do_get_diff_target: bool = False) -> pd.DataFrame:
#     """
#     Возращает pd.DataFrame с целевой переменной, сдвинутой на заданное количество дней вперёд (если targets_horizont > 0)
#     или назад (если targets_horizont < 0)
#     Если do_get_diff_target = True, то возращается также разность между сдвинутой и исходной целевыми переменными
#     """
#     df_copy = df.copy(deep=True)
#     day_offset = pd.DateOffset(days=targets_horizont)
#     df_copy['Date_shifted'] = df_copy['Date'] - day_offset

#     suffix = '_shifted'
#     merged = pd.merge(df_copy, df_copy,
#                       left_on=['Symbol', 'Date'],
#                       right_on=['Symbol', 'Date_shifted'],
#                       how='left', suffixes=('', suffix))

#     merged[new_target_name] = merged[target_col + suffix]
#     merged = merged[df_copy.columns.tolist() + [new_target_name]]
#     merged.drop(['Date_shifted'], axis=1, inplace=True)

#     # display(merged.tail(30))

#     if do_get_diff_target:
#         merged[f"diff_{new_target_name}"] = merged[new_target_name] - merged[target_col]
#         return merged

#     else:
#         return merged


def create_base_data_frames(data_getter_fn: Callable,
                            saving_cols: List[str] | None = None,
                            **kwargs) -> pd.DataFrame:
    """
    Создание базовых DataFrames для обучения модели
    Только данные про свечи
    """
    df = get_multiple_symbols_data(data_getter_fn, **kwargs)

    if saving_cols is not None:
        df = df[saving_cols]

    df = df.sort_values(by=["Date"])
    return df


def create_base_data_frames_as_dict(data_getter_fn: Callable,
                                    **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Создание базовых DataFrames для обучения модели
    Только данные про свечи
    """
    df = get_multiple_symbols_data(data_getter_fn, **kwargs)
    # df = df.sort_values(by=["Symbol", "Date"])
    df = df.sort_values(by=["Symbol", "Date"])
    res_dict = dict(list(df.groupby('Symbol')))
    return res_dict


def create_featured_data_frame(data_getter_fn: Callable,
                               target_scaler: Literal["min-max", "standard"] = "standard",
                               base_target_col: str = BASE_TARGET_COL,
                               interval: Literal["1d", "1h", "1m"] = "1d",
                               symbols_list: List[str] = ["SBER", "YDEX", "T"],
                               t_token: str = None,
                               scalers: Union[MinMaxScaler, StandardScaler, None] = None,
                               cols_for_scale: List[str] | None = None,
                               do_drop_na: bool = True,
                               target_horizont: int = TARGET_HORIZONT,
                               **kwargs) -> Dict[str, pd.DataFrame]:
    """
    """
    df1 = create_base_data_frames(
        data_getter_fn,
        interval=interval,
        token=t_token,
        # saving_cols=['Date', 'Symbol', 'Volume', BASE_TARGET_COL],
        symbols=symbols_list
        # symbols=["YDEX", "SBER", "T"]
    )

    if scalers is None:
        base_scale_df, scalers = add_scale_target(df1, target_scaler=target_scaler,
                                                  target_col=base_target_col,
                                                  new_target_name=f"scaled_{base_target_col}",
                                                  train_end_date=TRAIN_END_DATE)
    else:
        base_scale_df = df1

    if cols_for_scale is None:
        cols_for_scale = ['Open', 'High', 'Low']

    scaled_df = scale_col(base_scale_df,
                          cols=cols_for_scale,
                          scalers=scalers,
                          prefix='scaled_')



    base_target_df = add_shift_col_by_join(scaled_df,
                                        target_col=f"scaled_{base_target_col}",
                                        do_get_diff_target=True,
                                        target_horizont=target_horizont)


    shifted_features_df = calculate_shifted_features(base_target_df,
                                                    target_col=f"scaled_{base_target_col}")


    date_shifts = [
                pd.DateOffset(days=7),
                pd.DateOffset(days=14),
                pd.DateOffset(days=21),
                pd.DateOffset(days=28),
                pd.DateOffset(months=1),
                ]

    joined_features_df = calculate_joined_features(shifted_features_df,
                                                   target_col=f"scaled_{base_target_col}",
                                                   date_shifts=date_shifts,
                                                   do_add_diff_target=True)

    time_features_df = calculate_time_features(joined_features_df)

    # display(time_features_df.isna().sum().tail(30))

    if do_drop_na:
        time_features_df.dropna(inplace=True)

    # check for

    return time_features_df, scalers


def split_df_by_dates(df: pd.DataFrame,
                      train_start_date: str = None,
                      train_end_date: str = TRAIN_END_DATE,
                      test_start_date: str = TEST_START_DATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделение DataFrame на обучающую, валидационную и тестовую выборки
    """
    if train_start_date is None:
        train_df = df[df['Date'] <= train_end_date]
    else:
        train_df = df[(df['Date'] >= train_start_date) & (df['Date'] <= train_end_date)]

    val_df = df[(train_end_date < df['Date']) & (df['Date'] <= test_start_date)]
    test_df = df[test_start_date < df['Date']]

    return train_df, val_df, test_df

def create_X_y_df_from_df(df: pd.DataFrame,
                          feature_cols: List[str] = None,
                          target_col: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Создание DataFrame с признаками и целевой переменной
    """
    if target_col in df.columns:
        y = df[target_col]
    else:
        y = None
    X = df.copy(deep=True)
    X.drop(['diff_target', 'target', 'Date', 'Symbol',
            'Open', 'High', 'Low', 'Close',
            'scaled_Open', 'scaled_High', 'scaled_Low',],
            axis=1, inplace=True)
    if feature_cols is not None:
        X = X[feature_cols]

    return X, y

def create_grouped_X_y_df_from_df(df: pd.DataFrame,
                          feature_cols: List[str] = None,
                          target_col: str = TARGET_COL,
                          group_cols: List[str] = ['Symbol']) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Создание DataFrame с признаками и целевой переменной
    """
    res_dict = {}
    for symbol, group in df.groupby(group_cols):
        X, y = create_X_y_df_from_df(group, feature_cols, target_col)
        res_dict[symbol[0]] = X, y

    return res_dict


def create_data_frames(data_getter_fn: Callable,
                       **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Создание DataLoader для обучения модели

    Args:
        X_train, X_val, X_test: DataFrame с признаками
        y_train, y_val, y_test: Series с целевыми значениями
        batch_size: Размер батча

    Returns:
        train_loader, val_loader, test_loader: DataLoader для обучения, валидации и тестирования
    """
    # Преобразуем DataFrame в тензоры PyTorch
    df = get_multiple_symbols_data(data_getter_fn, **kwargs)

    new_df = prepare_data(df)

    train_df, val_df, test_df = timeseries_split(new_df)
    print("Data dataframes created successfully")
    print(f"{train_df.shape=} | {val_df.shape=} | {test_df.shape=}")
    print(f"{train_df.columns=} | {val_df.columns=} | {test_df.columns=}")

    return train_df, val_df, test_df


# def create_data_sets(
#     data_getter_fn: Callable,
#     feature_cols: Optional[List[str]] = FEATURE_COLS,
#     target_col: str = TARGET_COL,
#     **kwargs
# ) -> Tuple[StockDataset, StockDataset, StockDataset]:
#     """Создание DataLoader'ов для обучения, валидации и тестирования."""
#     # Получение данных
#     train_df, val_df, test_df = create_data_frames(data_getter_fn, **kwargs)

#     print(f"{train_df.shape=} | {val_df.shape=} | {test_df.shape=}")
#     print(f"{train_df.head()=} | {val_df.head()=} | {test_df.head()=}")

#     # Преобразуем DataFrame в тензоры PyTorch
#     X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
#     X_val, y_val = val_df[feature_cols].values, val_df[target_col].values
#     X_test, y_test = test_df[feature_cols].values, test_df[target_col].values


#     # Создание датасетов
#     train_dataset = StockDataset(X_train, y_train)
#     val_dataset = StockDataset(X_val, y_val)
#     test_dataset = StockDataset(X_test, y_test)
#     print("Data datasets created successfully")

#     return train_dataset, val_dataset, test_dataset


def create_data_loaders(data_getter_fn: Callable,
                        feature_cols: Optional[List[str]] = FEATURE_COLS,
                        target_col: str = TARGET_COL,
                        batch_size: int = 32,
                        **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создание DataLoader для обучения модели

    Args:
        X_train, X_val, X_test: DataFrame с признаками
        y_train, y_val, y_test: Series с целевыми значениями
        batch_size: Размер батча

    Returns:
        train_loader, val_loader, test_loader: DataLoader для обучения, валидации и тестирования
    """
    train_ds, val_ds, test_ds = create_data_sets(data_getter_fn,
                                                 target_col=target_col,
                                                 feature_cols=feature_cols,
                                                 **kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False  # Не перемешиваем данные для временных рядов
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )
    print("Data loaders created successfully")

    return train_loader, val_loader, test_loader


def comprehensive_stationarity_test(timeseries, title):
    """
    Комплексная проверка стационарности временного ряда с помощью различных тестов
    """
    print(f'=== Результаты тестов стационарности для {title} ===')

    # Убираем NaN значения
    ts_clean = timeseries.dropna()

    if len(ts_clean) < 10:
        print("Недостаточно данных для проведения тестов")
        return

    # 1. Расширенный тест Дики-Фуллера (ADF)
    print("\n1. Расширенный тест Дики-Фуллера (ADF):")
    try:
        adf_result = adfuller(ts_clean, autolag='AIC')
        print(f"   Статистика теста: {adf_result[0]:.6f}")
        print(f"   p-значение: {adf_result[1]:.6f}")
        print(f"   Критические значения:")
        for key, value in adf_result[4].items():
            print(f"     {key}: {value:.6f}")

        if adf_result[1] <= 0.05:
            print("   Результат: Ряд стационарен (отвергаем H0)")
        else:
            print("   Результат: Ряд нестационарен (принимаем H0)")
    except Exception as e:
        print(f"   Ошибка при выполнении ADF теста: {e}")

    # 2. Тест KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
    print("\n2. Тест KPSS:")
    try:
        kpss_result = kpss(ts_clean, regression='c', nlags='auto')
        print(f"   Статистика теста: {kpss_result[0]:.6f}")
        print(f"   p-значение: {kpss_result[1]:.6f}")
        print(f"   Критические значения:")
        for key, value in kpss_result[3].items():
            print(f"     {key}: {value:.6f}")

        if kpss_result[1] >= 0.05:
            print("   Результат: Ряд стационарен (принимаем H0)")
        else:
            print("   Результат: Ряд нестационарен (отвергаем H0)")
    except Exception as e:
        print(f"   Ошибка при выполнении KPSS теста: {e}")

    # 3. Тест Филлипса-Перрона (упрощенная версия через ADF с разными лагами)
    print("\n3. Модифицированный тест Дики-Фуллера (без автолагов):")
    try:
        pp_result = adfuller(ts_clean, maxlag=1, autolag=None)
        print(f"   Статистика теста: {pp_result[0]:.6f}")
        print(f"   p-значение: {pp_result[1]:.6f}")

        if pp_result[1] <= 0.05:
            print("   Результат: Ряд стационарен (отвергаем H0)")
        else:
            print("   Результат: Ряд нестационарен (принимаем H0)")
    except Exception as e:
        print(f"   Ошибка при выполнении модифицированного теста: {e}")

    # 4. Тест на нормальность остатков (Jarque-Bera)
    print("\n4. Тест Jarque-Bera на нормальность:")
    try:
        jb_stat, jb_pvalue = stats.jarque_bera(ts_clean)
        print(f"   Статистика теста: {jb_stat:.6f}")
        print(f"   p-значение: {jb_pvalue:.6f}")

        if jb_pvalue >= 0.05:
            print("   Результат: Данные имеют нормальное распределение")
        else:
            print("   Результат: Данные не имеют нормального распределения")
    except Exception as e:
        print(f"   Ошибка при выполнении теста Jarque-Bera: {e}")

    # 5. Тест Льюнга-Бокса на автокорреляцию
    print("\n5. Тест на автокорреляцию (упрощенный):")
    try:
        # Вычисляем автокорреляцию первого порядка
        if len(ts_clean) > 1:
            autocorr = np.corrcoef(ts_clean[:-1], ts_clean[1:])[0, 1]
            print(f"   Автокорреляция 1-го порядка: {autocorr:.6f}")

            if abs(autocorr) < 0.1:
                print("   Результат: Слабая автокорреляция (признак стационарности)")
            else:
                print("   Результат: Сильная автокорреляция (признак нестационарности)")
    except Exception as e:
        print(f"   Ошибка при вычислении автокорреляции: {e}")

    # 6. Проверка постоянства дисперсии
    print("\n6. Анализ дисперсии по периодам:")
    try:
        # Разделяем ряд на две части и сравниваем дисперсии
        mid_point = len(ts_clean) // 2
        first_half = ts_clean[:mid_point]
        second_half = ts_clean[mid_point:]

        var1 = np.var(first_half)
        var2 = np.var(second_half)

        print(f"   Дисперсия первой половины: {var1:.6f}")
        print(f"   Дисперсия второй половины: {var2:.6f}")
        print(f"   Отношение дисперсий: {var2/var1:.6f}")

        # F-тест для сравнения дисперсий
        f_stat = var2 / var1 if var2 > var1 else var1 / var2
        f_pvalue = 2 * (1 - stats.f.cdf(f_stat, len(second_half)-1, len(first_half)-1))

        print(f"   F-статистика: {f_stat:.6f}")
        print(f"   p-значение F-теста: {f_pvalue:.6f}")

        if f_pvalue >= 0.05:
            print("   Результат: Дисперсия постоянна (признак стационарности)")
        else:
            print("   Результат: Дисперсия непостоянна (признак нестационарности)")
    except Exception as e:
        print(f"   Ошибка при анализе дисперсии: {e}")

    print('='*60)



# ================================================
# make_features



import pandas as pd
import numpy as np
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset, Day, Easter
from datetime import datetime, timedelta
from typing import List

# from market_oracle_lib.consts import (
#     BASE_TARGET_COL,
#     TARGET_HORIZONT,
#     TARGET_COL
# )


def calculate_shifted_features(df: pd.DataFrame,
                               target_col: str = BASE_TARGET_COL,
                               prefix: str = "scaled_") -> pd.DataFrame:
    """
    Вычисление финансовых признаков на основе временного ряда цен акций.

    Args:
        df: DataFrame с данными цен акций (должен содержать колонки: Open, High, Low, Close, Volume)

    Returns:
        DataFrame с добавленными техническими индикаторами
    """
    # Создаем копию DataFrame, чтобы не изменять оригинал
    result = df.copy()

    # --- Расчет изменений за период времени ---

    # Изменение цены за различные периоды (в абсолюте)
    result['diff_shift_1d'] = result[target_col].diff(1)
    result['diff_shift_2d'] = result[target_col].diff(2)
    result['diff_shift_3d'] = result[target_col].diff(3)
    result['diff_shift_4d'] = result[target_col].diff(4)
    result['diff_shift_5d'] = result[target_col].diff(5)  # ~1 неделя (5 торговых дней)
    result['diff_shift_10d'] = result[target_col].diff(10)  # ~2 недели
    result['diff_shift_21d'] = result[target_col].diff(21)  # ~1 месяц
    result['diff_shift_63d'] = result[target_col].diff(63)  # ~3 месяца
    result['diff_shift_126d'] = result[target_col].diff(126)  # ~6 месяцев
    result['diff_shift_189d'] = result[target_col].diff(189)  # ~9 месяцев
    result['diff_shift_252d'] = result[target_col].diff(252)  # ~1 год

    # Изменение цены за различные периоды (в %)
    result['pct_diff_shift_1d'] = result[target_col].pct_change(1)
    result['pct_diff_shift_2d'] = result[target_col].pct_change(2)
    result['pct_diff_shift_3d'] = result[target_col].pct_change(3)
    result['pct_diff_shift_4d'] = result[target_col].pct_change(4)
    result['pct_diff_shift_5d'] = result[target_col].pct_change(5)  # ~1 неделя (5 торговых дней)
    result['pct_diff_shift_10d'] = result[target_col].pct_change(10)  # ~2 недели
    result['pct_diff_shift_21d'] = result[target_col].pct_change(21)  # ~1 месяц
    result['pct_diff_shift_63d'] = result[target_col].pct_change(63)  # ~3 месяца
    result['pct_diff_shift_126d'] = result[target_col].pct_change(126)  # ~6 месяцев
    result['pct_diff_shift_189d'] = result[target_col].pct_change(189)  # ~9 месяцев
    result['pct_diff_shift_252d'] = result[target_col].pct_change(252)  # ~1 год

    # --- Скользящие средние (MA) ---

    # Простые скользящие средние (SMA)
    result['SMA_5'] = result[target_col].rolling(window=5).mean()
    result['SMA_10'] = result[target_col].rolling(window=10).mean()
    result['SMA_20'] = result[target_col].rolling(window=20).mean()
    result['SMA_50'] = result[target_col].rolling(window=50).mean()
    result['SMA_200'] = result[target_col].rolling(window=200).mean()

    # Экспоненциальные скользящие средние (EMA)
    result['EMA_5'] = result[target_col].ewm(span=5, adjust=False).mean()
    result['EMA_10'] = result[target_col].ewm(span=10, adjust=False).mean()
    result['EMA_20'] = result[target_col].ewm(span=20, adjust=False).mean()
    result['EMA_50'] = result[target_col].ewm(span=50, adjust=False).mean()
    result['EMA_200'] = result[target_col].ewm(span=200, adjust=False).mean()

    # --- Осцилляторы ---

    # RSI (Relative Strength Index)
    delta = result[target_col].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # avg_gain_14 = gain.rolling(window=14).mean()
    # avg_loss_14 = loss.rolling(window=14).mean()
    avg_gain_14 = gain.ewm(span=14, adjust=False).mean()
    avg_loss_14 = loss.ewm(span=14, adjust=False).mean()

    rs_14 = avg_gain_14 / avg_loss_14
    result['RSI_14'] = 100 - (100 / (1 + rs_14))

    # MACD (Moving Average Convergence Divergence)
    result['MACD'] = result['EMA_10'] - result['EMA_20']
    result['MACD_signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    result['MACD_hist'] = result['MACD'] - result['MACD_signal']

    # Bollinger Bands
    result['BB_middle'] = result[target_col].rolling(window=20).mean()
    result['BB_std'] = result[target_col].rolling(window=20).std()
    result['BB_upper'] = result['BB_middle'] + 2 * result['BB_std']
    result['BB_lower'] = result['BB_middle'] - 2 * result['BB_std']
    result['BB_width'] = (result['BB_upper'] - result['BB_lower']) / result['BB_middle']

    # # OBV (On-Balance Volume)
    # result['OBV'] = np.where(
    #     result[target_col] > result[target_col].shift(1),
    #     result['Volume'],
    #     np.where(
    #         result[target_col] < result[target_col].shift(1),
    #         -result['Volume'],
    #         0
    #     )
    # ).cumsum()

    # ATR (Average True Range)
    result['TR'] = np.maximum(
        np.maximum(
            result[f'{prefix}High'] - result[f'{prefix}Low'],
            np.abs(result[f'{prefix}High'] - result[f'{prefix}Close'].shift(1))
        ),
        np.abs(result[f'{prefix}Low'] - result[f'{prefix}Close'].shift(1))
    )
    result['ATR_14'] = result['TR'].rolling(window=14).mean()

    # Stochastic Oscillator
    result['Stoch_%K'] = 100 * ((result[f'{prefix}Close'] - result[f'{prefix}Low'].rolling(window=14).min()) /
                               (result[f'{prefix}High'].rolling(window=14).max() - result[f'{prefix}Low'].rolling(window=14).min()))
    result['Stoch_%D'] = result['Stoch_%K'].rolling(window=3).mean()

    # --- Уровни Фибоначчи ---
    # Для расчета уровней Фибоначчи находим последние максимум и минимум
    # Обычно их вычисляют для определенных периодов
    # Здесь возьмем период в 100 дней

    for window in [50, 100, 200]:
        roll_max = result[f'{prefix}High'].rolling(window=window).max()
        roll_min = result[f'{prefix}Low'].rolling(window=window).min()

        # Уровни Фибоначчи
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]

        for level in fib_levels:
            level_name = f'Fib_{level*100:.0f}_{window}d'
            result[level_name] = roll_min + (roll_max - roll_min) * level

        # Относительная позиция текущей цены между мин и макс
        range_size = roll_max - roll_min
        result[f'Price_position_{window}d'] = (result[f'{prefix}Close'] - roll_min) / range_size.replace(0, np.nan)

    # --- Волатильность ---

    # Вычисляем историческую волатильность
    result['volatility_5d'] = result['pct_diff_shift_1d'].rolling(window=5).std() * np.sqrt(252)
    result['volatility_10d'] = result['pct_diff_shift_1d'].rolling(window=10).std() * np.sqrt(252)
    result['volatility_20d'] = result['pct_diff_shift_1d'].rolling(window=20).std() * np.sqrt(252)

    # Удаляем NaN значения или заменяем их на 0
    # result = result.fillna(0)

    return result


def add_shift_col_by_join(df: pd.DataFrame,
                          target_col: str = BASE_TARGET_COL,
                          new_target_name: str = TARGET_COL,
                          target_horizont: int = TARGET_HORIZONT,
                          do_get_diff_target: bool = False) -> pd.DataFrame:
    """
    Возращает pd.DataFrame с целевой переменной, сдвинутой на заданное количество дней вперёд (если targets_horizont > 0)
    или назад (если targets_horizont < 0)
    Если do_get_diff_target = True, то возращается также разность между сдвинутой и исходной целевыми переменными
    """
    df_copy = df.copy(deep=True)
    day_offset = pd.DateOffset(days=target_horizont)
    df_copy['Date_shifted'] = df_copy['Date'] - day_offset

    suffix = '_shifted'
    merged = pd.merge(df_copy, df_copy,
                      left_on=['Symbol', 'Date'],
                      right_on=['Symbol', 'Date_shifted'],
                      how='left', suffixes=('', suffix))

    merged[new_target_name] = merged[target_col + suffix]
    merged = merged[df_copy.columns.tolist() + [new_target_name]]
    merged.drop(['Date_shifted'], axis=1, inplace=True)

    # display(merged.tail(30))

    if do_get_diff_target:
        merged[f"diff_{new_target_name}"] = merged[new_target_name] - merged[target_col]
        return merged

    else:
        return merged


def fill_missing_dates(df,
                       target_cols = [TARGET_COL],
                       freq='D'):
    # Убедимся, что Date — это datetime
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Для каждого Symbol создаём полный индекс по датам
    filled = []
    for symbol, group in df.groupby('Symbol'):
        # Получим полный диапазон дат

        all_dates = pd.date_range(group['Date'].min(), group['Date'].max(),
                                  freq=freq, tz=group['Date'].dt.tz)

        group = group.set_index('Date')
        original_dates = set(group.index)
        # reindex по всем датам, добавляем столбец Symbol
        group = group.reindex(all_dates)
        group['Symbol'] = symbol
        # Заполняем target_col ближайшим предыдущим значением
        for target_col in target_cols:
            group[target_col] = group[target_col].ffill()

        group['was_added'] = ~group.index.isin(original_dates)

        filled.append(group)

    # Собираем обратно
    result = pd.concat(filled).reset_index().rename(columns={'index': 'Date'})
    # Если нужно, сортируем
    result = result.sort_values(['Date', 'Symbol']).reset_index(drop=True)
    return result


def calculate_joined_features(df: pd.DataFrame,
                              target_col: str = BASE_TARGET_COL,
                              prefix: str = "scaled_",
                              group_cols: List[str] = ["Symbol"],
                              date_col: str = "Date",
                              do_add_diff_target: bool = False,
                              date_shifts: List[pd.DateOffset] = []) -> pd.DataFrame:
    """
    Вычисление финансовых признаков на основе временного ряда цен акций.

    Args:
        df: DataFrame с данными цен акций (должен содержать колонки: Open, High, Low, Close, Volume)
    """

    filled_date_df = fill_missing_dates(df,
                                        target_cols=[target_col])
                                        # target_cols=['scaled_Close', 'Open', 'High', 'Low', 'Volume'])


    # TODO: добавить сдвиг целевой переменной на заданные даты
    result_df = filled_date_df.copy(deep=True)

    for date_shift in date_shifts:
        # Создаем временный датафрейм со сдвинутыми датами
        lagged_df = filled_date_df[group_cols + [date_col, target_col]].copy(deep=True)
        lagged_df[date_col] = lagged_df[date_col] + date_shift

        # display(lagged_df.tail(30))

        # Переименовываем value_col в lagged_value
        lagged_col_name = f"{target_col}_lag_{date_shift}"
        lagged_df = lagged_df.rename(columns={target_col: lagged_col_name})

        # Делаем merge только по group_col и date_col
        result_df = pd.merge(
            result_df,
            lagged_df,
            on=group_cols+[date_col],
            how="left",  # left join, чтобы сохранить все исходные строки
        )

        if do_add_diff_target:
            result_df[f"diff_{lagged_col_name}"] = result_df[lagged_col_name] - result_df[target_col]

    result_df = result_df[~result_df['was_added']]
    result_df.drop(columns=['was_added'], inplace=True)
    return result_df


def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисление временных признаков на основе даты и времени.
    """
    result_df = df.copy(deep=True)

    result_df['day_of_week'] = result_df['Date'].dt.dayofweek
    result_df['day_of_month'] = result_df['Date'].dt.day
    result_df['day_of_year'] = result_df['Date'].dt.dayofyear
    result_df['week_of_year'] = result_df['Date'].dt.isocalendar().week
    # result_df['week_of_month'] = result_df['Date'].dt.week

    result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]) * 1
    # Дни до конца месяца
    result_df['days_to_month_end'] = result_df['Date'].dt.daysinmonth - result_df['day_of_month']
    # Дни с начала месяца
    result_df['days_from_month_start'] = result_df['day_of_month'] - 1
    result_df['month_size'] = result_df['Date'].dt.daysinmonth

    # Дни до конца квартала (приблизительно)
    result_df['days_to_quarter_end'] = 13 * 7 - (result_df['week_of_year'] % 13) * 7 - result_df['Date'].dt.dayofweek
    # Дни с начала квартала (приблизительно)
    result_df['days_from_quarter_start'] = (result_df['week_of_year'] % 13) * 7 + result_df['Date'].dt.dayofweek

    # Дни до конца года
    result_df['days_to_year_end'] = 365 - result_df['day_of_year']
    # Дни с начала года
    result_df['days_from_year_start'] = result_df['day_of_year'] - 1

    # Добавление признаков, связанных с праздниками
    class RussianHolidayCalendar(AbstractHolidayCalendar):
        """
        Календарь основных российских праздников
        """
        rules = [
            Holiday('Новый год', month=1, day=1),
            Holiday('Рождество', month=1, day=7),
            Holiday('День защитника Отечества', month=2, day=23),
            Holiday('Международный женский день', month=3, day=8),
            Holiday('Праздник Весны и Труда', month=5, day=1),
            Holiday('День Победы', month=5, day=9),
            Holiday('День России', month=6, day=12),
            Holiday('День народного единства', month=11, day=4),
        ]

    # Получаем список праздников для диапазона дат в датафрейме
    min_date = result_df['Date'].min()
    max_date = result_df['Date'].max()

    calendar = RussianHolidayCalendar()
    holidays = calendar.holidays(start=min_date, end=max_date)

    # Создаем признаки
    result_df['is_holiday'] = result_df['Date'].isin(holidays) * 1

    # Добавляем отдельные фичи для конкретных праздников
    def days_to_specific_holiday(date, month, day):
        """Вычисляет количество дней до конкретного праздника"""
        holiday_date = datetime(date.year, month, day).date()
        if date.date() > holiday_date:
            holiday_date = datetime(date.year + 1, month, day).date()
        return (holiday_date - date.date()).days

    # Дни до Нового года
    result_df['days_to_new_year'] = result_df['Date'].apply(
        lambda x: days_to_specific_holiday(x, 1, 1)
    )

    # Дни до Дня Победы
    result_df['days_to_victory_day'] = result_df['Date'].apply(
        lambda x: days_to_specific_holiday(x, 5, 9)
    )

    # Дни до Дня России
    result_df['days_to_russia_day'] = result_df['Date'].apply(
        lambda x: days_to_specific_holiday(x, 6, 12)
    )


    return result_df



# ================================
# tsf_predict_service.py


# from market_oracle_lib.data import data_funcs
# from market_oracle_lib.data import t_bank
# from market_oracle_lib.model import LSTMModel
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import os
from IPython.display import clear_output
import pickle
from IPython.display import display

from pytorch_lightning.loggers import WandbLogger


def make_prod_predict_by_cbt(ticker: str,
                      loaded_models: dict[str, CatBoostRegressor],
                      saved_scalers: dict[str, MinMaxScaler | StandardScaler],
                      t_token: str,
                      interval: str = "1d",
                      is_diff_target: bool = False):
    data_df, _ = data_funcs.create_featured_data_frame(t_bank.get_symbol_data,
                                                interval=interval,
                                                token=t_token,
                                                symbols_list=[ticker],
                                                t_token=t_token,
                                                scalers=saved_scalers,
                                                cols_for_scale=['Close', 'Open', 'High', 'Low'],
                                                do_drop_na=False)

    last_row = data_df.iloc[[-1]]
    input_row, _ = data_funcs.create_X_y_df_from_df(last_row, target_col='target')
    pred = loaded_models[ticker].predict(input_row)
    if is_diff_target:
        # display(last_row)
        pred = pred + last_row.iloc[0]['scaled_Close']
    scaled_pred = saved_scalers[ticker].inverse_transform(pred.reshape(-1, 1))[0, 0].item()
    return scaled_pred


def load_lstm_model(model_path: str,
                    input_size: int = 23,
                    hidden_size: int = 128,
                    output_size: int = 1) -> LSTMModel:
    model = LSTMModel.load_from_checkpoint(
        model_path,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )
    model.eval()
    model.to('cpu')
    return model


def load_catboost_models(models_dir_path: str) -> dict[str, CatBoostRegressor]:
    loaded_models = {}
    for file_name in os.listdir(models_dir_path):
        ticker = file_name.split("_")[0]
        model_path = f"{models_dir_path}/{file_name}"
        if os.path.exists(model_path):
            model = CatBoostRegressor()
            model.load_model(model_path)
            loaded_models[ticker] = model
            print(f"Модель CatBoost для {ticker} загружена из {model_path}")
    # clear_output()
    return loaded_models


def load_scalers(scalers_dir_path: str) -> dict[str, MinMaxScaler | StandardScaler]:
    loaded_scalers = {}
    for file_name in os.listdir(scalers_dir_path):
        ticker = file_name.split("_")[0]
        scaler_path = f"{scalers_dir_path}/{file_name}"
        if os.path.exists(scaler_path):
            loaded_scalers[ticker] = pickle.load(open(scaler_path, 'rb'))
            # print(f"{ticker} : mean = {loaded_scalers[ticker].mean_[0]:.1f} | scale = {loaded_scalers[ticker].scale_[0]:.1f} | path = {scaler_path}")

    return loaded_scalers


# ===========================================================
# new_lstm_model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StockDataset(Dataset):
    """Dataset для временных рядов акций"""

    def __init__(self, df: pd.DataFrame, sequence_length: int = 30,
                 y_col: str = 'target',
                 features_scalers: dict[str, StandardScaler] | None = None,
                 step_size_range: Tuple[int, int] = (1, 5), max_sequences_per_symbol: int = 1000,
                 do_need_all_preds: bool = False):
        """
        Args:
            df: DataFrame с колонками [Date, Symbol, Close_today, target, features...]
            sequence_length: длина последовательности для LSTM
            step_size_range: диапазон размеров шага для создания последовательностей (случайный)
            max_sequences_per_symbol: максимальное количество последовательностей на символ
        """
        # print(f"Start init StockDataset")
        self.sequence_length = sequence_length
        self.step_size_range = step_size_range
        self.max_sequences_per_symbol = max_sequences_per_symbol
        self.y_col = y_col

        # Подготовка данных
        self.df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

        # Получаем список фичей (исключаем служебные колонки)
        self.feature_cols = [col for col in df.columns if col not in
                             ['Date', 'Symbol',
                                        #   'Close_today',
                                          'target', 'diff_target']]

        # Создаем мапинг символов к индексам
        self.symbols = sorted(df['Symbol'].unique())
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        self.n_symbols = len(self.symbols)

        # Нормализация фичей по каждому символу отдельно
        self.scalers = {}
        self.normalized_data = {}
        self.scaled_close = {}

        for symbol in self.symbols:
            symbol_data = self.df[self.df['Symbol'] == symbol].copy()
            self.scaled_close[symbol] = symbol_data['scaled_Close']

            # Скейлер для фичей
            if features_scalers is None:
                scaler = StandardScaler()
                symbol_data[self.feature_cols] = scaler.fit_transform(symbol_data[self.feature_cols])
            else:
                scaler = features_scalers[symbol]
                symbol_data[self.feature_cols] = scaler.transform(symbol_data[self.feature_cols])
            # symbol_data[self.feature_cols] = symbol_data[self.feature_cols].astype(np.float32)

            self.scalers[symbol] = scaler
            self.normalized_data[symbol] = symbol_data

        # Создаем последовательности
        self.sequences = self._create_sequences(do_need_all_preds=do_need_all_preds)

    def _create_sequences(self, do_need_all_preds: bool = False) -> List[Dict]:
        """Создает последовательности для обучения с переменным шагом"""
        sequences = []

        for symbol in self.symbols:
            symbol_data = self.normalized_data[symbol]
            # display(symbol_data)

            symbol_idx = self.symbol_to_idx[symbol]

            if len(symbol_data) < self.sequence_length:
                continue

            # Создаем последовательности с переменным шагом
            possible_starts = list(range(len(symbol_data) - self.sequence_length))
            # print(f"possible_starts: {len(possible_starts)}")

            if not do_need_all_preds and len(possible_starts) > self.max_sequences_per_symbol:
                # Используем случайные стартовые позиции, но с переменным шагом
                # print("NOT ALL PREDS")
                selected_starts = []
                current_pos = 0

                while current_pos < len(possible_starts) and len(selected_starts) < self.max_sequences_per_symbol:
                    selected_starts.append(possible_starts[current_pos])
                    # Случайный шаг
                    step = random.randint(*self.step_size_range)
                    current_pos += step

                # Добавляем еще несколько случайных позиций для разнообразия
                remaining_slots = self.max_sequences_per_symbol - len(selected_starts)
                if remaining_slots > 0:
                    additional_starts = random.sample(
                        [pos for pos in possible_starts if pos not in selected_starts],
                        min(remaining_slots, len(possible_starts) - len(selected_starts))
                    )
                    selected_starts.extend(additional_starts)

                start_indices = selected_starts
            else:
                # print("ALL PREDS")
                start_indices = possible_starts


            for start_idx in start_indices:
                seq_data = symbol_data.iloc[start_idx:start_idx + self.sequence_length]

                # Фичи для последовательности
                features = torch.FloatTensor(seq_data[self.feature_cols].values).float()

                symbol_scaled_close = self.scaled_close[symbol][start_idx:start_idx + self.sequence_length]
                scaled_close = torch.FloatTensor(symbol_scaled_close.values).float()
                # Таргеты для каждого шага в последовательности
                targets = torch.FloatTensor(seq_data[self.y_col].values).float()

                sequences.append({
                    'features': features,
                    'targets': targets,
                    'symbol_idx': symbol_idx,
                    'scaled_Close': scaled_close,
                    'symbol': symbol,
                    'dates': seq_data['Date'].tolist()
                })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def get_feature_dim(self):
        return len(self.feature_cols)


class StockLSTM(pl.LightningModule):
    """LSTM модель для предсказания цен акций"""

    def __init__(self,
                 feature_dim: int,
                 n_symbols: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 embedding_dim: int = 32,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 do_predict_diff: bool = False):
        super().__init__()

        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.n_symbols = n_symbols
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.do_predict_diff = do_predict_diff

        # Эмбеддинги для символов
        self.symbol_embedding = nn.Embedding(n_symbols, embedding_dim)

        # Итоговая размерность входа для LSTM
        lstm_input_size = feature_dim + embedding_dim

        # LSTM слои - исправлено: убрали двойной учет размерности
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,  # Теперь корректная размерность
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bias=True  # Явно указываем bias
        )

        # Убираем лишний linear_layer, он не использовался
        # self.linear_layer = nn.Linear(feature_dim + embedding_dim, hidden_dim)

        # Слой для предсказания цены
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, features, symbol_idx):
        """
        Args:
            features: (batch_size, seq_len, feature_dim)
            symbol_idx: (batch_size,)
        Returns:
            predictions: (batch_size, seq_len) - предсказания для каждого шага
        """
        batch_size, seq_len, feat_dim = features.shape
        # print(f"Start forward")
        # print(f"{batch_size=}, {seq_len=}, {feat_dim=}")

        # Проверяем, что feature_dim соответствует ожидаемому
        if feat_dim != self.feature_dim:
            print(f"Warning: Expected feature_dim={self.feature_dim}, got {feat_dim}")

        # Получаем эмбеддинги символов
        symbol_emb = self.symbol_embedding(symbol_idx)  # (batch_size, embedding_dim)

        # Расширяем эмбеддинги для каждого временного шага
        symbol_emb_expanded = symbol_emb.unsqueeze(1).expand(batch_size, seq_len, -1)

        # Объединяем фичи с эмбеддингами
        lstm_input = torch.cat([features, symbol_emb_expanded], dim=-1)

        # print(f"lstm_input.shape: {lstm_input.shape}")
        # print(f"Expected LSTM input_size: {self.feature_dim + self.embedding_dim}")

        # Проверяем размерности перед передачей в LSTM
        assert lstm_input.shape[-1] == self.feature_dim + self.embedding_dim, \
            f"LSTM input size mismatch: got {lstm_input.shape[-1]}, expected {self.feature_dim + self.embedding_dim}"

        # Пропускаем через LSTM
        try:
            lstm_out, _ = self.lstm(lstm_input)  # (batch_size, seq_len, hidden_dim)
        except RuntimeError as e:
            print(f"LSTM error: {e}")
            print(f"Input shape: {lstm_input.shape}")
            print(f"Input dtype: {lstm_input.dtype}")
            print(f"Input device: {lstm_input.device}")
            raise

        # Применяем dropout
        lstm_out = self.dropout(lstm_out)

        # Предсказываем цену для каждого шага
        predictions = self.price_predictor(lstm_out).squeeze(-1)  # (batch_size, seq_len)

        return predictions

    def _calculate_loss(self, predictions, targets):
        """Вычисляет MSE loss с весами для разных позиций"""
        # Даем больший вес последним предсказаниям
        seq_len = predictions.shape[1]
        weights = torch.linspace(0.5, 1.0, seq_len, device=predictions.device)
        weights = weights.unsqueeze(0).expand_as(predictions)

        mse_loss = F.mse_loss(predictions[:, -1], targets[:, -1], reduction='none')
        # weighted_loss = (mse_loss * weights).mean()
        # print(f"mse_loss: {mse_loss[:15]}")
        weighted_loss = mse_loss.mean()

        return weighted_loss

    def training_step(self, batch, batch_idx):
        features = batch['features']
        targets = batch['targets']
        symbol_idx = batch['symbol_idx']
        scaled_close = batch['scaled_Close']

        predictions = self(features, symbol_idx)
        loss = self._calculate_loss(predictions, targets)

        # Логируем метрики
        self.log('train_loss', loss, prog_bar=True)

        # Также считаем MAE для предсказания
        # last_pred_mae = F.l1_loss(predictions, targets)
        last_pred_mape = mean_absolute_percentage_error(targets.cpu().detach().numpy(),
                                                        predictions.cpu().detach().numpy())
        self.log('train_mape_last', last_pred_mape)

        # direct_accuracy
        if self.do_predict_diff:
            pred_diff = predictions
            target_diff = targets
        else:
            pred_diff = predictions - scaled_close
            target_diff = targets - scaled_close

        direct_acc = (pred_diff * target_diff > 0).float().mean()
        self.log('train_direct_accuracy', direct_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        features = batch['features']
        targets = batch['targets']
        symbol_idx = batch['symbol_idx']
        scaled_close = batch['scaled_Close']

        predictions = self(features, symbol_idx)
        # print(f"predictions: {predictions[:15]}")
        # print(f"targets: {targets[:15]}")
        loss = self._calculate_loss(predictions, targets)

        # Логируем метрики
        self.log('val_loss', loss, prog_bar=True)

        # MAE для предсказания
        last_pred_mape = mean_absolute_percentage_error(targets.cpu().detach().numpy(),
                                                        predictions.cpu().detach().numpy())
        self.log('val_mape_last', last_pred_mape)

        # direct_accuracy
        if self.do_predict_diff:
            pred_diff = predictions
            target_diff = targets
        else:
            pred_diff = predictions - scaled_close
            target_diff = targets - scaled_close

        direct_acc = (pred_diff * target_diff > 0).float().mean()
        self.log('val_direct_accuracy', direct_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


def collate_fn(batch):
    """Функция для правильного формирования батчей"""
    features = torch.stack([item['features'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    symbol_idx = torch.LongTensor([item['symbol_idx'] for item in batch])
    scaled_close = torch.stack([item['scaled_Close'] for item in batch])

    return {
        'features': features,
        'targets': targets,
        'symbol_idx': symbol_idx,
        'scaled_Close': scaled_close,
        'symbols': [item['symbol'] for item in batch],
        'dates': [item['dates'] for item in batch]
    }


def create_dataloaders(df: pd.DataFrame,
                      sequence_length: int = 30,
                      batch_size: int = 32,
                      test_size: float = 0.2,
                      val_size: float = 0.1,
                      train_start_date: str | None = None,
                      train_end_date: str = TRAIN_END_DATE,
                      val_end_date: str = TEST_START_DATE,
                      y_col: str = 'target') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Создает DataLoader'ы для обучения, валидации и тестирования"""

    # Разделяем данные по времени (более реалистично для временных рядов)
    df_sorted = df.sort_values('Date')

    # Разделяем данные
    if train_start_date is None:
        train_df = df_sorted[df_sorted['Date'] < train_end_date]
    else:
        train_df = df_sorted[(df_sorted['Date'] >= train_start_date) & (df_sorted['Date'] < train_end_date)]
    val_df = df_sorted[(df_sorted['Date'] >= train_end_date) & (df_sorted['Date'] < val_end_date)]
    # display(val_df.groupby('Symbol').size())
    test_df = df_sorted[df_sorted['Date'] >= val_end_date]

    # Создаем датасеты
    train_dataset = StockDataset(train_df, sequence_length=sequence_length, y_col=y_col)
    val_dataset = StockDataset(val_df, sequence_length=sequence_length, max_sequences_per_symbol=50,
                               y_col=y_col, features_scalers=train_dataset.scalers)
    test_dataset = StockDataset(test_df, sequence_length=sequence_length, max_sequences_per_symbol=50,
                               y_col=y_col, features_scalers=train_dataset.scalers)

    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )

    return train_loader, val_loader, test_loader, train_dataset.get_feature_dim(), train_dataset.n_symbols


def predict_on_test_data(model: StockLSTM, test_df: pd.DataFrame,
                        features_scalers: Dict[str, StandardScaler] | None = None,
                        sequence_length: int = 30) -> pd.DataFrame:
    """
    Делает предсказания на тестовых данных для каждой даты

    Returns:
        DataFrame с колонками [Date, Symbol, Actual_Price, Predicted_Price]
    """
    model.eval()

    # Создаем тестовый датасет
    test_dataset = StockDataset(test_df, sequence_length=sequence_length,
                                max_sequences_per_symbol=1000,
                                do_need_all_preds=True,
                                features_scalers=features_scalers)
    # display(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    predictions_list = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            targets = batch['targets']
            symbol_idx = batch['symbol_idx']
            symbols = batch['symbols']
            scaled_close = batch['scaled_Close']
            dates_batch = batch['dates']

            # Получаем предсказания
            predictions = model(features, symbol_idx)

            # Берем предсказание с последнего временного шага
            last_predictions = predictions[:, -1].cpu().numpy()
            last_targets = targets[:, -1].cpu().numpy()

            # Сохраняем результаты
            for i in range(len(last_predictions)):
                predictions_list.append({
                    'Date': dates_batch[i][-1],  # Последняя дата в последовательности
                    'Symbol': symbols[i],
                    'Actual_Target': last_targets[i],
                    'Predicted_Price': last_predictions[i],
                    'scaled_Close': scaled_close[i][-1].item()
                })

    return pd.DataFrame(predictions_list)


def eval_lstm_on_test_data(model: StockLSTM,
                           test_df: pd.DataFrame,
                           scalers: Dict[str, MinMaxScaler],
                           features_scalers: Dict[str, StandardScaler] | None = None,
                           sequence_length: int = 30,
                           y_col: str = 'Actual_Target',
                           do_pred_diff_target: bool = False,
                           do_scale: bool = True,
                           is_test_data: bool = True,
                           sleep_time: float = 1,
                           do_return_metrics: bool = False) -> pd.DataFrame:
    """Оценивает модель LSTM на тестовых данных"""
    # Делаем предсказания на тестовых данных
    print("Делаем предсказания...")
    predictions_df = predict_on_test_data(model, test_df, sequence_length=sequence_length,
                                          features_scalers=features_scalers)
    if is_test_data:
        real_need_predictions_df = predictions_df[predictions_df['Date'] >= TEST_START_DATE]
    else:
        real_need_predictions_df = predictions_df

    if do_scale:
        transform_lambda = lambda x: scalers[x.name].inverse_transform(x.values.reshape(-1, 1))[:, 0]
    else:
        transform_lambda = lambda x: x

    if do_pred_diff_target:
        real_need_predictions_df['Predicted_Price'] += real_need_predictions_df['scaled_Close']

    real_need_predictions_df['real_Close'] = real_need_predictions_df.groupby('Symbol')['scaled_Close'].transform(
        transform_lambda
    )
    real_need_predictions_df['real_pred'] = real_need_predictions_df.groupby('Symbol')['Predicted_Price'].transform(
        transform_lambda
    )
    real_need_predictions_df['real_target'] = real_need_predictions_df.groupby('Symbol')[y_col].transform(
        transform_lambda
    )

    # if do_pred_diff_target:
    #     real_need_predictions_df['real_pred'] = real_need_predictions_df['real_pred'] + real_need_predictions_df['real_Close']


    # print("Примеры предсказаний:")
    # display(real_need_predictions_df.head(10))

    for sym in real_need_predictions_df['Symbol'].unique():
        cur_test_df = real_need_predictions_df[real_need_predictions_df['Symbol'] == sym]

        pred = cur_test_df['real_pred'].values
        cur_price = cur_test_df['real_Close'].values
        cur_target = cur_test_df['real_target'].values

        display(cur_test_df)

        # Подсчет direct_accuracy
        pred_diff = pred - cur_price
        target_diff = cur_target - cur_price
        direct_accuracy = np.mean((pred_diff * target_diff) > 0)

        plt.plot(cur_test_df['Date'], cur_target, label='original')
        plt.plot(cur_test_df['Date'], pred, label='pred')
        plt.legend()
        plt.show()

        print(f"MSE for {sym}: {mean_squared_error(cur_target, pred):.6f}")
        print(f"MAPE for {sym}: {mean_absolute_percentage_error(cur_target, pred):.6f}")
        print(f"Direct Accuracy for {sym}: {direct_accuracy:.6f}")
        print(f"cur_price: {cur_price[:15]}")
        print(f"pred: {pred[:15]}")
        print(f"target: {cur_target[:15]}")
        print(f"pred_diff: {pred_diff[:15]}")
        print(f"target_diff: {target_diff[:15]}")

        clear_output(wait=True)
        sleep(sleep_time)


    # Вычисляем метрики
    mae = mean_absolute_error(real_need_predictions_df['real_pred'], real_need_predictions_df['real_target'])
    mse = mean_squared_error(real_need_predictions_df['real_pred'], real_need_predictions_df['real_target'])
    mape = mean_absolute_percentage_error(real_need_predictions_df['real_pred'], real_need_predictions_df['real_target'])
    # Подсчет direct_accuracy
    pred_diff = real_need_predictions_df['real_pred'] - real_need_predictions_df['real_Close']
    target_diff = real_need_predictions_df['real_target'] - real_need_predictions_df['real_Close']
    direct_accuracy = np.mean((pred_diff * target_diff) > 0)

    print(f"\nМетрики на тестовых данных:")
    print(f"MSE: {mse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"Direct Accuracy: {direct_accuracy:.4f}")

    if do_return_metrics:
        return real_need_predictions_df, {'mse': mse, 'mape': mape, 'direct_accuracy': direct_accuracy}
    else:
        return real_need_predictions_df


def train_lstm_model(train_df: pd.DataFrame,
                sequence_length: int = 30,
                batch_size: int = 32,
                max_epochs: int = 100,
                learning_rate: float = 1e-3,
                hidden_dim: int = 128,
                embedding_dim: int = 32,
                num_layers: int = 2,
                dropout: float = 0.2,
                y_col: str = 'target',
                train_start_date: str | None = None,
                accelerator: str = 'auto',
                wandb_name_iter: int = 1) -> StockLSTM:
    """Обучает модель LSTM"""

    # Создаем DataLoader'ы
    train_loader, val_loader, test_loader, feature_dim, n_symbols = create_dataloaders(
        train_df, sequence_length, batch_size,
        y_col=y_col, train_start_date=train_start_date,
    )
    print(f"Loaders are ready\n {len(train_loader)=}\n {len(val_loader)=}\n {len(test_loader)=}\n {feature_dim=}\n {n_symbols=}")

    do_predict_diff = (True if y_col == 'diff_target' else False)
    print(f"do_predict_diff: {do_predict_diff}")
    # Создаем модель
    model = StockLSTM(
        feature_dim=feature_dim,
        n_symbols=n_symbols,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        do_predict_diff=do_predict_diff,
        dropout=dropout
    )

    # Настраиваем колбэки
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='best-stock-lstm-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode='min'
    )

    # Создаём WandbLogger
    wandb_name = f'stock-lstm-{wandb_name_iter}'

    print(f"wandb_name: {wandb_name}")
    wandb_logger = WandbLogger(
        project='stock-lstm',  # замените на ваш проект
        name=wandb_name
    )

    # Создаем тренер
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        # callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        accelerator=accelerator,
        precision=16,
        log_every_n_steps=50
    )

    # Обучаем модель
    trainer.fit(model, train_loader, val_loader)

    # return model, (train_loader, val_loader, test_loader)
    return model, train_loader.dataset.scalers
    # return model


# Пример использования
if __name__ == "__main__":
    # Предполагаем, что у вас есть DataFrame df с нужными колонками
    # df = pd.read_csv('your_stock_data.csv')

    # Пример создания тестовых данных
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMD']

    data = []
    for symbol in symbols:
        for date in dates:
            # Генерируем фейковые данные
            close_today = 100 + np.random.randn() * 10
            close_7_days = close_today + np.random.randn() * 5

            data.append({
                'Date': date,
                'Symbol': symbol,
                'Close_today': close_today,
                'target': close_7_days,
                'feature_1': np.random.randn(),
                'feature_2': np.random.randn(),
                'feature_3': np.random.randn(),
                'feature_4': np.random.randn(),
                'feature_5': np.random.randn(),
            })

    df = pd.DataFrame(data)

    # Обучаем модель
    print("Начинаем обучение модели...")
    model = train_lstm_model(df, sequence_length=30, batch_size=32, max_epochs=50)

    # Делаем предсказания на тестовых данных
    print("Делаем предсказания...")
    test_df = df[df['Date'] >= '2023-01-01']  # Последний год как тест
    predictions_df = predict_on_test_data(model, test_df)

    print("Примеры предсказаний:")
    print(predictions_df.head(10))

    # Вычисляем метрики
    mae = np.mean(np.abs(predictions_df['Actual_Price'] - predictions_df['Predicted_Price']))
    mse = np.mean((predictions_df['Actual_Price'] - predictions_df['Predicted_Price']) ** 2)

    print(f"\nМетрики на тестовых данных:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")



# ===========================================================
# for_cbt.py
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from time import sleep


def estimate_best_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # Оцениваем качество модели на тестовых данных
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_predictions)
    val_mse = mean_squared_error(y_val, val_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    print(f"MSE на тренировочных данных: {train_mse:.6f}")
    print(f"MSE на валидационных данных: {val_mse:.6f}")
    print(f"MSE на тестовых данных: {test_mse:.6f}")

    # Построение графика зависимости лосса от итераций
    plt.figure(figsize=(10, 6))
    plt.plot(model.get_evals_result()['learn']['RMSE'], label='Обучающая выборка')
    plt.plot(model.get_evals_result()['validation']['RMSE'], label='Валидационная выборка')
    plt.xlabel('Итерации')
    plt.ylabel('RMSE')
    plt.title('Зависимость MSE от количества итераций')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Выводим важность признаков
    feature_importances = model.get_feature_importance(prettified=True)
    print("\nВажность признаков:")
    print(feature_importances.head(10))  # Топ-10 признаков

    return model


def train_simple_cbt(X_train, y_train, X_val, y_val, X_test, y_test,
                     cbt_params: dict = None,
                     do_plot: bool = False):
    if cbt_params is None:
        cbt_params = {
            'iterations': 2000,
            'depth': 10,
            'learning_rate': 0.01,
            'l2_leaf_reg': 0.2,
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': 100,
            'use_best_model': True
        }
    model = CatBoostRegressor(**cbt_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=do_plot)

    estimate_best_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    return model


def eval_cbts_on_test(models, scalers,
                      test_df: pd.DataFrame,
                      y_col: str = 'target',
                      do_pred_diff_target: bool = False,
                      do_scale: bool = True,
                      sleep_time: float = 1):
    all_symbols = test_df['Symbol'].unique()
    # all_symbols = list(models.keys())
    all_y = []
    all_pred = []
    all_cur_price = []
    for sym in all_symbols:
        cur_test_df = test_df[test_df['Symbol'] == sym]

        X_test, y_test = create_X_y_df_from_df(cur_test_df, target_col=y_col)

        if do_scale:
            y_test = scalers[sym].inverse_transform(y_test.values.reshape(-1, 1))[:, 0]

        pred = models[sym].predict(X_test)
        cur_price = cur_test_df['scaled_Close'].values
        if do_pred_diff_target:
            pred = pred + cur_price

        if do_scale:
            pred = scalers[sym].inverse_transform(pred.reshape(-1, 1))[:, 0]

        # Подсчет direct_accuracy
        cur_scaled_price = scalers[sym].inverse_transform(cur_price.reshape(-1, 1))[:, 0]
        pred_diff = pred - cur_scaled_price
        target_diff = y_test - cur_scaled_price
        direct_accuracy = np.mean((pred_diff * target_diff) > 0)

        plt.plot(cur_test_df.index, y_test, label='original')
        plt.plot(cur_test_df.index, pred, label='pred')
        plt.legend()
        plt.show()

        print(f"MSE for {sym}: {mean_squared_error(y_test, pred):.6f}")
        print(f"MAPE for {sym}: {mean_absolute_percentage_error(y_test, pred):.6f}")
        print(f"Direct Accuracy for {sym}: {direct_accuracy:.6f}")
        print(f"pred_diff: {pred_diff[:15]}")
        print(f"target_diff: {target_diff[:15]}")
        print(f"cur_scaled_price: {cur_scaled_price[:15]}")

        clear_output(wait=True)
        sleep(sleep_time)

        all_y.append(y_test)
        all_pred.append(pred)
        all_cur_price.append(cur_scaled_price)
    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    all_cur_price = np.concatenate(all_cur_price)

    mse = mean_squared_error(all_y, all_pred)
    mape = mean_absolute_percentage_error(all_y, all_pred)
    direct_accuracy = np.mean((all_pred - all_cur_price) * (all_y - all_cur_price) > 0)

    print(f"MSE: {mse:.6f}")
    print(f"MAPE: {mape:.6f}")
    print(f"Direct Accuracy: {direct_accuracy:.6f}")
    return mape


def random_search_catboost(X_train, y_train, X_val, y_val, X_test, y_test,
                          n_iter: int = 50,
                          random_state: int = 42,
                          use_gpu: bool = True,
                          rs_verbose: int = 1,
                          cbt_verbose: int = 1) -> CatBoostRegressor:
    """
    Подбор гиперпараметров для CatBoost с помощью RandomSearch

    Args:
        X_train, y_train: данные для обучения
        X_val, y_val: данные для валидации
        n_iter: количество итераций поиска
        cv: количество фолдов для кросс-валидации
        random_state: seed для воспроизводимости

    Returns:
        Лучшая модель CatBoost
    """
    # Определяем пространство параметров для поиска
    param_dist = {
        # Важные параметры (оказывают наибольшее влияние на качество)
        'learning_rate': np.logspace(-5, 0, 100),  # Скорость обучения
        'depth': np.arange(4, 12),                 # Глубина деревьев
        'l2_leaf_reg': np.logspace(-2, 2, 100),    # L2 регуляризация
        'iterations': [500, 1000, 2000],    # Количество итераций
        # 'min_data_in_leaf': np.arange(1, 100, 5),
        # 'max_leaves': np.arange(31, 127, 10),
        'task_type': ['GPU'] if use_gpu else ['CPU'],
    }

    # Создаем базовую модель
    base_model = CatBoostRegressor(
        loss_function='RMSE',
        random_seed=random_state,
        verbose=cbt_verbose
    )

    # Создаем RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=[(np.arange(len(X_train)),
             np.arange(len(X_train), len(X_train) + len(X_val)))],  # Используем предопределенные train/val индексы
        scoring='neg_root_mean_squared_error',
        random_state=random_state,
        n_jobs=-1,
        verbose=rs_verbose
    )

    # Объединяем train и val данные для поиска
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    # Обучаем модель
    random_search.fit(X_combined, y_combined)

    # return random_search.best_estimator_

    # Создаем и обучаем финальную модель с лучшими параметрами
    best_model = CatBoostRegressor(
        **random_search.best_params_,
        loss_function='RMSE',
        random_seed=random_state,
        verbose=False
    )

    # Обучаем на всех данных
    best_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )

    estimate_best_model(best_model, X_train, y_train, X_val, y_val, X_test, y_test)

    return best_model


# ===========================================================
# CNN_Attention_LSTM.py


class EnhancedStockLSTM(pl.LightningModule):
    def __init__(self,
                 feature_dim: int,
                 n_symbols: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 embedding_dim: int = 32,
                 cnn_channels: int = 64,
                 num_heads: int = 4,
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 do_predict_diff: bool = False,
                 do_use_cnn: bool = True,
                 do_use_attention: bool = True,
                 do_use_lstm: bool = True,
                 num_attention_layers: int = 5):
        super().__init__()

        self.save_hyperparameters()

        self.feature_dim = feature_dim
        self.n_symbols = n_symbols
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.cnn_channels = cnn_channels
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.do_predict_diff = do_predict_diff
        self.do_use_cnn = do_use_cnn
        self.do_use_attention = do_use_attention
        self.do_use_lstm = do_use_lstm
        self.num_attention_layers = num_attention_layers

        # Эмбеддинги для символов
        self.symbol_embedding = nn.Embedding(n_symbols, embedding_dim)

        # Сверточный слой для обработки временных рядов
        if do_use_cnn:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(feature_dim, cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(cnn_channels),
                nn.ReLU(),
                nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(cnn_channels),
                nn.ReLU(),
                nn.Conv1d(cnn_channels, feature_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU()
            )
            # Gate для CNN skip-connection
            self.cnn_gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid()
            )

        # Проекция признаков
        if self.hidden_dim < feature_dim + embedding_dim:
            raise ValueError(f"hidden_dim must be greater than feature_dim + embedding_dim, but got {self.hidden_dim} < {feature_dim + embedding_dim}")
        self.feature_projection = nn.Linear(
            feature_dim + embedding_dim,
            hidden_dim
        )

        # Инициализация проекции
        with torch.no_grad():
            self.feature_projection.weight.zero_()
            self.feature_projection.bias.zero_()
            for i in range(min(feature_dim + embedding_dim, hidden_dim)):
                self.feature_projection.weight[i, i] = 1.0
            if hidden_dim > feature_dim + embedding_dim:
                nn.init.kaiming_uniform_(self.feature_projection.weight[feature_dim+embedding_dim:, :], a=5**0.5)
                nn.init.zeros_(self.feature_projection.bias[feature_dim+embedding_dim:])

        # Создаем несколько слоев внимания
        if do_use_attention:
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(num_attention_layers)
            ])

            # Добавляем нормализацию для каждого слоя внимания
            self.attention_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_attention_layers)
            ])

            # Gates для attention skip-connections
            self.attention_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid()
                ) for _ in range(num_attention_layers)
            ])

        # LSTM слои
        if do_use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bias=True
            )
            # Gate для LSTM skip-connection
            self.lstm_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )

        # MLP для финального предсказания
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, features, symbol_idx):
        batch_size, seq_len, feat_dim = features.shape

        if feat_dim != self.feature_dim:
            print(f"Warning: Expected feature_dim={self.feature_dim}, got {feat_dim}")

        # 1. Применяем сверточные слои
        if self.do_use_cnn:
            conv_input = features.transpose(1, 2)
            conv_output = self.conv_layers(conv_input)
            conv_output = conv_output.transpose(1, 2)

            # Применяем gate для CNN skip-connection
            gate_input = torch.cat([conv_output, features], dim=-1)
            gate = self.cnn_gate(gate_input)
            conv_output = gate * conv_output + features
        else:
            conv_output = features

        # 2. Получаем и расширяем эмбеддинги символов
        symbol_emb = self.symbol_embedding(symbol_idx)
        symbol_emb_expanded = symbol_emb.unsqueeze(1).expand(batch_size, seq_len, -1)

        # 3. Объединяем выход CNN с эмбеддингами
        combined_features = torch.cat([conv_output, symbol_emb_expanded], dim=-1)

        # 4. Проецируем в нужную размерность
        projected_features = self.feature_projection(combined_features)

        # 5. Применяем несколько слоев внимания с gated skip-connections
        if self.do_use_attention:
            attn_output = projected_features
            for i, (attention_layer, norm_layer, gate_layer) in enumerate(zip(
                self.attention_layers, self.attention_norms, self.attention_gates)):

                # Нормализация перед вниманием
                attn_input = norm_layer(attn_output)
                # Применяем внимание
                attn_layer_output, _ = attention_layer(attn_input, attn_input, attn_input)

                # Применяем gate для attention skip-connection
                gate_input = torch.cat([attn_layer_output, attn_output], dim=-1)
                gate = gate_layer(gate_input)
                attn_output = gate * attn_layer_output + attn_output

                # Применяем dropout
                attn_output = self.dropout(attn_output)
        else:
            attn_output = projected_features

        # 6. Пропускаем через LSTM
        if self.do_use_lstm:
            lstm_out, _ = self.lstm(attn_output)

            # Применяем gate для LSTM skip-connection
            gate_input = torch.cat([lstm_out, attn_output], dim=-1)
            gate = self.lstm_gate(gate_input)
            lstm_out = gate * lstm_out + attn_output
        else:
            lstm_out = attn_output

        # 7. Применяем dropout
        lstm_out = self.dropout(lstm_out)

        # 8. Делаем финальное предсказание
        predictions = self.price_predictor(lstm_out).squeeze(-1)

        return predictions

    def _calculate_loss(self, predictions, targets):
        """Вычисляет MSE loss с весами для разных позиций"""
        seq_len = predictions.shape[1]
        weights = torch.linspace(0.5, 1.0, seq_len, device=predictions.device)
        weights = weights.unsqueeze(0).expand_as(predictions)

        mse_loss = F.mse_loss(predictions, targets, reduction='none')
        weighted_loss = (mse_loss * weights).mean()

        return weighted_loss

    def training_step(self, batch, batch_idx):
        features = batch['features']
        targets = batch['targets']
        symbol_idx = batch['symbol_idx']
        scaled_close = batch['scaled_Close']

        predictions = self(features, symbol_idx)
        loss = self._calculate_loss(predictions, targets)

        # Логируем метрики
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae_last', F.l1_loss(predictions, targets))

        # Вычисляем direct accuracy
        if self.do_predict_diff:
            pred_diff = predictions
            target_diff = targets
        else:
            pred_diff = predictions - scaled_close
            target_diff = targets - scaled_close

        direct_acc = (pred_diff * target_diff > 0).float().mean()
        self.log('train_direct_accuracy', direct_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        features = batch['features']
        targets = batch['targets']
        symbol_idx = batch['symbol_idx']
        scaled_close = batch['scaled_Close']

        predictions = self(features, symbol_idx)
        loss = self._calculate_loss(predictions, targets)

        # Логируем метрики
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae_last', F.l1_loss(predictions, targets))

        # Вычисляем direct accuracy
        if self.do_predict_diff:
            pred_diff = predictions
            target_diff = targets
        else:
            pred_diff = predictions - scaled_close
            target_diff = targets - scaled_close

        direct_acc = (pred_diff * target_diff > 0).float().mean()
        self.log('val_direct_accuracy', direct_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


def train_enhanced_lstm_model(train_df: pd.DataFrame,
                sequence_length: int = 30,
                batch_size: int = 32,
                max_epochs: int = 100,
                learning_rate: float = 1e-3,
                hidden_dim: int = 128,
                embedding_dim: int = 32,
                num_layers: int = 2,
                dropout: float = 0.2,
                do_use_cnn: bool = True,
                do_use_attention: bool = True,
                do_use_lstm: bool = True,
                y_col: str = 'target',
                train_start_date: str | None = None,
                accelerator: str = 'auto',
                wandb_name_iter: int = 1) -> EnhancedStockLSTM:
    """Обучает модель EnhancedStockLSTM"""

    # Создаем DataLoader'ы
    train_loader, val_loader, test_loader, feature_dim, n_symbols = create_dataloaders(
        train_df, sequence_length, batch_size,
        y_col=y_col, train_start_date=train_start_date,
    )
    print(f"Loaders are ready\n {len(train_loader)=}\n {len(val_loader)=}\n {len(test_loader)=}\n {feature_dim=}\n {n_symbols=}")

    do_predict_diff = (True if y_col == 'diff_target' else False)
    print(f"do_predict_diff: {do_predict_diff}")
    # Создаем модель
    model = EnhancedStockLSTM(
        feature_dim=feature_dim,
        n_symbols=n_symbols,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        do_predict_diff=do_predict_diff,
        dropout=dropout,
        do_use_cnn=do_use_cnn,
        do_use_attention=do_use_attention,
        do_use_lstm=do_use_lstm
    )

    # Настраиваем колбэки
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='best-stock-lstm-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode='min'
    )

    # Создаём WandbLogger
    wandb_name = f'enhanced-stock-lstm-{wandb_name_iter}'

    print(f"wandb_name: {wandb_name}")
    wandb_logger = WandbLogger(
        project='enhanced-stock-lstm',  # замените на ваш проект
        name=wandb_name
    )

    # Создаем тренер
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        # callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        accelerator=accelerator,
        precision=16,
        log_every_n_steps=50
    )

    # Обучаем модель
    trainer.fit(model, train_loader, val_loader)

    # return model, (train_loader, val_loader, test_loader)
    return model, train_loader.dataset.scalers
    # return model
    # Создаем модель


# ===========================================================
# other