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
                               batch_size: int = 100) -> List[Dict]:
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

    return result[:n]
