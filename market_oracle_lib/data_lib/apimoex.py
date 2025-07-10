import apimoex
import requests
import pandas as pd
from typing import Dict


interval_mapping: Dict[str, int] = {
    "1m":  1,
    "10m": 10,
    "1h":  60,
    "1d":  24,
    "1w":  7,
    "1mo": 31,
    "1q":  4,
}


def get_symbol_data(
    symbol: str,
    # start_date: str = "2020-01-01",
    # end_date: str = "2024-01-01",
    interval: str = "1d",
    **kwargs
) -> pd.DataFrame:
    """Получение данных для одного символа с MOEX."""
    try:
        with requests.Session() as session:
            data = apimoex.get_market_candles(
                session,
                symbol,
                interval_mapping[interval],
                # start=start_date,
                # end=end_date,
                **kwargs
            )

            if data:
                df = pd.DataFrame(data)
                # Переименовываем колонки в соответствии с форматом MOEX
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'begin': 'Date'
                })
                df['Symbol'] = symbol
                return df
            return pd.DataFrame()

    except Exception as e:
        print(f"Ошибка при получении данных для {symbol}: {e}")
        return pd.DataFrame()
