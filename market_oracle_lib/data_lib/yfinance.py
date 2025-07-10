import yfinance as yf
import pandas as pd
from typing import List, Tuple
from torch.utils.data import DataLoader
from market_oracle_lib.data_lib.data_funcs import (
    prepare_data, timeseries_split, create_data_loaders
)



def get_symbol_data(symbol: str,
                    # start_date: str = "2020-01-01",
                    # end_date: str = "2024-01-01",
                    **kwargs) -> pd.DataFrame:
    """Получение данных для одного символа."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(**kwargs)
        df['Symbol'] = symbol
        return df
    except Exception as e:
        print(f"Ошибка при получении данных для {symbol}: {e}")
        return pd.DataFrame()
