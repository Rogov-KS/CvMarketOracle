import pandas as pd
import numpy as np
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import DateOffset, Day, Easter
from datetime import datetime, timedelta
from typing import List

from market_oracle_lib.consts import (
    BASE_TARGET_COL,
    TARGET_HORIZONT,
    TARGET_COL
)


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
