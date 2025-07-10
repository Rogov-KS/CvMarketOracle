from copy import deepcopy
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional, Callable, Union, Literal, Dict
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

from market_oracle_lib.data_lib.make_features import (
    calculate_shifted_features,
    calculate_joined_features,
    add_shift_col_by_join,
    calculate_time_features,
)
from tqdm import tqdm
from IPython.display import clear_output, display
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from market_oracle_lib.consts import (
    FEATURE_COLS,
    BASE_TARGET_COL,
    TARGET_COL,
    US_DEFAULT_SYMBOLS,
    TARGET_HORIZONT,
    TRAIN_END_DATE,
    TEST_START_DATE,
)


class StockDataset(Dataset):
    """
    Dataset for stock data.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


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
                                        do_get_diff_target=True)


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
                      train_end_date: str = TRAIN_END_DATE,
                      test_start_date: str = TEST_START_DATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Разделение DataFrame на обучающую, валидационную и тестовую выборки
    """
    train_df = df[df['Date'] <= train_end_date]
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


def create_data_sets(
    data_getter_fn: Callable,
    feature_cols: Optional[List[str]] = FEATURE_COLS,
    target_col: str = TARGET_COL,
    **kwargs
) -> Tuple[StockDataset, StockDataset, StockDataset]:
    """Создание DataLoader'ов для обучения, валидации и тестирования."""
    # Получение данных
    train_df, val_df, test_df = create_data_frames(data_getter_fn, **kwargs)

    print(f"{train_df.shape=} | {val_df.shape=} | {test_df.shape=}")
    print(f"{train_df.head()=} | {val_df.head()=} | {test_df.head()=}")

    # Преобразуем DataFrame в тензоры PyTorch
    X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
    X_val, y_val = val_df[feature_cols].values, val_df[target_col].values
    X_test, y_test = test_df[feature_cols].values, test_df[target_col].values


    # Создание датасетов
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    print("Data datasets created successfully")

    return train_dataset, val_dataset, test_dataset


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
