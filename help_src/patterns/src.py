import numpy as np
from scipy import stats
import pandas as pd
import plotly.graph_objects as go
from .classes import PriceChannel


def isPivot(df, candle_index, window_param):
    """
    Определяет, является ли свеча точкой разворота (pivot/fractal)
    с использованием операций pandas для анализа окна.

    Аргументы:
    df: DataFrame с данными OHLC. Ожидаются колонки 'Low' и 'High'.
    candle_index: индекс свечи для проверки.
    window_param: количество свечей до и после для определения разворота.

    Возвращает:
    1: если точка разворота вверх (pivot high).
    2: если точка разворота вниз (pivot low).
    3: если одновременно вверх и вниз.
    0: если не является точкой разворота или окно неполное.
    """
    # Проверка, что полное окно существует вокруг свечи
    if candle_index - window_param < 0 or candle_index + window_param >= len(df):
        return 0

    current_candle = df.iloc[candle_index]
    current_high = current_candle['High']
    current_low = current_candle['Low']

    # Определяем срез DataFrame для окна:
    # window_param свечей до, сама свеча, и window_param свечей после
    start_slice = candle_index - window_param
    # +1 потому что срез Python не включает верхнюю границу
    end_slice = candle_index + window_param + 1

    window_df_slice = df.iloc[start_slice:end_slice]

    # Является ли High текущей свечи максимальным в окне?
    is_pivot_high = (current_high == window_df_slice['High'].max())

    # Является ли Low текущей свечи минимальным в окне?
    is_pivot_low = (current_low == window_df_slice['Low'].min())

    if is_pivot_high and is_pivot_low:
        return 3
    elif is_pivot_high:
        return 1
    elif is_pivot_low:
        return 2
    else:
        return 0


def add_pivot_column(df: pd.DataFrame,
                   window_param: int,
                   low_col: str = 'Low',
                   high_col: str = 'High',
                   pivot_col: str = 'isPivot',
                   offset: float = 1e-3) -> pd.DataFrame:
    """
    Добавляет колонку 'isPivot' в DataFrame, содержащую типы разворотов.
    """
    df['isPivot'] = [isPivot(df, i, window_param) for i in range(len(df))]
    df['pointpos'] = df.apply(lambda row: calculate_point_pos(row, low_col=low_col, high_col=high_col,
                                                              pivot_col=pivot_col, offset=offset), axis=1)
    return df

# Пример функции для расчета позиций точек для графика, адаптированная
def calculate_point_pos(row,
                        low_col='Low',
                        high_col='High',
                        pivot_col='isPivot',
                        offset=1e-3):
    """
    Рассчитывает позицию точки для графика в зависимости от типа точки разворота.
    """
    pivot_value = row[pivot_col]
    if pivot_value == 2:  # Pivot Low
        return row[low_col] - offset
    elif pivot_value == 1:  # Pivot High
        return row[high_col] + offset
    else:
        return np.nan


def collect_channel(df: pd.DataFrame,
                    candle: int,
                    backcandles: int,
                    window: int,
                    should_refine: bool = False) -> PriceChannel:
    """
    Определяет параметры ценового канала (верхняя и нижняя линии тренда)
    на основе точек разворота за предыдущий период.

    Args:
        df (pd.DataFrame): DataFrame с историческими данными цен, должен содержать колонки 'High', 'Low'.
        candle (int): Индекс текущей свечи, относительно которой строится канал.
        backcandles (int): Количество предыдущих свечей для анализа (построения канала).
        window (int): Размер окна для определения точек разворота (параметр для isPivot).

    Returns:
        PriceChannel: Объект PriceChannel с параметрами канала или
                      пустой PriceChannel, если построить не удалось.
    """
    start_idx_for_pivot_detection = candle - backcandles - window
    end_idx_for_pivot_detection = candle - window

    invalid_indices = (
        start_idx_for_pivot_detection < 0 or
        end_idx_for_pivot_detection > len(df) or
        start_idx_for_pivot_detection >= end_idx_for_pivot_detection
    )
    if invalid_indices:
        return PriceChannel()

    localdf = df.iloc[start_idx_for_pivot_detection:end_idx_for_pivot_detection].copy()

    if localdf.empty:
        return PriceChannel()

    pivot_values = [isPivot(localdf, i, window) for i in range(len(localdf))]
    localdf['isPivot'] = pivot_values

    highs_mask = (localdf['isPivot'] == 1) | (localdf['isPivot'] == 3)
    lows_mask = (localdf['isPivot'] == 2) | (localdf['isPivot'] == 3)

    highs = localdf[highs_mask].High.values
    idxhighs = localdf.loc[highs_mask, "Date"]
    idxhighs = (idxhighs - idxhighs.min()).dt.total_seconds()
    # idxhighs = np.where(highs_mask)[0]

    lows = localdf[lows_mask].Low.values
    idxlows = localdf.loc[lows_mask, "Date"]
    idxlows = (idxlows - idxlows.min()).dt.total_seconds()
    # idxlows = np.where(lows_mask)[0]

    sl_lows, interc_lows, r_sq_l = 0.0, 0.0, 0.0
    sl_highs, interc_highs, r_sq_h = 0.0, 0.0, 0.0

    if len(lows) >= 2:
        res_lows = stats.linregress(idxlows, lows)
        sl_lows = res_lows.slope
        interc_lows = res_lows.intercept
        r_sq_l = res_lows.rvalue**2 if not np.isnan(res_lows.rvalue) else 0.0

    if len(highs) >= 2:
        res_highs = stats.linregress(idxhighs, highs)
        sl_highs = res_highs.slope
        interc_highs = res_highs.intercept
        r_sq_h = res_highs.rvalue**2 if not np.isnan(res_highs.rvalue) else 0.0


    # Начало localdf в оригинальном df:
    localdf_start_orig_idx = candle - backcandles - window
    # Конец localdf в оригинальном df:
    localdf_end_orig_idx = candle - window

    price_channel = PriceChannel(
        df=df,
        slope_lows=sl_lows,
        intercept_lows=interc_lows,
        r_squared_lows=r_sq_l,
        slope_highs=sl_highs,
        intercept_highs=interc_highs,
        r_squared_highs=r_sq_h,
        start_idx=localdf_start_orig_idx,
        end_idx=localdf_end_orig_idx
    )

    if should_refine:
        price_channel = refine_channel(price_channel, should_refine)

    return price_channel


def refine_channel(price_channel: PriceChannel, should_refine: bool = False) -> PriceChannel:
    """
    Уточняет ценовой канал, корректируя его границы на основе максимальных отклонений пивотов.

    Args:
        price_channel (PriceChannel): Исходный ценовой канал
        should_refine (bool): Флаг, указывающий нужно ли выполнять уточнение

    Returns:
        PriceChannel: Уточненный ценовой канал или исходный, если should_refine=False
    """
    if not should_refine:
        return price_channel

    # Получаем индексы для анализа
    start_idx = price_channel.start_idx
    end_idx = price_channel.end_idx

    # Получаем данные для анализа
    analysis_df = price_channel.df.iloc[start_idx:end_idx].copy()

    # Находим пивоты в анализируемом диапазоне
    pivot_values = [isPivot(analysis_df, i, 2) for i in range(len(analysis_df))]
    analysis_df['isPivot'] = pivot_values

    # Разделяем пивоты на верхние и нижние
    highs_mask = (analysis_df['isPivot'] == 1) | (analysis_df['isPivot'] == 3)
    lows_mask = (analysis_df['isPivot'] == 2) | (analysis_df['isPivot'] == 3)

    # Получаем даты для пивотов
    high_dates = analysis_df.loc[highs_mask, "Date"]
    low_dates = analysis_df.loc[lows_mask, "Date"]

    # Преобразуем даты в секунды для расчета отклонений
    high_seconds = (high_dates - high_dates.min()).dt.total_seconds()
    low_seconds = (low_dates - low_dates.min()).dt.total_seconds()

    # Рассчитываем предсказанные значения для пивотов
    high_predictions = price_channel.slope_highs * high_seconds + price_channel.intercept_highs
    low_predictions = price_channel.slope_lows * low_seconds + price_channel.intercept_lows

    # Находим максимальные отклонения
    high_errors = analysis_df.loc[highs_mask, 'High'].values - high_predictions
    low_errors = analysis_df.loc[lows_mask, 'Low'].values - low_predictions

    max_high_error = np.max(high_errors) if len(high_errors) > 0 else 0
    max_low_error = np.min(low_errors) if len(low_errors) > 0 else 0

    # Создаем новый канал с уточненными параметрами
    refined_channel = PriceChannel(
        df=price_channel.df,
        slope_lows=price_channel.slope_lows,
        intercept_lows=price_channel.intercept_lows + max_low_error,
        r_squared_lows=price_channel.r_squared_lows,
        slope_highs=price_channel.slope_highs,
        intercept_highs=price_channel.intercept_highs + max_high_error,
        r_squared_highs=price_channel.r_squared_highs,
        start_idx=price_channel.start_idx,
        end_idx=price_channel.end_idx
    )

    return refined_channel


def calculate_breakout_series(df_full: pd.DataFrame,
                              price_channel: PriceChannel) -> pd.Series:
    """
    Рассчитывает pd.Series, указывающий для каждой свечи в df_full,
    выходит ли она за рамки предоставленного ценового канала.

    Args:
        df_full (pd.DataFrame): Полный DataFrame с данными OHLC и 'Date'.
                                Индекс должен быть совместим для создания Series.
        price_channel (PriceChannel): Предварительно рассчитанный объект
                                      ценового канала.

    Returns:
        pd.Series: Серия со статусами пробоя для каждой свечи в df_full.
                   Индекс серии будет таким же, как у df_full.
                   Значения:
                       0 - нет пробоя.
                       1 - пробой вверх.
                       2 - пробой вниз.
                       3 - одновременный пробой.
                       -1 - канал невалиден / ошибка расчета.
    """
    if price_channel is None or not price_channel.is_valid():
        return pd.Series([-1] * len(df_full), index=df_full.index,
                         name="isBreakout", dtype=int)

    candle_dates = df_full[price_channel.x_name]

    # Рассчитываем разницу во времени для каждой свечи от начала канала
    # Это необходимо для применения линейной регрессии канала
    date_difference = (candle_dates -
                       price_channel.x_start)
    time_deltas_seconds = date_difference.dt.total_seconds()

    # Рассчитываем значения верхней и нижней границ канала для каждой свечи
    upper_channel_values = (price_channel.slope_highs * time_deltas_seconds +
                            price_channel.intercept_highs)
    lower_channel_values = (price_channel.slope_lows * time_deltas_seconds +
                            price_channel.intercept_lows)

    highs = df_full['High']
    lows = df_full['Low']

    # Определяем, находятся ли даты свечей в пределах временного диапазона канала
    # Предполагается, что price_channel.x_start и price_channel.x_end определены,
    # если price_channel.is_valid() вернул True.
    dates_in_channel_range = (
        (candle_dates >= price_channel.x_start) &
        (candle_dates <= price_channel.x_end)
    )


    # Определяем пробои, только если свеча находится внутри временных границ канала
    breakout_up = (highs > upper_channel_values) & dates_in_channel_range
    breakout_down = (lows < lower_channel_values) & dates_in_channel_range

    # Инициализируем статусы (0 - нет пробоя)
    statuses = pd.Series(0, index=df_full.index, name="isBreakout", dtype=int)

    # Устанавливаем статусы пробоя
    statuses[breakout_up] = 1  # Пробой вверх
    statuses[breakout_down] = 2  # Пробой вниз
    # Одновременный пробой (если свеча шире канала и находится внутри временных границ)
    statuses[breakout_up & breakout_down] = 3

    return statuses


def find_static_levels(df,
                       level_type='support',
                       window_size=5,
                       low_col='Low',
                       high_col='High'):
    """
    Находит статические уровни поддержки или сопротивления на основе
    локальных экстремумов, определяемых функцией isPivot.

    Args:
        df (pd.DataFrame): Датафрейм с рыночными данными (должен содержать
                           low_col, high_col).
        level_type (str): Тип уровней для поиска ('support' или 'resistance').
        window_size (int): Размер окна для определения локального экстремума
                           (параметр для isPivot).
        low_col (str): Название столбца с минимальными ценами.
        high_col (str): Название столбца с максимальными ценами.

    Returns:
        pd.Series: Серия со значениями уровней (ценами в точках локальных
                   экстремумов).
    """
    if low_col not in df.columns:
        raise ValueError(f"Столбец '{low_col}' не найден в DataFrame.")
    if high_col not in df.columns:
        raise ValueError(f"Столбец '{high_col}' не найден в DataFrame.")
    if level_type not in ['support', 'resistance']:
        raise ValueError(
            "Параметр 'level_type' должен быть 'support' или 'resistance'."
        )

    temp_df = df.copy()
    # Расчет типов разворота для каждой свечи
    temp_df['pivot_type'] = [
        isPivot(temp_df, i, window_size) for i in range(len(temp_df))
    ]

    levels_series = pd.Series(dtype=float)
    if level_type == 'support':
        # Уровни поддержки: pivot low (2) или both (3)
        support_mask = (temp_df['pivot_type'] == 2) | \
                       (temp_df['pivot_type'] == 3)
        if support_mask.any():
            levels_series = temp_df.loc[support_mask, low_col]
    elif level_type == 'resistance':
        # Уровни сопротивления: pivot high (1) или both (3)
        resistance_mask = (temp_df['pivot_type'] == 1) | \
                          (temp_df['pivot_type'] == 3)
        if resistance_mask.any():
            levels_series = temp_df.loc[resistance_mask, high_col]

    if not levels_series.empty:
        return levels_series.drop_duplicates().sort_values()
    # Возвращаем пустую Series правильного типа, если уровни не найдены
    return pd.Series(dtype=float)


def calculate_support_levels(
        df, static_price_column='Low', static_window_size=5,
        bernoulli_params=None
    ):
    """
    Рассчитывает уровни поддержки для заданного DataFrame.

    Args:
        df (pd.DataFrame): Входной DataFrame с рыночными данными.
        static_price_column (str): Столбец для расчета статических уровней
                                   (например, 'Low').
        static_window_size (int): Размер окна для статических уровней.
        bernoulli_params (dict, optional): Параметры для расчета уровней
                                           поддержки Бернулли.

    Returns:
        dict: Словарь с рассчитанными уровнями поддержки.
              {'static': pd.Series, 'bernoulli': pd.Series (если реализовано)}
    """

    # Для поддержки используется low_col. high_col нужен для find_static_levels,
    # но не используется активно здесь, т.к. level_type='support'.
    static_supports = find_static_levels(
        df,
        level_type='support',
        window_size=static_window_size,
        low_col=static_price_column,  # Используем static_price_column как low_col
        high_col='High'  # Стандартное имя для high_col по умолчанию
    ).to_numpy()

    results = {
        'static': static_supports
    }

    if bernoulli_params:
        # Здесь будет вызов функции для расчета уровней Бернулли
        # bernoulli_supports = calculate_bernoulli_supports(df, **bernoulli_params)
        # results['bernoulli'] = bernoulli_supports
        print("Расчет уровней Бернулли пока не реализован.")
        pass

    return results


def calculate_indicators(df: pd.DataFrame, close_col: str = 'Close') -> dict:
    """
    Рассчитывает технические индикаторы для DataFrame.

    Args:
        df (pd.DataFrame): DataFrame с историческими данными
        close_col (str): Имя колонки с ценами закрытия

    Returns:
        dict: Словарь с рассчитанными индикаторами
    """
    indicators = {}

    # SMA
    indicators['SMA'] = df[close_col].rolling(window=20).mean()

    # EMA
    indicators['EMA'] = df[close_col].ewm(span=20, adjust=False).mean()

    # MACD
    exp1 = df[close_col].ewm(span=12, adjust=False).mean()
    exp2 = df[close_col].ewm(span=26, adjust=False).mean()
    indicators['MACD'] = exp1 - exp2
    indicators['MACD_signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['RSI'] = 100 - (100 / (1 + rs))

    return indicators

def visualize_data(df,
                   offset=1e-3,
                   pivot_highs=None,
                   pivot_lows=None,
                   pivot_both=None,
                   price_ch=None,
                   candle=None,
                   backcandles=None,
                   window=None,
                   open_col='Open',
                   close_col='Close',
                   low_col='Low',
                   high_col='High',
                   pivot_col='isPivot',
                   show_breakout=False,
                   do_show=True,
                   save_path=None,
                   title=None,
                   add_oscilators: list[str] = []
                   ):
    """
    Визуализирует данные о точках разворота и ценовой канал.

    Args:
        df (pd.DataFrame): DataFrame с историческими данными цен
        add_oscilators (list[str]): Список осцилляторов для отображения
                                   Поддерживаемые значения: ["MACD", "SMA", "EMA", "RSI"]
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Определяем, какие осцилляторы нужно отображать в отдельных subplots
    separate_plots = [osc for osc in add_oscilators if osc in ["MACD", "RSI"]]
    n_subplots = len(separate_plots)

    # Создаем subplots
    fig = make_subplots(
        rows=n_subplots + 1,  # +1 для основного графика
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    # Рассчитываем индикаторы
    indicators = calculate_indicators(df, close_col)

    # Добавляем основной график со свечами
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df[open_col],
            high=df[high_col],
            low=df[low_col],
            close=df[close_col],
            name='Candlesticks'
        ),
        row=1, col=1
    )

    # Добавляем SMA и EMA на основной график
    if "SMA" in add_oscilators:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=indicators['SMA'],
                name="SMA(20)",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )

    if "EMA" in add_oscilators:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=indicators['EMA'],
                name="EMA(20)",
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )

    # Добавляем осцилляторы в отдельные subplots
    for i, osc in enumerate(separate_plots, start=2):
        if osc == "MACD":
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=indicators['MACD'],
                    name="MACD",
                    line=dict(color='blue', width=1)
                ),
                row=i, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=indicators['MACD_signal'],
                    name="Signal",
                    line=dict(color='red', width=1)
                ),
                row=i, col=1
            )

        elif osc == "RSI":
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=indicators['RSI'],
                    name="RSI",
                    line=dict(color='purple', width=1)
                ),
                row=i, col=1
            )
            # Добавляем уровни перекупленности/перепроданности
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=i, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=i, col=1)

    # Добавляем линии канала, если они есть
    if price_ch is not None:
        if price_ch.slope_lows != 0 or price_ch.intercept_lows != 0:
            fig.add_trace(
                go.Scattergl(
                    x=price_ch.get_x_channel(),
                    y=price_ch.get_lower_line(),
                    mode='lines',
                    name=f'Lower Channel (R²: {price_ch.r_squared_lows:.2f})',
                    line=dict(color='blue', width=3)
                ),
                row=1, col=1
            )

        if price_ch.slope_highs != 0 or price_ch.intercept_highs != 0:
            fig.add_trace(
                go.Scattergl(
                    x=price_ch.get_x_channel(),
                    y=price_ch.get_upper_line(),
                    mode='lines',
                    name=f'Upper Channel (R²: {price_ch.r_squared_highs:.2f})',
                    line=dict(color='red', width=3)
                ),
                row=1, col=1
            )

    # Обновляем layout
    fig.update_layout(
        title=title or 'Price Channel with Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=300 * (n_subplots + 1)  # Увеличиваем высоту для каждого subplot
    )

    if do_show:
        fig.show()
    if save_path:
        fig.write_image(save_path)

    # --- Конец кода для визуализации ---
