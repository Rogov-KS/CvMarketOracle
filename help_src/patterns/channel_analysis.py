import numpy as np
import pandas as pd
from typing import List, Dict
from .src import isPivot, collect_channel, refine_channel
from .classes import PriceChannel
from tqdm import tqdm


def analyze_price_channels(
    df: pd.DataFrame,
    target_candle: int,
    window: int = 3,
    min_backcandles: int = 20,
    max_backcandles: int = 100,
    backcandles_step: int = 5,
    should_refine: bool = True
) -> List[Dict]:
    """
    Анализирует ценовые каналы для разных пивотов и размеров окна.

    Args:
        df (pd.DataFrame): DataFrame с историческими данными цен
        target_candle (int): Индекс свечи, для которой анализируем каналы
        window (int): Размер окна для определения пивотов
        min_backcandles (int): Минимальное количество свечей для анализа
        max_backcandles (int): Максимальное количество свечей для анализа
        backcandles_step (int): Шаг для перебора количества свечей
        should_refine (bool): Нужно ли уточнять каналы

    Returns:
        List[Dict]: Список словарей с результатами анализа каналов
    """
    # Проверяем, есть ли уже колонка с пивотами
    if 'isPivot' not in df.columns:
        # Находим все пивоты в диапазоне до целевой свечи
        pivot_values = [isPivot(df, i, window) for i in range(target_candle)]
        df['isPivot'] = pivot_values

    # Получаем индексы всех пивотов
    pivot_indices = np.arange(len(df))[df['isPivot'] > 0].tolist()

    results = []
    # Для каждого пивота
    for pivot_idx in tqdm(pivot_indices):
        # Для каждого размера окна
        # Рассчитываем backcandles как разницу между target_candle и pivot_idx
        backcandles = target_candle - pivot_idx

        # Если pivot_idx правее target_candle или backcandles меньше минимального, пропускаем
        if (df.iloc[pivot_idx]['Date'] > df.iloc[target_candle]['Date']) or (backcandles < min_backcandles):
            continue

        # Если backcandles больше максимального, берем максимальное значение
        # if backcandles > max_backcandles:
        #     backcandles = max_backcandles

        # Строим канал
        channel = collect_channel(df, target_candle, backcandles, window)

        # Если канал невалидный, пропускаем
        if not channel.is_valid:
            continue

        # Уточняем канал если нужно
        if should_refine:
            channel = refine_channel(channel, should_refine=True)
        # Получаем все точки в диапазоне канала
        channel_points = df.iloc[channel.start_idx:channel.end_idx]


        # Фильтруем только пивоты
        channel_pivots = channel_points[channel_points['isPivot'] > 0]

        if len(channel_pivots) < 2:  # Нужно минимум 2 пивота для оценки
            continue

        # Получаем ошибки только для пивотов
        high_pivots = channel_pivots[channel_pivots['isPivot'].isin([1, 3])]
        high_pred = channel.get_upper_line(high_pivots['Date'])

        low_pivots = channel_pivots[channel_pivots['isPivot'].isin([2, 3])]
        low_pred = channel.get_lower_line(low_pivots['Date'])

        high_errors = high_pivots['High'].values - high_pred
        low_errors = low_pivots['Low'].values - low_pred

        # Метрики
        high_mse = (np.mean(high_errors ** 2)
                   if len(high_errors) > 0 else np.inf)
        low_mse = (np.mean(low_errors ** 2)
                  if len(low_errors) > 0 else np.inf)
        high_mae = (np.mean(np.abs(high_errors))
                   if len(high_errors) > 0 else np.inf)
        low_mae = (np.mean(np.abs(low_errors))
                  if len(low_errors) > 0 else np.inf)

        # Сохраняем результаты
        results.append({
            'pivot_idx': pivot_idx,
            'backcandles': backcandles,
            'channel': channel,
            'high_mse': high_mse,
            'low_mse': low_mse,
            'high_mae': high_mae,
            'low_mae': low_mae,
            'num_high_pivots': len(high_pivots),
            'num_low_pivots': len(low_pivots),
            'r_squared_highs': channel.r_squared_highs,
            'r_squared_lows': channel.r_squared_lows
        })

    return results