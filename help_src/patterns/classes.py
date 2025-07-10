from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PriceChannel:
    """Датакласс для представления ценового канала."""
    # Датафрейм, в котором построен канал
    df: pd.DataFrame = None
    # Имя столбца даты
    x_name: str = 'Date'

    # Параметры нижней линии тренда
    slope_lows: float = 0.0
    intercept_lows: float = 0.0
    r_squared_lows: float = 0.0

    # Параметры верхней линии тренда
    slope_highs: float = 0.0
    intercept_highs: float = 0.0
    r_squared_highs: float = 0.0

    # Индексы начала и конца канала
    start_idx: int = 0
    end_idx: int = 0

    def get_lower_line(self, x_series: pd.Series=None) -> np.ndarray:
        """
        Возвращает x и y координаты для построения нижней линии тренда.

        Returns:
            tuple[pd.Series, pd.Series]: Кортеж из (x, y) координат
        """
        diff_x = self.get_x_diff(x_series)
        y_low = self.slope_lows * diff_x + self.intercept_lows
        return y_low

    def get_upper_line(self, x_series: pd.Series=None) -> np.ndarray:
        """
        Возвращает x и y координаты для построения верхней линии тренда.

        Returns:
            tuple[pd.Series, pd.Series]: Кортеж из (x, y) координат
        """
        diff_x = self.get_x_diff(x_series)
        y_high = self.slope_highs * diff_x + self.intercept_highs
        return y_high

    def get_x_diff(self, x_series: pd.Series=None) -> np.array:
        """
        Возвращает разницу между x координатами для построения канала.
        """
        if x_series is None:
            x_channel = self.get_x_channel()
        else:
            x_channel = x_series
        return (x_channel - self.x_start).dt.total_seconds()

    def get_x_channel(self) -> np.array:
        """
        Возвращает x координаты для построения канала.
        """
        return self.df[self.x_name][self.start_idx : self.end_idx]

    @property
    def x_start(self):
        return self.df[self.x_name].iloc[self.start_idx]

    @property
    def x_end(self):
        return self.df[self.x_name].iloc[self.end_idx]

    def is_valid(self) -> bool:
        """
        Проверяет, является ли канал валидным.

        Returns:
            bool: True если канал валидный, False иначе
        """
        return (
            self.slope_lows != 0.0 and
            self.slope_highs != 0.0 and
            self.start_idx < self.end_idx
        )

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"x_name='{self.x_name}', "
                f"slope_lows={self.slope_lows:.4f}, "
                f"intercept_lows={self.intercept_lows:.4f}, "
                f"r_squared_lows={self.r_squared_lows:.4f}, "
                f"slope_highs={self.slope_highs:.4f}, "
                f"intercept_highs={self.intercept_highs:.4f}, "
                f"r_squared_highs={self.r_squared_highs:.4f}, "
                f"start_idx={self.start_idx}, "
                f"end_idx={self.end_idx}, "
                f"df_shape={self.df.shape if self.df is not None else None})")
