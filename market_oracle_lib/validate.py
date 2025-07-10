import os
import pandas as pd
import torch
import numpy as np
import datetime
import csv
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def validate_model(
    model: pl.LightningModule,
    dataloader: DataLoader,
    results_file: str = "validation_results.csv",
    run_name: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Валидация модели Lightning на данных и сохранение результатов в CSV файл.

    Args:
        model: PyTorch Lightning модель
        dataloader: DataLoader с данными для валидации
        results_file: Путь к CSV файлу для сохранения результатов
        run_name: Название эксперимента (если None, будет использован текущий timestamp)
        additional_info: Дополнительная информация для сохранения (гиперпараметры и т.д.)

    Returns:
        Dict[str, float]: Словарь с метриками валидации
    """
    # Установка модели в режим оценки
    model.eval()
    device = next(model.parameters()).device

    # Подготовка для сбора предсказаний и истинных значений
    all_preds = []
    all_targets = []

    # Проход по данным без вычисления градиентов
    with torch.no_grad():
        for batch in dataloader:
            # Получаем входные данные и целевые значения
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Получаем предсказания модели
            outputs = model(x)

            # Собираем предсказания и целевые значения
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Объединяем предсказания и целевые значения
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Расчет метрик
    metrics = {}
    metrics['mse'] = mean_squared_error(all_targets, all_preds)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(all_targets, all_preds)
    metrics['r2'] = r2_score(all_targets, all_preds)

    # Если модель имеет собственные метрики, используем их тоже
    if hasattr(model, 'compute_metrics'):
        model_metrics = model.compute_metrics(torch.tensor(all_preds), torch.tensor(all_targets))
        for k, v in model_metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.item()
            else:
                metrics[k] = v

    # Получение текущего времени
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Если имя запуска не указано, используем текущее время
    if run_name is None:
        run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Создаем строку результатов
    results_row = {
        'timestamp': timestamp,
        'run_name': run_name,
        **metrics
    }

    # Добавляем дополнительную информацию, если она предоставлена
    if additional_info:
        for key, value in additional_info.items():
            results_row[key] = value

    # Записываем результаты в CSV файл
    results_path = Path(results_file)
    file_exists = results_path.exists()

    with open(results_path, 'a', newline='') as csvfile:
        fieldnames = list(results_row.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Если файл не существует, записываем заголовки
        if not file_exists:
            writer.writeheader()

        # Записываем данные
        writer.writerow(results_row)

    print(f"Validation results saved to {results_file}")
    print(f"Metrics: {metrics}")

    return metrics

def validate_and_visualize(
    model: pl.LightningModule,
    dataloader: DataLoader,
    results_file: str = "validation_results.csv",
    save_figures: bool = True,
    figures_dir: str = "validation_figures",
    run_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Расширенная валидация модели с визуализацией результатов

    Args:
        model: PyTorch Lightning модель
        dataloader: DataLoader с данными для валидации
        results_file: Путь к CSV файлу для сохранения результатов
        save_figures: Сохранять ли графики
        figures_dir: Директория для сохранения графиков
        run_name: Название эксперимента

    Returns:
        Dict[str, float]: Словарь с метриками валидации
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Настройка стиля графиков
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    except ImportError:
        print("Matplotlib или Seaborn не установлены. Визуализация отключена.")
        save_figures = False

    # Установка модели в режим оценки
    model.eval()
    device = next(model.parameters()).device

    # Подготовка для сбора предсказаний и истинных значений
    all_preds = []
    all_targets = []

    # Проход по данным без вычисления градиентов
    with torch.no_grad():
        for batch in dataloader:
            # Получаем входные данные и целевые значения
            x, y = batch
            x, y = x.to(device), y.to(device)

            # Получаем предсказания модели
            outputs = model(x)

            # Собираем предсказания и целевые значения
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Объединяем предсказания и целевые значения
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Если run_name не указан, генерируем его
    if run_name is None:
        run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Расчет метрик
    metrics = {}
    metrics['mse'] = mean_squared_error(all_targets, all_preds)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(all_targets, all_preds)
    metrics['r2'] = r2_score(all_targets, all_preds)

    # Запись результатов в CSV
    metrics_for_csv = validate_model(
        model=model,
        dataloader=dataloader,
        results_file=results_file,
        run_name=run_name,
        additional_info={
            'model_type': model.__class__.__name__,
            'n_samples': len(all_targets)
        }
    )

    # Визуализация результатов
    if save_figures:
        # Создаем директорию для графиков, если она не существует
        os.makedirs(figures_dir, exist_ok=True)

        # График сравнения предсказанных и истинных значений
        plt.figure(figsize=(10, 6))
        plt.scatter(all_targets, all_preds, alpha=0.5)
        plt.plot([-3, 3], [-3, 3], 'r--')  # Линия y=x
        plt.xlabel('Истинные значения')
        plt.ylabel('Предсказанные значения')
        plt.title(f'Предсказания vs. Истинные значения (R² = {metrics["r2"]:.4f})')
        plt.grid(True)
        plt.savefig(f"{figures_dir}/{run_name}_predictions.png")

        # Гистограмма ошибок
        errors = all_preds - all_targets
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.xlabel('Ошибка предсказания')
        plt.ylabel('Частота')
        plt.title(f'Распределение ошибок (MAE = {metrics["mae"]:.4f}, RMSE = {metrics["rmse"]:.4f})')
        plt.grid(True)
        plt.savefig(f"{figures_dir}/{run_name}_errors.png")

        plt.close('all')

    return metrics

# if __name__ == "__main__":
#     # Пример использования
#     import argparse

#     parser = argparse.ArgumentParser(description='Validate a PyTorch Lightning model')
#     parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
#     parser.add_argument('--data_path', type=str, required=True, help='Path to the validation data')
#     parser.add_argument('--results_file', type=str, default='validation_results.csv', help='Path to the results CSV file')
#     parser.add_argument('--run_name', type=str, default=None, help='Name of the validation run')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size for validation')
#     parser.add_argument('--visualize', action='store_true', help='Generate visualizations of validation results')

#     args = parser.parse_args()

#     # Здесь можно добавить загрузку модели и данных и запустить валидацию
#     print("Example usage:")
#     print("python validate.py --model_path=models/my_model.ckpt --data_path=data/validation.csv --run_name=experiment1 --visualize")
