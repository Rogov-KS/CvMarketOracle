import hydra
from omegaconf import DictConfig
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from market_oracle_lib.data.data_classes import MyDataModule
from market_oracle_lib.model import LSTMModel
import torch


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    pl.seed_everything(42)

    # Инициализация датамодуля
    dm = MyDataModule(config=config)
    dm.setup()

    # Загрузка чекпоинта
    checkpoint_path = os.path.join("./checkpoints", "my_test_checkpoint",
                                   "epoch=00-val_loss=172009.4531.ckpt")
    model = LSTMModel.load_from_checkpoint(
        checkpoint_path,
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        output_size=config["model"]["output_size"],
    )
    model.eval()
    model.to('cpu')
    # Получение одного батча данных
    X, y = next(iter(dm.val_dataloader()))
    print(f"X: {X}")
    print(f"X.shape: {X.shape}")
    print(f"y: {y}")
    print(f"y.shape: {y.shape}")

    # Применение модели
    with torch.no_grad():
        predictions = model(X.to('cpu'))

    print(f"predictions: {predictions}")
    return predictions


if __name__ == "__main__":
    main()
