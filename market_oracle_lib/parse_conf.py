import os
from pytorch_lightning.loggers import CSVLogger, WandbLogger, MLFlowLogger
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    # ModelCheckpoint,
    RichModelSummary,
    DeviceStatsMonitor,
    RichProgressBar,
    Callback
)
from market_oracle_lib.data import apimoex, yfinance
from market_oracle_lib.consts import (
    US_DEFAULT_SYMBOLS,
    RU_DEFAULT_SYMBOLS
)


def get_data_getter_fn(data_getter_fn_name: str):
    if data_getter_fn_name == "apimoex":
        return apimoex.get_symbol_data
    elif data_getter_fn_name == "yfinance":
        return yfinance.get_symbol_data
    else:
        err_str = f"Invalid data getter function name: {data_getter_fn_name}"
        raise ValueError(err_str)


def get_data_symbols(data_getter_fn_name: str):
    if data_getter_fn_name == "apimoex":
        return RU_DEFAULT_SYMBOLS
    elif data_getter_fn_name == "yfinance":
        return US_DEFAULT_SYMBOLS
    else:
        err_str = f"Invalid data symbols name: {data_getter_fn_name}"
        raise ValueError(err_str)


def get_loggers(logger_names: list[str], exp_name: str) -> list[Logger]:
    loggers: list[Logger] = []
    for name in logger_names:
        if name == "csv":
            csv_logger = CSVLogger(
                save_dir="../.logs/my-csv-logs",
                name=exp_name
            )
            loggers.append(csv_logger)
        elif name == "wandb":
            wandb_logger = WandbLogger(
                project="mlops-logging-demo",
                name=exp_name,
                save_dir="../.logs/my-wandb-logs",
            )
            loggers.append(wandb_logger)
        elif name == "mlflow":
            mlflow_logger = MLFlowLogger(
                experiment_name=exp_name,
                tracking_uri="http://127.0.0.1:8080",
                save_dir="../.logs/my-mlflow-logs",
            )
            loggers.append(mlflow_logger)
        else:
            raise ValueError(f"Invalid logger name: {name}")
    return loggers


def get_callbacks(callback_names: list[str]) -> list[Callback]:
    callbacks: list[Callback] = []
    for name in callback_names:
        if name == "learning_rate_monitor":
            callbacks.append(LearningRateMonitor(logging_interval="step"))
        elif name == "device_stats_monitor":
            callbacks.append(DeviceStatsMonitor())
        elif name == "rich_model_summary":
            callbacks.append(RichModelSummary(max_depth=2))
        elif name == "rich_progress_bar":
            callbacks.append(RichProgressBar())
        # elif name == "model_checkpoint":
        #     callbacks.append(ModelCheckpoint(
        #         dirpath=os.path.join("../checkpoints", "my-test-checkpoint"),
        #         filename="{epoch:02d}-{val_loss:.4f}",
        #         monitor="val_loss",
        #         save_top_k=config["training"]["save_top_k"],
        #     ))
        else:
            raise ValueError(f"Invalid callback name: {name}")
    return callbacks


def print_dict(d: dict):
    for k, v in d.items():
        print(f"{k}: {v}\n\n")
    print("="*25)
