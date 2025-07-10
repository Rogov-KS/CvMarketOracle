import hydra
from omegaconf import DictConfig
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from market_oracle_lib.data.data_classes import MyDataModule
from market_oracle_lib.model import LSTMModel
from market_oracle_lib.parse_conf import get_loggers, get_callbacks


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    pl.seed_everything(42)

    dm = MyDataModule(config=config)

    model = LSTMModel(
        input_size=config["model"]["input_size"],
        hidden_size=config["model"]["hidden_size"],
        output_size=config["model"]["output_size"],
    )

    loggers = get_loggers(config["logging"]["loggers_names"], config["artifacts"]["experiment_name"])

    callbacks = get_callbacks(config["callbacks"]["callbacks_names"])

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join("./checkpoints", "my_test_checkpoint"),
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=config["training"]["save_top_k"],
            every_n_train_steps=config["callbacks"]["model_checkpoint"]["every_n_train_steps"],
            every_n_epochs=config["callbacks"]["model_checkpoint"]["every_n_epochs"],
        )
    )

    trainer = pl.Trainer(
        accelerator=config["training"]["accelerator"],
        precision=config["training"]["trainer"]["precision"],
        max_epochs=config["training"]["trainer"]["max_epochs"],
        accumulate_grad_batches=config["training"]["trainer"]["accumulate_grad_batches"],
        val_check_interval=config["training"]["trainer"]["val_check_interval"],
        overfit_batches=config["training"]["trainer"]["overfit_batches"],
        num_sanity_val_steps=config["training"]["trainer"]["num_sanity_val_steps"],
        deterministic=config["training"]["trainer"]["deterministic"],
        benchmark=config["training"]["trainer"]["benchmark"],
        gradient_clip_val=config["training"]["trainer"]["gradient_clip_val"],
        profiler=None,
        log_every_n_steps=config["training"]["trainer"]["log_every_n_steps"],
        detect_anomaly=config["training"]["trainer"]["detect_anomaly"],
        enable_checkpointing=config["training"]["trainer"]["enable_checkpointing"],
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
