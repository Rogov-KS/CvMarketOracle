import torch
import torch.nn as nn
import pytorch_lightning as pl


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        learning_rate: float = 0.001,
        loss_fn: nn.Module = nn.MSELoss(),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # print("\n", lstm_out.shape)
        # predictions = self.fc(lstm_out[:, -1, :])
        predictions = self.fc(lstm_out)
        return predictions

    # def _step(self, batch, batch_idx, stage):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = nn.MSELoss()(y_hat, y)
    #     self.log(f"{stage}_loss", loss,
    #              prog_bar=True, on_step=True, on_epoch=True)
    #     return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(f"train_loss", loss,
                 prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Calculate real quantile - % of values where prediction was higher than actual
        higher_than_actual = (y_hat > y).float().mean()
        lower_than_actual = (y_hat < y).float().mean()

        self.log(f"{stage}_loss", loss,
                 prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_higher_ratio", higher_than_actual,
                 prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_lower_ratio", lower_than_actual,
                 prog_bar=True, on_step=True, on_epoch=True)

        return_dict = {
                       f"{stage}_loss": loss,
                       f"{stage}_higher_ratio": higher_than_actual,
                       f"{stage}_lower_ratio": lower_than_actual
                      }
        return return_dict

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
