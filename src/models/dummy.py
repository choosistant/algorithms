import pytorch_lightning as pl
import torch


class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Return random number for forward
        return torch.rand(1)

    def loss(self, batch, prediction):
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"train_loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        torch.stack([x["train_loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x["val_loss"] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["test_loss"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
