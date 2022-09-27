import pytorch_lightning as pl
import torch
from transformers import AutoModelForQuestionAnswering


class QuestionAnsweringModel(pl.LightningModule):
    def __init__(
        self, qa_model_name: str = "deepset/roberta-base-squad2", lr: float = 1e-5
    ):
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        self.lr = lr

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs[0]
        return {"train_loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


def test_qa_model():
    model = QuestionAnsweringModel()
    print(model)


if __name__ == "__main__":
    test_qa_model()
