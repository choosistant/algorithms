import pytorch_lightning as pl
import torch
from datasets import load_metric
from transformers import AutoModelForQuestionAnswering


class QuestionAnsweringModel(pl.LightningModule):
    def __init__(
        self, qa_model_name: str = "deepset/roberta-base-squad2", lr: float = 1e-5
    ):
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        self.lr = lr
        self._squad_metric = load_metric("squad")

    def forward(self, x):
        return self.model(input_ids=x["input_ids"], attention_mask=x["attention_mask"])

    def training_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = model_output.loss
        return {"train_loss": loss}

    def validation_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = model_output.loss
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        pred_answer_start = torch.argmax(outputs.start_logits, dim=1)
        pred_answer_end = torch.argmax(outputs.end_logits, dim=1)
        if "start_positions" not in batch or "end_positions" not in batch:
            raise ValueError(
                "Examples in batch is not labelled. Cannot compute accuracy."
            )
        given_answer_start = torch.squeeze(batch["start_positions"])
        given_answer_end = torch.squeeze(batch["end_positions"])
        print(f"pred_start: {pred_answer_start}, pred_end: {pred_answer_end}")
        print(f"answer_start: {given_answer_start}, answer_end: {given_answer_end}")
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


def test_qa_model():
    model = QuestionAnsweringModel()
    print(model)


if __name__ == "__main__":
    test_qa_model()
