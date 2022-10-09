import pytorch_lightning as pl
import torch

from src.data import AmazonReviewQADataModule
from src.models.qa import QuestionAnsweringModel


def fine_tune_model():
    model = QuestionAnsweringModel(qa_model_name='bert-base-cased')

    dm = AmazonReviewQADataModule(
        file_path="data/sample.json",
        tokenizer=model.tokenizer,
        batch_size=8,
        verbose=False,
    )
    dm.setup()
    dm.prepare_data()

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )
    trainer.fit(model=model, datamodule=dm)

    predictions = trainer.predict(model, dm)

    metrics = model.compute_metrics(predictions)
    print(metrics)


if __name__ == "__main__":
    fine_tune_model()
