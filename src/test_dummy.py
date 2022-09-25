import pytorch_lightning as pl

from src.algorithms.dummy import DummyModel
from src.data import AmazonReviewEvaluationDataModule, AmazonReviewLabeledDataset


def test_dummy_model():
    ds = AmazonReviewLabeledDataset(
        file_path="data/project-1-at-2022-09-25-08-41-12f06b11.json"
    )
    dm = AmazonReviewEvaluationDataModule(data_set=ds, batch_size=1)

    trainer = pl.Trainer(max_epochs=1, gpus=0)

    model = DummyModel()
    trainer.test(model=model, dataloaders=dm)


if __name__ == "__main__":
    test_dummy_model()
