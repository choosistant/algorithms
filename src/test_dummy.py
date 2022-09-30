import pytorch_lightning as pl

from src.data import AmazonReviewEvaluationDataModule, AmazonReviewLabeledDataset
from src.models.dummy import DummyModel


def test_dummy_model():
    ds = AmazonReviewLabeledDataset(file_path="../data/dataset.json")
    dm = AmazonReviewEvaluationDataModule(data_set=ds, batch_size=1)

    trainer = pl.Trainer(max_epochs=1, gpus=0)

    model = DummyModel()
    trainer.test(model=model, dataloaders=dm)


if __name__ == "__main__":
    test_dummy_model()
