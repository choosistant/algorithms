import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch


@dataclass
class AnnotatedTextSegment:
    label: str
    segment_start: int
    segment_end: int
    segment: str


@dataclass
class AnnotatedExample:
    review_text: str
    labels: List[AnnotatedTextSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "review_text": self.review_text,
            "labels": [x.__dict__ for x in self.labels],
        }


def parse_annotation_from_label_studio(exported_annotation_item) -> dict:
    review_text = exported_annotation_item["data"]["reviewText"]
    example = AnnotatedExample(review_text=review_text)

    for annotation_object in exported_annotation_item["annotations"]:
        labeled_segments = annotation_object["result"]
        for labeled_segment in labeled_segments:
            vals = labeled_segment["value"]
            example.labels.append(
                AnnotatedTextSegment(
                    label=vals["labels"][0],
                    segment_start=vals["start"],
                    segment_end=vals["end"],
                    segment=vals["text"],
                )
            )
    return example.to_dict()


class AmazonReviewLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str):
        self._file_path = Path(file_path)
        if not self._file_path.exists():
            raise ValueError(f"File {self._file_path} does not exist.")

        self._data = self._load_data()

    def _load_data(self):
        with open(self._file_path, "r") as f:
            data = json.load(f)
            examples = [parse_annotation_from_label_studio(a) for a in data]
        return examples

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class AmazonReviewEvaluationDataModule(pl.LightningDataModule):
    # Data modules decouple data-related logic from the model module.
    # The aim is to make the model logic work with different datasets
    # without chaning the model module.

    def __init__(self, data_set: torch.utils.data.Dataset, batch_size: int = 32):
        super().__init__()
        self._data_set = data_set
        self._batch_size = batch_size

    def setup(self, stage=None):
        pass

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self._data_set,
            batch_size=self._batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
        )
