import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer, pipeline


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


class AmazonReviewQADataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_path: str,
        qa_model_name: str = "deepset/roberta-base-squad2",
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._file_path = Path(file_path)
        if not self._file_path.exists():
            raise ValueError(f"Data file {self._file_path} does not exist.")

        self._tokenizer = AutoTokenizer.from_pretrained(qa_model_name, use_fast=True)

    def setup(self, stage=None):
        pass

    def prepare_data(self) -> None:
        review_texts, labels = self._parse_data()
        print(f"Found {len(review_texts)} review texts and {len(labels)} labels.")
        for i in range(1):
            print(f"Review text: {review_texts[i]}")
            print(f"Segment: {labels[i].segment}")
            start_idx = labels[i].segment_start
            end_idx = labels[i].segment_end
            review_text = review_texts[i]
            s = review_text[start_idx:end_idx]
            print(f"Extracted: {s}")

    def _parse_data(self) -> Tuple[List[str], List[AnnotatedTextSegment]]:
        review_texts: List[str] = []
        labels: List[AnnotatedTextSegment] = []
        with open(self._file_path, "r") as f:
            items = json.load(f)
            for item in items:
                review_text = item["data"]["reviewText"]
                for annotation_object in item["annotations"]:
                    labeled_segments = annotation_object["result"]
                    for labeled_segment in labeled_segments:
                        vals = labeled_segment["value"]
                        labeled_segment = AnnotatedTextSegment(
                            label=vals["labels"][0],
                            segment_start=vals["start"],
                            segment_end=vals["end"],
                            segment=vals["text"],
                        )
                        review_texts.append(review_text)
                        labels.append(labeled_segment)
        return review_texts, labels


def test_data():
    dm = AmazonReviewQADataModule(
        file_path="data/project-1-at-2022-09-25-08-41-12f06b11.json"
    )
    dm.prepare_data()

    output = dm._tokenizer(
        "This is a table", "What is it", truncation=True, padding=True
    )
    decoded = dm._tokenizer.decode(output["input_ids"])
    print(output)
    print(f"Decoded: {decoded}")

    qa_pipeline = pipeline(
        task="question-answering", model="deepset/roberta-base-squad2"
    )

    print(qa_pipeline.tokenizer)

    output = qa_pipeline.tokenizer.encode(
        question="Who is this?", context="This is a table"
    )
    print(output)

    pass


if __name__ == "__main__":
    test_data()
