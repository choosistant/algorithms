import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import evaluate as huggingface_evaluate
import nltk
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.qa import QuestionAnsweringModel

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class AnnotatedTextSegment:
    label: str
    segment_start: int
    segment_end: int
    segment: str


@dataclass
class AnnotatedExample:
    id: int
    item_id: str
    review_text: str
    labels: List[AnnotatedTextSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "review_text": self.review_text,
            "labels": [x.__dict__ for x in self.labels],
        }


class AmazonReviewQADataset(Dataset):
    def __init__(self, items: List[Dict[str, torch.Tensor]]) -> None:
        super().__init__()
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


@dataclass(frozen=True)
class QuestionAnsweringModelInput(dict):
    example_ids: torch.Tensor
    input_ids: torch.tensor
    attention_mask: torch.tensor
    context_mask: torch.tensor
    start_positions: torch.tensor
    end_positions: torch.tensor

    def to_dict(self):
        return self.__dict__


class QuestionAnsweringInputEncoder:
    def __init__(self, tokenizer, doc_stride: int = 128) -> None:
        self._tokenizer = tokenizer

        # The model input is limited to 512 tokens. We need to split the document into
        # sub documents if it is longer than 512 tokens.
        self._doc_max_length = self._tokenizer.model_max_length

        # When splitting the document into sub documents, we need to overlap the
        # sub documents by a few tokens to make sure that the answer is not split
        # between two sub documents. The stride determines how many tokens we
        # overlap. Stride determines how many tokens a document text can overlap
        # between any parts when splitting it into separate sub documents.
        self._doc_stride = doc_stride

        self._question_map = {
            "benefit": [
                "What is the first benefit of this item, if any?",
                "What is the second benefit of this item, if any?",
                "What is the third benefit of this item, if any?",
            ],
            "drawback": [
                "What is the first drawback of this item, if any?",
                "What is the second drawback of this item, if any?",
                "What is the third drawback of this item, if any?",
            ],
        }

    def encode(
        self,
        input_text: str,
        example_id: int,
        labels: List[AnnotatedTextSegment] = None,
    ) -> List[QuestionAnsweringModelInput]:

        ret_input = []

        # Chunks are the sub documents that we will feed to the model.
        # It may be useful to split the document into sub documents.
        # For now, we just use the entire document as a single chunk.
        chunk_start_idx, chunk_end_idx = 0, len(input_text)

        for label_val, questions in self._question_map.items():
            answer_ranges = self._extract_answers(
                chunk_start_idx=chunk_start_idx,
                chunk_end_idx=chunk_end_idx,
                labels=labels,
                label_val=label_val,
            )

            for q_idx, question in enumerate(questions):
                if len(answer_ranges) > q_idx:
                    answer_range_for_question = [answer_ranges[q_idx]]
                else:
                    answer_range_for_question = []

                encoded_inputs = self._encode_qa_input(
                    context=input_text,
                    question=question,
                    answer_ranges=answer_range_for_question,
                    example_id=example_id,
                )
                assert len(encoded_inputs) >= 1
                ret_input += encoded_inputs

        return ret_input

    def _encode_qa_input(
        self,
        context: str,
        question: str,
        answer_ranges: List[Tuple[int, int]],
        example_id: Optional[int] = None,
    ) -> List[QuestionAnsweringModelInput]:
        encoded_qa_inputs = []

        encoded_question_and_context = self._tokenizer.encode_plus(
            question,
            context,
            truncation="only_second",
            max_length=self._doc_max_length,
            stride=self._doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        offset_mapping = encoded_question_and_context.pop("offset_mapping")
        # Remove `overflow_to_sample_mapping` key
        encoded_question_and_context.pop("overflow_to_sample_mapping")

        for i, offsets in enumerate(offset_mapping):
            seq_ids = np.array(encoded_question_and_context.sequence_ids(i))
            context_mask = seq_ids == 1
            context_token_indices = np.where(context_mask)[0]
            context_start_idx = context_token_indices[0]
            context_end_idx = context_token_indices[-1]

            input_ids = encoded_question_and_context["input_ids"][i]
            attention_masks = encoded_question_and_context["attention_mask"][i]

            # Assume that the [CLS] token is at the beginning of the input.
            # Alternatively, we can find the index of the [CLS] token:
            #   input_ids.index(self._tokenizer.cls_token_id)
            bos_index = 0

            # If there is no answer, we set the start and end positions
            # to the [CLS] token.
            if len(answer_ranges) == 0:
                encoded_qa_inputs.append(
                    QuestionAnsweringModelInput(
                        example_ids=torch.tensor([example_id]),
                        input_ids=input_ids,
                        attention_mask=attention_masks,
                        context_mask=context_mask,
                        start_positions=torch.tensor([bos_index]),
                        end_positions=torch.tensor([bos_index]),
                    )
                )
                continue

            context_boundary_start = offsets[context_start_idx][0]
            context_boundary_end = offsets[context_end_idx][1]

            for answer_start_idx, answer_end_idx in answer_ranges:
                if (
                    context_boundary_start > answer_start_idx
                    or context_boundary_end < answer_end_idx
                ):

                    encoded_qa_inputs.append(
                        QuestionAnsweringModelInput(
                            example_ids=torch.tensor([example_id]),
                            input_ids=input_ids,
                            attention_mask=attention_masks,
                            context_mask=context_mask,
                            start_positions=torch.tensor([bos_index]),
                            end_positions=torch.tensor([bos_index]),
                        )
                    )
                else:
                    token_start_index = context_start_idx
                    while offsets[token_start_index][0] < answer_start_idx:
                        token_start_index += 1

                    token_end_index = context_end_idx
                    while offsets[token_end_index][1] > answer_end_idx:
                        token_end_index -= 1
                    token_end_index += 1

                    encoded_qa_inputs.append(
                        QuestionAnsweringModelInput(
                            example_ids=torch.tensor([example_id]),
                            input_ids=input_ids,
                            attention_mask=attention_masks,
                            context_mask=context_mask,
                            start_positions=torch.tensor([token_start_index]),
                            end_positions=torch.tensor([token_end_index]),
                        )
                    )

        return encoded_qa_inputs

    def _extract_answers(
        self,
        chunk_start_idx: int,
        chunk_end_idx: int,
        labels: List[AnnotatedTextSegment],
        label_val: str,
    ) -> List[Tuple[int, int]]:
        answers = []
        if labels is not None and len(labels) > 0:
            for current_annoted_segment in labels:
                if current_annoted_segment.label != label_val:
                    continue
                if (
                    current_annoted_segment.segment_start >= chunk_start_idx
                    and current_annoted_segment.segment_end <= chunk_end_idx
                ):
                    answer_len = (
                        current_annoted_segment.segment_end
                        - current_annoted_segment.segment_start
                    )
                    answer_start_idx = (
                        current_annoted_segment.segment_start - chunk_start_idx
                    )
                    answer_end_idx = answer_start_idx + answer_len
                    answers.append((answer_start_idx, answer_end_idx))
        return answers


class AmazonReviewQADataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_path: str,
        tokenizer,
        batch_size: int = 32,
        doc_stride: int = 128,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self._file_path = Path(file_path)
        if not self._file_path.exists():
            raise ValueError(f"Data file {self._file_path} does not exist.")

        self._verbose = verbose

        self._batch_size = batch_size

        self._tokenizer = tokenizer

        self._prepared_data: List[Dict[str, torch.Tensor]] = []

    def setup(self, stage=None):
        nltk.download("punkt")

    def prepare_data(self) -> None:
        examples = self._parse_annoted_examples()
        if self._verbose:
            print(f"Found {len(examples)} examples.")

        self._prepared_data.clear()

        encoder = QuestionAnsweringInputEncoder(self._tokenizer)

        for example in examples:
            encoded_inputs = encoder.encode(
                input_text=example.review_text,
                example_id=example.id,
                labels=example.labels,
            )

            self._prepared_data += [item.to_dict() for item in encoded_inputs]

        if self._verbose:
            print(f"Prepared {len(self._prepared_data)} examples.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.get_data_set(),
            batch_size=self._batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.get_data_set(),
            batch_size=self._batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_data_set(self) -> AmazonReviewQADataset:
        return AmazonReviewQADataset(self._prepared_data)

    def _split_text_into_chunks_by_sentence_pairings(
        self, review_text: str, n_sentences_to_pair: int = 3
    ) -> List[Tuple[int, int]]:
        sentences = nltk.sent_tokenize(review_text, language="english")

        print(f"Found {len(sentences)} sentences in example.")

        chunks: List[Tuple[int, int]] = []

        cur_start_index = 0
        offset = n_sentences_to_pair - 1
        for j in range(len(sentences) - offset):
            sentence_start = sentences[j]
            sentence_end = sentences[j + offset]

            chunk_start_idx = review_text.index(sentence_start, cur_start_index)
            chunk_end_idx = review_text.index(sentence_end, cur_start_index) + len(
                sentence_end
            )
            chunks.append((chunk_start_idx, chunk_end_idx))
            cur_start_index = chunk_start_idx
        return chunks

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

    def _parse_annoted_examples(self) -> List[AnnotatedExample]:
        with open(self._file_path, "r") as f:
            items = json.load(f)
            for i in items:
                print(i)
                print(type(i))
            exit(1)
            return [self._parse_annotated_example(item) for item in items]

    def _parse_annotated_example(self, exported_annotation_item) -> AnnotatedExample:
        example = AnnotatedExample(
            id=exported_annotation_item["data"]["index"],
            item_id=exported_annotation_item["data"]["asin"],
            review_text=exported_annotation_item["data"]["reviewText"],
        )

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
        return example


def compute_metrics(predictions, tokenizer, verbose=False):
    predicted_answers = []
    given_answers = []

    for batch_predictions in predictions:
        example_ids = batch_predictions["example_id"]

        for j in range(len(example_ids)):
            example_id = example_ids[j]
            input_ids = batch_predictions["input_ids"][j]

            given_answer_start = batch_predictions["given_answer_start"][j]
            given_answer_end = batch_predictions["given_answer_end"][j]
            pred_answer_start = batch_predictions["pred_answer_start"][j]
            pred_answer_end = batch_predictions["pred_answer_end"][j]
            pred_score = batch_predictions["pred_score"][j]

            predicted_text = tokenizer.decode(
                input_ids[pred_answer_start:pred_answer_end]
            )
            given_text = tokenizer.decode(
                input_ids[given_answer_start:given_answer_end]
            )

            if verbose:
                print("\n")
                print(f"Example ID: {example_id:10d}  ", end="")
                print(
                    f"given: [{given_answer_start:3d}, {given_answer_end:3d}] ", end=""
                )
                print(f"pred: [{pred_answer_start:3d}, {pred_answer_end:3d}] ", end="")
                print(f"pred score: {pred_score:0.4f}")

                input_text = tokenizer.decode(input_ids)
                input_text = input_text.replace("<pad>", "")
                print(f"  Input: {input_text}")
                print(f"  Annotated answer: {given_text}")
                print(f"  Predicted answer: {predicted_text}")

            predicted_answers.append(
                {"id": str(example_id), "prediction_text": predicted_text}
            )
            given_answers.append(
                {
                    "id": str(example_id),
                    "answers": {
                        "text": [given_text],
                        "answer_start": [int(given_answer_start)],
                    },
                }
            )

    squad_metric = huggingface_evaluate.load("squad")
    metric_output = squad_metric.compute(
        predictions=predicted_answers, references=given_answers
    )

    if verbose:
        print(metric_output)

    return metric_output


def test_data():
    model = QuestionAnsweringModel()

    dm = AmazonReviewQADataModule(
        file_path="data/sample.json", tokenizer=model.tokenizer, verbose=False
    )
    dm.setup()
    dm.prepare_data()

    print("Making predictions...")
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1)
    predictions = trainer.test(model, dm)

    compute_metrics(predictions, model.tokenizer, verbose=True)

    pass


if __name__ == "__main__":
    test_data()
