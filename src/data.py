import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.modeling_outputs import QuestionAnsweringModelOutput

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


class AmazonReviewQADataModule(pl.LightningDataModule):
    def __init__(
        self,
        file_path: str,
        qa_model_name: str = "deepset/roberta-base-squad2",
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

        self._tokenizer = AutoTokenizer.from_pretrained(qa_model_name, use_fast=True)

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
            "benefit": "What are the benefits of this item, if any?",
            "drawback": "What are the drawbacks of this item, if any?",
        }

        self._prepared_data: List[Dict[str, torch.Tensor]] = []

    def setup(self, stage=None):
        nltk.download("punkt")

    def prepare_data(self) -> None:
        examples = self._parse_annoted_examples()

        if self._verbose:
            print(f"Found {len(examples)} examples.")

        self._prepared_data.clear()

        for example in examples:
            # Chunks are the sub documents that we will feed to the model.
            # It may be useful to split the document into sub documents.
            # For now, we just use the entire document as a single chunk.
            chunk_start_idx, chunk_end_idx = 0, len(example.review_text)

            # Extract the chunk text from the orig review text.
            chunk = example.review_text[chunk_start_idx:chunk_end_idx]
            if self._verbose:
                print("=" * 80)
                print(f"Chunk: {chunk}")
                print("=" * 80)

            for label_val, question in self._question_map.items():
                if self._verbose:
                    print(f"Question: {question} (label: {label_val})")

                answer_ranges = self._extract_answers(
                    chunk_start_idx=chunk_start_idx,
                    chunk_end_idx=chunk_end_idx,
                    labels=example.labels,
                    label_val=label_val,
                )
                encoded_inputs = self._encode_qa_input(
                    example=example,
                    context=chunk,
                    question=question,
                    answer_ranges=answer_ranges,
                )
                assert len(encoded_inputs) >= 1
                self._prepared_data += encoded_inputs
        if self._verbose:
            print(f"Prepared {len(self._prepared_data)} examples.")

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._get_data_set(),
            batch_size=self._batch_size,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def get_data_set(self) -> AmazonReviewQADataset:
        return AmazonReviewQADataset(self._prepared_data)

    def _extract_answers(
        self,
        chunk_start_idx: int,
        chunk_end_idx: int,
        labels: List[AnnotatedTextSegment],
        label_val: str,
    ) -> List[Tuple[int, int]]:
        answers = []
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

    def _encode_qa_input(
        self,
        example: AnnotatedExample,
        context: str,
        question: str,
        answer_ranges: List[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
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

            if self._verbose:
                print(f"  Decoded input: {self._tokenizer.decode(input_ids)}")

            # Assume that the [CLS] token is at the beginning of the input.
            # Alternatively, we can find the index of the [CLS] token:
            #   input_ids.index(self._tokenizer.cls_token_id)
            bos_index = 0

            # If there is no answer, we set the start and end positions
            # to the [CLS] token.
            if len(answer_ranges) == 0:
                encoded_qa_inputs.append(
                    {
                        "example_ids": torch.tensor([example.id]),
                        "input_ids": input_ids,
                        "attention_mask": attention_masks,
                        "context_mask": context_mask,
                        "start_positions": torch.tensor([bos_index]),
                        "end_positions": torch.tensor([bos_index]),
                    }
                )
                continue

            context_boundary_start = offsets[context_start_idx][0]
            context_boundary_end = offsets[context_end_idx][1]

            for answer_start_idx, answer_end_idx in answer_ranges:
                item = {
                    "example_ids": torch.tensor([example.id]),
                    "input_ids": input_ids,
                    "attention_mask": attention_masks,
                    "context_mask": context_mask,
                }
                encoded_qa_inputs.append(item)

                if (
                    context_boundary_start > answer_start_idx
                    or context_boundary_end < answer_end_idx
                ):
                    if self._verbose:
                        print("   No benefit answer in this chunk!")

                    item["start_positions"] = torch.tensor([bos_index])
                    item["end_positions"] = torch.tensor([bos_index])
                else:
                    if self._verbose:
                        original_answer = context[answer_start_idx:answer_end_idx]
                        print(f"  Original answer: {original_answer}")

                    token_start_index = context_start_idx
                    while offsets[token_start_index][0] < answer_start_idx:
                        token_start_index += 1

                    token_end_index = context_end_idx
                    while offsets[token_end_index][1] > answer_end_idx:
                        token_end_index -= 1
                    token_end_index += 1

                    if self._verbose:
                        decoded_answer = self._tokenizer.decode(
                            input_ids[token_start_index:token_end_index]
                        )
                        print(f"  Decoded answer: {decoded_answer}")

                    item["start_positions"] = torch.tensor([token_start_index])
                    item["end_positions"] = torch.tensor([token_end_index])
                if self._verbose:
                    print("-" * 80)

        return encoded_qa_inputs

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


def test_data():
    dm = AmazonReviewQADataModule(file_path="data/sample.json", verbose=False)
    dm.setup()
    dm.prepare_data()

    # trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1)
    model = QuestionAnsweringModel()

    dataset: AmazonReviewQADataset = dm.get_data_set()

    # An annotated example can split into several features in a given data set,
    # so we map each example in the data set to its corresponding features.
    example_to_feature_indices_map = defaultdict(list)
    for i in range(len(dataset)):
        raw_item = dataset[i]
        example_id = raw_item["example_ids"]
        example_to_feature_indices_map[example_id].append(i)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    for i, batched_item in enumerate(dataloader):
        model_output: QuestionAnsweringModelOutput = model(batched_item)
        start_logits = model_output.start_logits
        end_logits = model_output.end_logits

        # The input to the model is:
        #   ```
        #   [CLS] question [SEP] context [SEP]
        #   ```
        # The context mask tells us which tokens are part of the context.
        context_mask = batched_item["context_mask"]

        # Since we want to mask out non-context tokens, we
        # invert the context mask.
        non_context_mask = torch.logical_not(context_mask)

        # Keep the [CLS] tokens as we use it to indicate that
        # the answer is not in the context.
        for j in range(non_context_mask.shape[0]):
            non_context_mask[j][0] = False

        # Mask the logits that are not part of the context because
        # we only want to consider the logits for the context.
        start_logits.masked_fill_(non_context_mask, -float("inf"))
        end_logits.masked_fill_(non_context_mask, -float("inf"))

        # Apply softmax to convert the logits to probabilities.
        start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
        end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]

        # Assuming the events “The answer starts at start_index” and
        # “The answer ends at end_index” to be independent, the probability
        # that the answer starts at start_index and ends at end_index is:
        # start_probabilities[start_index] × end_probabilities[end_index]
        # First, we need to compute all the possible products:
        scores = start_probabilities[:, None] * end_probabilities[None, :]

        # Then, we set the scores to 0 where start_index > end_index.
        # `triu()` returns the upper triangular part of a 2D tensor.
        scores = torch.triu(scores)

        # Next, we need to find the start_index and end_index that maximize
        # the probability of the answer. We use `torch.argmax()` to find the
        # indices of the maximum values in the scores tensor.
        max_index = scores.argmax().item()

        # PyTorch will return a single index in the flattened tensor.
        # We need to use the floor division // and modulus % operations
        # to get the start_index and end_index.
        pred_answer_start = max_index // scores.shape[1]
        pred_answer_end = max_index % scores.shape[1]
        pred_score = scores[pred_answer_start, pred_answer_end]

        # Get the indices of the tokens that are part of the context.
        given_answer_start = torch.squeeze(batched_item["start_positions"])
        given_answer_end = torch.squeeze(batched_item["end_positions"])

        print(f"given: [{given_answer_start:3d}, {given_answer_end:3d}] ", end="")
        print(f"pred: [{pred_answer_start:3d}, {pred_answer_end:3d}] ", end="")
        print(f"pred score: {pred_score:0.4f}")
        if i > 4:
            break

    # print("Making predictions...")
    # predictions = trainer.predict(model, dm)
    # print(predictions[0].loss)
    # print(predictions[0].start_logits)
    pass


if __name__ == "__main__":
    test_data()
