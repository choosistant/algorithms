import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import nltk
import numpy as np
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer


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
        doc_stride: int = 128,
    ) -> None:
        super().__init__()
        self._file_path = Path(file_path)
        if not self._file_path.exists():
            raise ValueError(f"Data file {self._file_path} does not exist.")

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

        self._benefit_question = "What are the benefits of this item, if any?"
        self._drawback_question = "What are the drawbacks of this item, if any?"

    def setup(self, stage=None):
        nltk.download("punkt")
        pass

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
        self, context: str, question: str, answer_ranges: List[Tuple[int, int]]
    ) -> List[dict]:
        encoded_qa_inputs = []

        encoded_question_and_context = self._tokenizer.encode_plus(
            question,
            context,
            truncation="only_second",
            max_length=self._doc_max_length,
            stride=self._doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        offset_mapping = encoded_question_and_context.pop("offset_mapping")

        for i, offsets in enumerate(offset_mapping):
            seq_ids = np.array(encoded_question_and_context.sequence_ids(i))
            context_token_indices = np.where(seq_ids == 1)[0]
            context_start_idx = context_token_indices[0]
            context_end_idx = context_token_indices[-1]

            input_ids = encoded_question_and_context["input_ids"][i]

            print(f"  Decoded input: {self._tokenizer.decode(input_ids)}")

            bos_index = input_ids.index(self._tokenizer.cls_token_id)

            context_boundary_start = offsets[context_start_idx][0]
            context_boundary_end = offsets[context_end_idx][1]

            for answer_start_idx, answer_end_idx in answer_ranges:
                item = encoded_question_and_context.copy()
                encoded_qa_inputs.append(item)
                item["start_positions"] = []
                item["end_positions"] = []

                if (
                    context_boundary_start > answer_start_idx
                    or context_boundary_end < answer_end_idx
                ):
                    print("   No benefit answer in this chunk!")

                    item["start_positions"].append(bos_index)
                    item["end_positions"].append(bos_index)
                else:
                    print(
                        f"  Original answer: {context[answer_start_idx:answer_end_idx]}"
                    )

                    token_start_index = context_start_idx
                    while offsets[token_start_index][0] < answer_start_idx:
                        token_start_index += 1

                    token_end_index = context_end_idx
                    while offsets[token_end_index][1] > answer_end_idx:
                        token_end_index -= 1
                    token_end_index += 1

                    labeled_answer = self._tokenizer.decode(
                        input_ids[token_start_index:token_end_index]
                    )
                    print(f"  Decoded answer: {labeled_answer}")

                    item["start_positions"].append(token_start_index)
                    item["end_positions"].append(token_end_index)
                print("-" * 80)
        return encoded_qa_inputs

    def prepare_data(self) -> None:
        examples = self._parse_annoted_examples()
        print(f"Found {len(examples)} examples.")

        for example in examples[30:32]:
            chunks = self._split_text_into_chunks_by_sentence_pairings(
                review_text=example.review_text
            )

            for chunk_start_idx, chunk_end_idx in chunks:
                chunk = example.review_text[chunk_start_idx:chunk_end_idx]
                benefit_segments = self._extract_answers(
                    chunk_start_idx, chunk_end_idx, example.labels, "benefit"
                )

                encoded_inputs = self._encode_qa_input(
                    context=chunk,
                    question=self._benefit_question,
                    answer_ranges=benefit_segments,
                )

                print(f"Encoded {len(encoded_inputs)} chunks.")

    def prepare_data2(self) -> None:
        examples = self._parse_annoted_examples()
        print(f"Found {len(examples)} examples.")

        output = []

        for example in examples[30:]:
            chunks = self._split_text_into_chunks_by_sentence_pairings(
                review_text=example.review_text
            )

            for chunk_start_idx, chunk_end_idx in chunks:

                chunk = example.review_text[chunk_start_idx:chunk_end_idx]
                benefit_segments = self._extract_answers(
                    chunk, chunk_start_idx, chunk_end_idx, example.labels, "benefit"
                )

                encoded_question_and_context = self._tokenizer.encode_plus(
                    self._benefit_question,
                    chunk,
                    truncation="only_second",
                    max_length=self._doc_max_length,
                    stride=self._doc_stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                )
                offset_mapping = encoded_question_and_context.pop("offset_mapping")

                print("\n")
                print("=" * 80)
                print(f"Chunk  : '{chunk}'")
                print("=" * 80)

                for i, offsets in enumerate(offset_mapping):
                    seq_ids = np.array(encoded_question_and_context.sequence_ids(i))
                    context_token_indices = np.where(seq_ids == 1)[0]
                    context_start_idx = context_token_indices[0]
                    context_end_idx = context_token_indices[-1]

                    input_ids = encoded_question_and_context["input_ids"][i]

                    print(f"  Decoded input: {self._tokenizer.decode(input_ids)}")

                    # print(f"Input ids: {encoded_question_and_context['input_ids']}")
                    bos_index = input_ids.index(self._tokenizer.cls_token_id)

                    context_boundary_start = offsets[context_start_idx][0]
                    context_boundary_end = offsets[context_end_idx][1]

                    for answer_start_idx, answer_end_idx in benefit_segments:
                        item = encoded_question_and_context.copy()
                        output.append(item)
                        item["start_positions"] = []
                        item["end_positions"] = []

                        if (
                            context_boundary_start > answer_start_idx
                            or context_boundary_end < answer_end_idx
                        ):
                            print("   No benefit answer in this chunk!")

                            item["start_positions"].append(bos_index)
                            item["end_positions"].append(bos_index)
                        else:
                            print(
                                f"  Benefit: {chunk[answer_start_idx:answer_end_idx]}"
                            )

                            token_start_index = context_start_idx
                            while offsets[token_start_index][0] < answer_start_idx:
                                token_start_index += 1

                            token_end_index = context_end_idx
                            while offsets[token_end_index][1] > answer_end_idx:
                                token_end_index -= 1
                            token_end_index += 1

                            labeled_answer = self._tokenizer.decode(
                                input_ids[token_start_index:token_end_index]
                            )
                            print(f"  Decoded benefit: {labeled_answer}")

                            item["start_positions"].append(token_start_index)
                            item["end_positions"].append(token_end_index)
                        print("-" * 80)

            break

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
        return example


def test_data():
    dm = AmazonReviewQADataModule(
        file_path="data/project-1-at-2022-09-25-08-41-12f06b11.json"
    )
    dm.setup()
    dm.prepare_data()

    question = "What is the best thing about this product?"
    context = "I love the color and the fit of this product. It is very comfortable."

    inputs = dm._tokenizer(
        question,
        context,
        truncation=True,
        padding=True,  # return_tensors="pt",
    )
    decoded = dm._tokenizer.decode(inputs["input_ids"])
    print(dm._tokenizer.model_max_length)
    # print(inputs)
    print(f"Decoded: {decoded}")

    # qa_pipeline = pipeline(
    #     task="question-answering", model="deepset/roberta-base-squad2"
    # )

    # print(qa_pipeline.tokenizer)

    # inputs = qa_pipeline.tokenizer.encode(
    #     question, context
    # )
    # print(inputs)

    pass


if __name__ == "__main__":
    test_data()
