import random
from dataclasses import dataclass
from typing import Dict, List, Protocol

import torch
from simpletransformers.seq2seq import Seq2SeqArgs, Seq2SeqModel
from torch.utils.data import DataLoader
from transformers import AutoModel

from src.data import QuestionAnsweringInputEncoder, QuestionAnsweringModelInput
from src.models.qa import QuestionAnsweringModel
from src.stop_watch import time_block


@dataclass
class LabeledSegment:
    segment: str
    label: str
    score: float

    def to_dict(self):
        return self.__dict__


class QuestionAnsweringPredictor:
    def __init__(
        self, qa_model_name: str, batch_size: int, cuda_device_no: int
    ) -> None:
        self._model = QuestionAnsweringModel(qa_model_name=qa_model_name)
        self._encoder = QuestionAnsweringInputEncoder(self._model.tokenizer)
        self._batch_size = batch_size
        self._cuda_device_no = cuda_device_no

    def predict(self, document: str) -> List[LabeledSegment]:
        example_id = random.getrandbits(32)

        with time_block("Took {:0.2f} seconds to encode"):
            encoded_inputs: List[QuestionAnsweringModelInput] = self._encoder.encode(
                input_text=document, example_id=example_id
            )
        data_loader = DataLoader(
            dataset=[item.to_dict() for item in encoded_inputs],
            batch_size=self._batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
        )

        with time_block("Took {:0.2f} seconds make predictions"):
            results = self._model.predict_all(
                data_loader=data_loader,
                inference_device=torch.device(f"cuda:{self._cuda_device_no}"),
            )

        with time_block("Took {:0.2f} seconds to convert labels"):
            labeled_segments = self._convert_labeled_segments(results=results)
        return labeled_segments

    def _convert_labeled_segments(
        self, results: Dict[str, list]
    ) -> List[LabeledSegment]:
        output_dict = {}
        for i in range(len(results["pred_answer"])):
            answer = results["pred_answer"][i].strip()
            label = results["label"][i]
            score = results["pred_score"][i]
            if len(answer) > 0:
                output_key = f"{label}:{answer}"
                if output_key in output_dict:
                    if output_dict[output_key].score >= score:
                        continue
                output_dict[output_key] = LabeledSegment(
                    segment=answer, label=label, score=score
                )
        return list(output_dict.values())


class Seq2SeqPredictor:
    """A seq2seq model that predicts labels for segments of text."""

    def __init__(
        self,
        model_name_seq2seq: str,
        cuda_device_no: int,
        encoder_decoder_type: str = "bart",
        encoder_decoder_name: str = "facebook/bart-large",
    ) -> None:
        self._backbone_model = AutoModel.from_pretrained(model_name_seq2seq)
        model_args = Seq2SeqArgs()
        self._model = Seq2SeqModel(
            encoder_decoder_type=encoder_decoder_type,
            encoder_decoder_name=encoder_decoder_name,
            args=model_args,
            model=self._backbone_model,
            use_cuda=True,
            cuda_device=cuda_device_no,
        )

    def predict(self, document: str) -> List[LabeledSegment]:
        with time_block("Took {:0.2f} seconds make seq2seq predictions"):
            results = self._model.predict([document])
            print(f"Output from seq2seq model: {results}")
        return [
            LabeledSegment(segment=pred, label="benefit", score=0.0) for pred in results
        ]


class Predictor(Protocol):
    """An interface for a model that can predict labels for segments of text."""

    def predict(self, document: str) -> List[LabeledSegment]:
        ...
