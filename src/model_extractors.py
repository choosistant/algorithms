import random
from dataclasses import dataclass
from typing import Dict, List, Protocol

import torch
from simpletransformers.seq2seq import Seq2SeqArgs, Seq2SeqModel
from torch.utils.data import DataLoader
from transformers import AutoModel, pipeline

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


class Predictor(Protocol):
    """An interface for a model that can predict labels for segments of text."""

    def predict(self, document: str) -> List[LabeledSegment]:
        ...

    @property
    def inference_device(self) -> str:
        ...


class QuestionAnsweringPredictor:
    def __init__(
        self, qa_model_name: str, batch_size: int, inference_device: str
    ) -> None:
        self._model = QuestionAnsweringModel(qa_model_name=qa_model_name)
        self._encoder = QuestionAnsweringInputEncoder(self._model.tokenizer)
        self._batch_size = batch_size
        self._inference_device = inference_device

    @property
    def inference_device(self) -> str:
        return self._inference_device

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
                inference_device=torch.device(self._inference_device),
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
        inference_device: str,
        encoder_decoder_type: str = "bart",
        encoder_decoder_name: str = "facebook/bart-large",
    ) -> None:
        use_cuda = inference_device.split(":")[0] == "cuda"
        device_id = int(inference_device.split(":")[1]) if use_cuda else 0

        self._backbone_model = AutoModel.from_pretrained(model_name_seq2seq)
        model_args = Seq2SeqArgs()
        self._model = Seq2SeqModel(
            encoder_decoder_type=encoder_decoder_type,
            encoder_decoder_name=encoder_decoder_name,
            args=model_args,
            model=self._backbone_model,
            use_cuda=use_cuda,
            cuda_device=device_id,
        )
        self._inference_device = inference_device

    @property
    def inference_device(self) -> str:
        return self._inference_device

    def predict(self, document: str) -> List[LabeledSegment]:
        with time_block("Took {:0.2f} seconds make seq2seq predictions"):
            results = self._model.predict([document])
            print(f"Output from seq2seq model: {results}")
        return [
            LabeledSegment(segment=pred, label="benefit", score=0.0) for pred in results
        ]


class QuestionAnsweringPredictorS:
    def __init__(self, qa_model_name: str) -> None:
        self._nlp = pipeline(
            "question-answering", model=qa_model_name, tokenizer=qa_model_name
        )

    def predict(self, document: str) -> List[LabeledSegment]:
        benefits = []
        drawback = []
        text = document
        while len(document.split(" ")) > 2:
            QA_input = {"question": "where are the benefits?", "context": document}
            res = self._nlp(QA_input)
            benefits.append(document[res.get("start") : res.get("end")])
            document = document[res.get("end") :]

        while len(text.split(" ")) > 2:
            QA_input = {"question": "where are the drawbacks?", "context": text}
            res = self._nlp(QA_input)
            drawback.append(text[res.get("start") : res.get("end")])
            text = text[res.get("end") :]

        output_benefit = [
            LabeledSegment(segment=pred, label="benefit", score=0.0)
            for pred in benefits[0:-1]
        ]
        output_drawback = [
            LabeledSegment(segment=pred, label="drawback", score=0.0)
            for pred in drawback[0:-1]
        ]
        return output_drawback + output_benefit
