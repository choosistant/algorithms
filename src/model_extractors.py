import random
from dataclasses import dataclass
from typing import Dict, List

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


class BenefitsAndDrawbacksExtractor:
    def __init__(self, model_qa: str, batch_size: int, model_seq2seq: str) -> None:
        self._qa_model = QuestionAnsweringModel(qa_model_name=model_qa)
        self._seq2seq_model = AutoModel.from_pretrained(model_seq2seq)
        self._encoder = QuestionAnsweringInputEncoder(self._qa_model.tokenizer)
        self._batch_size = batch_size

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
                inference_device=torch.device("cuda:0"),
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

    def _init_seq2seq(self):
        model_args = Seq2SeqArgs()
        model = Seq2SeqModel(
            encoder_decoder_type="bart",
            encoder_decoder_name="facebook/bart-large",
            args=model_args,
            model=self._seq2seq_model,
            use_cuda=False,
        )
        return model

    def predict_seq2seq(self, document: str) -> List[LabeledSegment]:
        example_id = random.getrandbits(32)
        model = self._init_seq2seq(self)

        with time_block("Took {:0.2f} seconds make seq2seq predictions"):
            results = model.predict([document])
        output_dict = LabeledSegment(
            id=example_id, segment=results, label=["benefits"], score=0.0
        )
        return list(output_dict.values())
