import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, validator
from torch.utils.data import DataLoader

from src.data import QuestionAnsweringInputEncoder, QuestionAnsweringModelInput
from src.models.qa import QuestionAnsweringModel


@dataclass
class ApiServerConfig:
    prediction_log_path: str = "./data/logs/predictions/logs.jsonl"
    api_log_path: str = "./data/logs/api/logs.txt"
    model_name: str = "choosistant/qa-model"
    batch_size: int = 8


@dataclass
class LabeledSegment:
    segment: str
    label: str
    score: float

    def to_dict(self):
        return self.__dict__


class BenefitsAndDrawbacksExtractor:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self._model = QuestionAnsweringModel(qa_model_name=model_name)
        self._encoder = QuestionAnsweringInputEncoder(self._model.tokenizer)
        self._batch_size = batch_size

    def predict(self, document: str) -> List[LabeledSegment]:
        example_id = random.getrandbits(32)
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

        results = self._model.predict_all(data_loader=data_loader)
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
                print(f"Searched for {output_key}")
                if output_key in output_dict:
                    if output_dict[output_key].score >= score:
                        continue
                output_dict[output_key] = LabeledSegment(
                    segment=answer, label=label, score=score
                )
        return list(output_dict.values())


class PredictRequest(BaseModel):
    url: str
    review_text: str

    @validator("review_text")
    def review_text_must_not_be_empty(cls, val: str):
        if val is None or len(val.strip()) == 0:
            raise ValueError("review_text must not be empty")
        return val


class PredictResponse(BaseModel):
    id: str
    segments: List[str]
    labels: List[str]
    scores: List[float]


app = FastAPI()
cnf = ApiServerConfig()


@app.on_event("startup")
def startup_event():
    global classifier
    global prediction_log_file
    logger.add(cnf.api_log_path, rotation="1 day", retention="7 days")
    logger.info("Starting API server")

    classifier = BenefitsAndDrawbacksExtractor(
        model_name=cnf.model_name,
        batch_size=cnf.batch_size,
    )

    # Ensure parent directories exist.
    Path(cnf.prediction_log_path).parent.mkdir(parents=True, exist_ok=True)
    prediction_log_file = open(cnf.prediction_log_path, "a")

    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    prediction_log_file.close()
    logger.info("Shutting down application")


@app.get("/")
def read_root():
    return {"message": "Please use the /predict endpoint to make predictions"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    start_time = datetime.now()
    predictions = classifier.predict(request.review_text)
    end_time = datetime.now()
    inference_time_ms = (end_time - start_time).total_seconds() * 1000

    # Generate a unique ID for this prediction for tracking purposes.
    prediction_id = uuid.uuid4().hex

    # construct the data to be logged
    log_info = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "request": request.dict(),
        "predictions": [o.to_dict() for o in predictions],
        "inference_time_ms": int(inference_time_ms),
        "inference_device": torch.cuda.get_device_name(),
    }
    prediction_log_file.write(f"{json.dumps(log_info)}\n")
    prediction_log_file.flush()

    # construct response
    return PredictResponse(
        id=prediction_id,
        segments=[pred.segment for pred in predictions],
        labels=[pred.label for pred in predictions],
        scores=[pred.score for pred in predictions],
    )
