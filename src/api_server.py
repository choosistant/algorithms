import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Protocol

import torch
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, validator

from src.model_extractors import Predictor, QuestionAnsweringPredictor, Seq2SeqPredictor

PREDICTORS: Dict[str, Predictor] = {
    "qa": classifier_qa,
    "seq2seq": classifier_seq2seq,
}


@dataclass
class ApiServerConfig:
    prediction_log_path: str = "./data/logs/predictions/logs.jsonl"
    api_log_path: str = "./data/logs/api/logs.txt"
    model_name_qa: str = "choosistant/qa-model"
    model_name_seq2seq: str = "choosistant/seq2seqmodel"
    batch_size: int = 8


class PredictRequest(BaseModel):
    url: str
    review_text: str
    model_type: str

    @validator("review_text")
    def review_text_must_not_be_empty(cls, val: str):
        if val is None or len(val.strip()) == 0:
            raise ValueError("review_text must not be empty")
        return val

    @validator("model_type")
    def model_type_(cls, val: str):
        if val not in PREDICTORS:
            raise ValueError(f"model_type must be one of {PREDICTORS.keys()}")
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
    global classifier_qa
    global classifier_seq2seq
    global prediction_log_file

    logger.add(cnf.api_log_path, rotation="1 day", retention="7 days")
    logger.info("Starting API server")

    classifier_qa = QuestionAnsweringPredictor(
        qa_model_name=cnf.model_name_qa, batch_size=cnf.batch_size, cuda_device_no=1
    )
    classifier_seq2seq = Seq2SeqPredictor(
        model_name_seq2seq=cnf.model_name_seq2seq, cuda_device_no=1
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
    predictor = PREDICTORS[request.model_type]
    predictions = predictor.predict(request.review_text)
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
