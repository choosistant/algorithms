from typing import Dict

import evaluate as huggingface_evaluate
import pytorch_lightning as pl
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers.modeling_outputs import QuestionAnsweringModelOutput

LABEL_ID_MAP = {0: "benefit", 1: "drawback"}


class QuestionAnsweringPostProcessor:
    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self._predict_results = {
            "example_id": [],
            "input_ids": [],
            "given_answer_start": [],
            "given_answer_end": [],
            "given_answer": [],
            "pred_answer_start": [],
            "pred_answer_end": [],
            "pred_score": [],
            "pred_answer": [],
            "label": [],
        }

    def process_batch(
        self,
        model_input: dict,
        model_output: QuestionAnsweringModelOutput,
    ) -> None:
        for j in range(len(model_input["example_ids"])):
            start_logits = model_output.start_logits[[j], :]
            end_logits = model_output.end_logits[[j], :]

            # The input to the model is:
            #   ```
            #   [CLS] question [SEP] context [SEP]
            #   ```
            # The context mask tells us which tokens are part of the context.
            context_mask = model_input["context_mask"][[j], :]

            # Since we want to mask out non-context tokens, we
            # invert the context mask.
            non_context_mask = torch.logical_not(context_mask)

            # Keep the [CLS] tokens as we use it to indicate that
            # the answer is not in the context.
            non_context_mask[0][0] = False

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

            # If the answer is not in the context, the start_index will be 0.
            # In this case, we set the end_index to 0 as well.
            if pred_answer_start == 0:
                pred_answer_end = 0
            else:
                pred_answer_end = max_index % scores.shape[1]

            # Find the prediction score.
            pred_score = scores[pred_answer_start, pred_answer_end].item()

            # Get the given labels for the item.
            given_answer_start = model_input["start_positions"][j].item()
            given_answer_end = model_input["end_positions"][j].item()

            input_ids = model_input["input_ids"][j]

            predicted_text = self._tokenizer.decode(
                input_ids[pred_answer_start:pred_answer_end]
            )
            given_text = self._tokenizer.decode(
                input_ids[given_answer_start:given_answer_end]
            )

            label_id = model_input["label_ids"][j].item()
            label = LABEL_ID_MAP[label_id]

            # Append the results to the list.
            self._predict_results["example_id"].append(
                model_input["example_ids"][j].item()
            )
            self._predict_results["input_ids"].append(input_ids)
            self._predict_results["given_answer_start"].append(given_answer_start)
            self._predict_results["given_answer_end"].append(given_answer_end)
            self._predict_results["given_answer"].append(given_text)
            self._predict_results["pred_answer_start"].append(pred_answer_start)
            self._predict_results["pred_answer_end"].append(pred_answer_end)
            self._predict_results["pred_score"].append(pred_score)
            self._predict_results["pred_answer"].append(predicted_text)
            self._predict_results["label"].append(label)

    def get_results(self):
        return self._predict_results


class QuestionAnsweringModel(pl.LightningModule):
    def __init__(
        self, qa_model_name: str = "deepset/roberta-base-squad2", lr: float = 1e-5
    ):
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(qa_model_name, use_fast=True)
        self.lr = lr

    def forward(self, x) -> QuestionAnsweringModelOutput:
        return self.model(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            start_positions=x["start_positions"],
            end_positions=x["end_positions"],
        )

    def training_step(self, batch, batch_idx):
        model_output = self(batch)
        loss = model_output.loss
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = model_output.loss
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        pred_answer_start = torch.argmax(outputs.start_logits, dim=1)
        pred_answer_end = torch.argmax(outputs.end_logits, dim=1)
        if "start_positions" not in batch or "end_positions" not in batch:
            raise ValueError(
                "Examples in batch is not labelled. Cannot compute accuracy."
            )
        given_answer_start = torch.squeeze(batch["start_positions"])
        given_answer_end = torch.squeeze(batch["end_positions"])
        print(f"pred_start: {pred_answer_start}, pred_end: {pred_answer_end}")
        print(f"answer_start: {given_answer_start}, answer_end: {given_answer_end}")
        return outputs

    def predict_step(
        self,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict:
        post_processor = QuestionAnsweringPostProcessor(tokenizer=self.tokenizer)
        model_output = self.forward(batch)
        post_processor.process_batch(model_input=batch, model_output=model_output)
        return post_processor.get_results()

    def predict_all(self, data_loader) -> Dict[str, list]:
        post_processor = QuestionAnsweringPostProcessor(tokenizer=self.tokenizer)
        for batch in data_loader:
            model_output = self.forward(batch)
            post_processor.process_batch(model_input=batch, model_output=model_output)
        return post_processor.get_results()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def compute_metrics(self, predictions):
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

                predicted_text = self.tokenizer.decode(
                    input_ids[pred_answer_start:pred_answer_end]
                )
                given_text = self.tokenizer.decode(
                    input_ids[given_answer_start:given_answer_end]
                )

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

        return metric_output

    def push_to_hub(self, repo_id: str, use_auth_token: str):
        print(f"Pushing model to the {repo_id} using token {use_auth_token}...")
        model_push_result = self.model.push_to_hub(
            repo_id=repo_id,
            use_auth_token=use_auth_token,
            use_temp_dir=True,
        )
        print(f"Model pushed to the Hub: {model_push_result}")
        print("Pushing tokenizer to the Hub...")
        token_push_result = self.tokenizer.push_to_hub(
            repo_id=repo_id,
            use_auth_token=use_auth_token,
            use_temp_dir=True,
        )
        print(f"Tokenizer pushed to the Hub: {token_push_result}")
        return model_push_result, token_push_result
