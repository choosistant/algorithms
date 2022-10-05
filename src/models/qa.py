import pytorch_lightning as pl
import torch
from datasets import load_metric
from transformers import AutoModelForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class QuestionAnsweringModel(pl.LightningModule):
    def __init__(
        self, qa_model_name: str = "deepset/roberta-base-squad2", lr: float = 1e-5
    ):
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        self.lr = lr
        self._squad_metric = load_metric("squad")

    def forward(self, x) -> QuestionAnsweringModelOutput:
        return self.model(input_ids=x["input_ids"], attention_mask=x["attention_mask"])

    def training_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = model_output.loss
        return {"train_loss": loss}

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
        model_output = self.forward(batch)
        return self.generate_batch_predictions(
            model_input=batch, model_output=model_output
        )

    def generate_batch_predictions(
        self,
        model_input: dict,
        model_output: QuestionAnsweringModelOutput,
    ) -> dict:
        predict_results = {
            "example_ids": [],
            "given_answer_start": [],
            "given_answer_end": [],
            "pred_answer_start": [],
            "pred_answer_end": [],
            "pred_score": [],
        }

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
            pred_answer_end = max_index % scores.shape[1]
            pred_score = scores[pred_answer_start, pred_answer_end]

            # Get the given labels for the item.
            given_answer_start = model_input["start_positions"][j].item()
            given_answer_end = model_input["end_positions"][j].item()

            # Append the results to the list.
            predict_results["example_ids"].append(model_input["example_ids"][j].item())
            predict_results["given_answer_start"].append(given_answer_start)
            predict_results["given_answer_end"].append(given_answer_end)
            predict_results["pred_answer_start"].append(pred_answer_start)
            predict_results["pred_answer_end"].append(pred_answer_end)
            predict_results["pred_score"].append(pred_score)

        return predict_results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


def test_qa_model():
    model = QuestionAnsweringModel()
    print(model)


if __name__ == "__main__":
    test_qa_model()
