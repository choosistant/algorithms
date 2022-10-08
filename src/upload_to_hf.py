import click

from src.models.qa import QuestionAnsweringModel


class ModelPusher:
    def __init__(self, local_model_id: str, repo_id: str, auth_token: str):
        self._repo_id = repo_id
        self._auth_token = auth_token
        self._model = QuestionAnsweringModel(qa_model_name=local_model_id)

    def run(self):
        print("Pushing model to the Hub...")
        result = self._model.push_to_hub(
            repo_id=self._repo_id,
            use_auth_token=self._auth_token,
        )
        print(f"Model pushed to the Hub: {result}")


@click.command(help="Push a model to HuggingFace.")
@click.option(
    "-l",
    "--local-model-id",
    type=click.STRING,
    required=False,
    default="deepset/roberta-base-squad2",
)
@click.option(
    "-r",
    "--repo-id",
    type=click.STRING,
    required=False,
    default="choosistant/qa-model",
)
@click.option(
    "-t",
    "--auth-token",
    type=click.STRING,
    required=False,
    default=None,
)
def main(**kwargs):
    ModelPusher(**kwargs).run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
