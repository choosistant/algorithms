# Algorithms

A repo for experimenting with different models.

## Development Setup

1. Install or active the required Python version ([pyenv](https://github.com/pyenv/pyenv) is highly recommended)

   ```bash
   curl https://pyenv.run | bash
   pyenv install
   ```

2. Install dependencies with [Poetry](https://python-poetry.org/docs/).

   ```bash
   poetry install
   poetry install --with torch-cpu # Install CPU only PyTorch.
   poetry install --with torch-gpu # Install PyTorch for GPU servers.
   ```

3. Install the pre-commit Git hooks:

   ```bash
   make precommit
   ```

## Getting Started

Run dummy model:

```bash
poetry run python -m src.test_dummy
```

Run API server for development purposes:

```bash
export INFERENCE_DEVICE=cuda:0 # Change to `cpu`
poetry run uvicorn src.api_server:app --reload
```

Run API server in production:

```bash
export INFERENCE_DEVICE=cuda:0 # Change to `cpu`
poetry run gunicorn -w 4 -k uvicorn.workers.UvicornWorker --timeout 300 --bind 0.0.0.0:8000 src.api_server:app
```
