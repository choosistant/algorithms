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
poetry run uvicorn src.api_server:app --reload
```
