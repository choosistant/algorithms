.DEFAULT_GOAL := all

install_packages:
	poetry install

sort_imports:
	poetry run isort .

format:
	poetry run black .

lint:
	poetry run flake8 .

test:
	poetry run pytest tests/

precommit:
	poetry run pre-commit install && poetry run pre-commit run --all-files

all: install_packages precommit test
