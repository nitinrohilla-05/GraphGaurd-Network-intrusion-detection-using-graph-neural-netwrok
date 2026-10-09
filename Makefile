.PHONY: demo test lint typecheck p0 p1 p2 train containment up down ps clean help

PYTHON ?= python
PYTEST ?= pytest
RUFF ?= ruff
MYPY ?= mypy
DOCKER_COMPOSE ?= docker compose

# Virtual environment detection
ifeq ($(OS),Windows_NT)
	VENV_SCRIPTS := .venv/Scripts
	ifneq ($(wildcard $(VENV_SCRIPTS)/python.exe),)
		PYTHON := $(VENV_SCRIPTS)/python.exe
		PYTEST := $(VENV_SCRIPTS)/pytest.exe
		RUFF := $(VENV_SCRIPTS)/ruff.exe
		MYPY := $(VENV_SCRIPTS)/mypy.exe
	endif
else
	VENV_BIN := .venv/bin
	ifneq ($(wildcard $(VENV_BIN)/python),)
		PYTHON := $(VENV_BIN)/python
		PYTEST := $(VENV_BIN)/pytest
		RUFF := $(VENV_BIN)/ruff
		MYPY := $(VENV_BIN)/mypy
	endif
endif

help:
	@echo GraphGuard Makefile Targets:
	@echo   demo       Run end-to-end detection and containment demo
	@echo   test       Run unit tests with pytest
	@echo   lint       Run linter with ruff
	@echo   typecheck  Run strict type checking with mypy
	@echo   p0         Run Phase P0 pipeline (lint + typecheck + test)
	@echo   train      Run Phase P1 training pipeline
	@echo   p1         Run Phase P1 pipeline (lint + typecheck + test + train)
	@echo   up         Start Docker Compose services (--env-file .env)
	@echo   down       Stop Docker Compose services
	@echo   ps         Check Docker Compose status
	@echo   clean      Remove temporary cache artifacts

demo:
	$(PYTHON) demo.py

test:
	$(PYTEST) tests/unit -v --tb=short

lint:
	$(RUFF) check graphguard tests

typecheck:
	$(MYPY) --strict graphguard/core

p0: lint typecheck test
	@echo Phase P0 verification complete: all checks passed.

train:
	$(PYTHON) -m graphguard.detect.train

p1: lint typecheck test train
	@echo Phase P1 verification complete: model trained and registered.

containment:
	$(PYTHON) -m graphguard.eval.containment_eval

p2: lint typecheck test containment
	@echo Phase P2 verification complete: containment benchmarks and frontier plot generated.

up:
	$(DOCKER_COMPOSE) -f deploy/docker-compose.yml --env-file .env up -d

down:
	$(DOCKER_COMPOSE) -f deploy/docker-compose.yml --env-file .env down

ps:
	$(DOCKER_COMPOSE) -f deploy/docker-compose.yml --env-file .env ps

clean:
	$(PYTHON) -c "import shutil, os; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.mypy_cache', '.ruff_cache', '__pycache__']]"
