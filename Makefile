.PHONY: setup data build eval app test

VENV := .venv
PY := $(VENV)/bin/python

# Нужен Python 3.11. Если его нет в системе, окружение создаётся через uv.
setup:
	@if command -v python3.11 >/dev/null 2>&1; then \
		python3.11 -m venv $(VENV) && $(PY) -m pip install -q -r requirements.txt; \
	elif command -v uv >/dev/null 2>&1; then \
		uv venv -q --python 3.11 $(VENV) && uv pip install -q --python $(PY) -r requirements.txt; \
	else \
		echo "Нужен Python 3.11 или uv"; exit 1; \
	fi

data:
	scripts/download_data.sh

build:
	scripts/build_all.sh

eval:
	PYTHONPATH=src $(PY) -W ignore -m bankrag.evaluate > /dev/null
	@echo "Отчёт: reports/metrics.md"

app:
	$(VENV)/bin/streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q
