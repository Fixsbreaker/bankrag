.PHONY: setup data lists build eval app test chat-eval check-llm

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
	$(PY) scripts/download_lists.py

# Перечни АФМ РК (открытый API api.websfm.kz). Можно обновлять каждый день, затем PYTHONPATH=src $(PY) -m bankrag.owners.
lists:
	$(PY) scripts/download_lists.py
	PYTHONPATH=src $(PY) -W ignore -m bankrag.owners

build:
	scripts/build_all.sh

eval:
	PYTHONPATH=src $(PY) -W ignore -m bankrag.evaluate > /dev/null
	@echo "Отчёт: reports/metrics.md"

# Приложение работает и без Ollama: тогда включается режим кнопок.
app:
	@$(PY) scripts/check_llm.py >/dev/null 2>&1 && echo "Ollama и модель готовы" || \
		echo "Ollama или модель недоступны: чат будет в режиме кнопок. Проверка: make check-llm"
	$(VENV)/bin/streamlit run app/streamlit_app.py

check-llm:
	$(PY) scripts/check_llm.py

# Оценка чата на 30 вопросах для двух моделей. Нужна запущенная Ollama.
chat-eval:
	$(PY) -W ignore scripts/eval_chat.py --model qwen3:14b
	$(PY) -W ignore scripts/eval_chat.py --model qwen3:8b

test:
	$(PY) -m pytest -q
