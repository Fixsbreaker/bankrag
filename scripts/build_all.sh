#!/usr/bin/env bash
# Полный конвейер: данные → признаки → модели → индексы → метрики.
# Каждый шаг идёт отдельным процессом: LightGBM нельзя совмещать с FAISS и torch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
PYTHON="$ROOT/.venv/bin/python"

run() {
  echo "==> $*"
  "$PYTHON" -W ignore -m "$@" > /dev/null
}

run bankrag.data_loader
run bankrag.labels
run bankrag.features
run bankrag.baseline_rule
run bankrag.model
run bankrag.similarity
run bankrag.kb_index
run bankrag.kz_scenarios
run bankrag.model --kz
run bankrag.kz_scenarios --seeds 10
run bankrag.model_audit
echo "Готово. Метрики считает следующий шаг: make eval"
