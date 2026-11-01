#!/usr/bin/env bash
# Скачивает набор IBM AML (HI-Small) с Kaggle в data/raw/.
# Нужен бесплатный токен Kaggle: ~/.kaggle/access_token или kaggle.json.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"
DATASET="ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
RAW="$ROOT/data/raw"

if [ ! -s "$HOME/.kaggle/access_token" ] && [ ! -s "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "Нет токена Kaggle. Создайте его на kaggle.com: Settings → API → Create New Token," >&2
  echo "и сохраните в ~/.kaggle/access_token (chmod 600)." >&2
  exit 1
fi

mkdir -p "$RAW"
for name in HI-Small_Trans.csv HI-Small_Patterns.txt HI-Small_accounts.csv; do
  if [ -s "$RAW/$name" ]; then
    echo "Уже есть: $name"
    continue
  fi
  "$KAGGLE" datasets download "$DATASET" --file "$name" --path "$RAW" --unzip --quiet
  # CLI может оставить архив, если не распаковал файл сам.
  if [ -f "$RAW/$name.zip" ]; then
    (cd "$RAW" && unzip -oq "$name.zip" && rm "$name.zip")
  fi
  [ -s "$RAW/$name" ] || { echo "Не удалось скачать $name" >&2; exit 1; }
  echo "Скачан: $name"
done
