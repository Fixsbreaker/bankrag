"""Скачивает перечни АФМ РК (лица, связанные с терроризмом, экстремизмом и ФРОМУ) в data/raw/lists/.

Источник — открытый REST API Агентства по финансовому мониторингу: api.websfm.kz.
Ключ и регистрация не нужны. После загрузки сверка работает без интернета.

    .venv/bin/python scripts/download_lists.py
    .venv/bin/python scripts/download_lists.py --sample    # по 50 записей из каждого перечня
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bankrag.watchlist import LIST_TITLES, LISTS_DIR  # noqa: E402

API = "https://api.websfm.kz/v1/sanctions/"
# Перечень → адрес API и параметры. Для ФРОМУ и СБ ООН сервис отдаёт только действующие записи.
SOURCES = {
    "sanction_category_1": ("sanction-terrorist/", {"status": "acting", "category": "sanction_category_1"}),
    "sanction_category_2": ("sanction-terrorist/", {"status": "acting", "category": "sanction_category_2"}),
    "sanction_category_3": ("sanction-weapon/", {"category": "sanction_category_3"}),
    "sanction_category_4": ("sanction-weapon/", {"category": "sanction_category_4"}),
}
PAGE = 500
USER_AGENT = "BankRAG student prototype (open data download)"


def fetch(url: str) -> dict:
    """GET с несколькими попытками: сервис иногда отвечает медленно."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("недостижимо")


def download(category: str, limit: int | None) -> list[dict]:
    """Все записи одного перечня, страница за страницей."""
    path, params = SOURCES[category]
    records: list[dict] = []
    offset = 0
    while True:
        page = PAGE if limit is None else min(PAGE, limit - len(records))
        url = API + path + "?" + urllib.parse.urlencode({**params, "limit": page, "offset": offset})
        data = fetch(url)
        records += data["results"]
        offset += len(data["results"])
        if not data.get("next") or not data["results"] or (limit is not None and len(records) >= limit):
            return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", action="store_true", help="по 50 записей из каждого перечня")
    args = parser.parse_args()
    out_dir = ROOT / LISTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"source": API, "downloaded_at": datetime.now().isoformat(timespec="seconds"), "sample": args.sample,
            "lists": {}}
    for category in SOURCES:
        records = download(category, 50 if args.sample else None)
        (out_dir / f"{category}.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
        meta["lists"][category] = {"title": LIST_TITLES[category], "records": len(records)}
        print(f"{LIST_TITLES[category]}: {len(records)}")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено в {out_dir}")


if __name__ == "__main__":
    main()
