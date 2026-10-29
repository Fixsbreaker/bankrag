"""Скачать зафиксированные PDF; хеш не позволяет незаметно подменить версию корпуса."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/sources.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/raw/official")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for item in json.loads(args.manifest.read_text(encoding="utf-8")):
        name = item["filename"]
        if Path(name).name != name:
            raise ValueError("filename должен быть именем файла без каталогов")
        target = args.out / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]:
            print(f"Проверен: {name}")
            continue
        with urlopen(Request(item["url"], headers={"User-Agent": "BankRAG research prototype"}), timeout=60) as response:
            data = response.read(30_000_001)
        if len(data) > 30_000_000 or not data.startswith(b"%PDF"):
            raise ValueError(f"{name}: ожидался PDF меньше 30 МБ")
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"{name}: документ изменился. Сначала перепроверьте разметку и обновите manifest.")
        target.write_bytes(data)
        print(f"Скачан: {name}")


if __name__ == "__main__":
    main()
