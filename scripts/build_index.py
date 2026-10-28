"""Строит индекс по документам из data/raw/ и сохраняет в index/.

    python scripts/build_index.py --kind hnsw
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402

from bankrag.embed import DEFAULT_MODEL, embed  # noqa: E402
from bankrag.index import KINDS, build_index  # noqa: E402
from bankrag.ingest import load_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=ROOT / "data" / "raw", type=Path)
    parser.add_argument("--out", default=ROOT / "index", type=Path)
    parser.add_argument("--kind", default="hnsw", choices=KINDS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    chunks = load_corpus(args.raw)
    if not chunks:
        sys.exit(f"нет документов в {args.raw}")
    print(f"документов: {len({c['source'] for c in chunks})}, фрагментов: {len(chunks)}")

    t = time.perf_counter()
    vectors = embed([c["text"] for c in chunks], "passage", args.model)
    print(f"эмбеддинги {vectors.shape} за {time.perf_counter() - t:.1f} с")

    index = build_index(args.kind, vectors)
    args.out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out / "faiss.index"))
    (args.out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "meta.json").write_text(json.dumps({"model": args.model, "kind": args.kind, "dim": vectors.shape[1]}), encoding="utf-8")
    print(f"индекс {args.kind} сохранён в {args.out}")


if __name__ == "__main__":
    main()
