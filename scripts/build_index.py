"""Строит индекс по проверенным документам из data/raw/official/.

    python scripts/build_index.py --kind hnsw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from bankrag.embed import DEFAULT_MODEL, embed  # noqa: E402
from bankrag.index import KINDS, build_index  # noqa: E402
from bankrag.ingest import SUPPORTED, load_corpus  # noqa: E402


def verify_sources(raw_dir: Path, manifest_path: Path) -> dict[str, dict]:
    """Корпус должен точно соответствовать манифесту до разбора и эмбеддинга."""
    if not raw_dir.is_dir():
        raise ValueError(f"нет каталога {raw_dir}; сначала запустите scripts/download_docs.py")
    if not manifest_path.is_file():
        raise ValueError(f"манифест не найден: {manifest_path}")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {}
    for source in entries:
        name = source["filename"]
        if not name or Path(name).name != name:
            raise ValueError(f"filename в манифесте должен быть именем файла без каталогов: {name!r}")
        if name in sources:
            raise ValueError(f"повтор filename в манифесте: {name}")
        for field in ("url", "sha256", "downloaded_on"):
            if not source.get(field):
                raise ValueError(f"{name}: отсутствует поле {field} в манифесте")
        sources[name] = source
    files = {str(path.relative_to(raw_dir)): path for path in raw_dir.rglob("*")
             if path.is_file() and path.suffix.lower() in SUPPORTED}
    extra, missing = set(files) - set(sources), set(sources) - set(files)
    if extra:
        raise ValueError("документы отсутствуют в манифесте: " + ", ".join(sorted(extra)))
    if missing:
        raise ValueError("не скачаны документы из манифеста: " + ", ".join(sorted(missing)))
    if not files:
        raise ValueError("корпус пуст")
    for name, path in files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != sources[name]["sha256"]:
            raise ValueError(f"документ {name} изменился: обновите манифест и перепроверьте разметку")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=ROOT / "data" / "raw" / "official", type=Path)
    parser.add_argument("--out", default=ROOT / "index", type=Path)
    parser.add_argument("--kind", default="hnsw", choices=KINDS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sources", type=Path, default=ROOT / "data/sources.json")
    args = parser.parse_args()

    try:
        sources = verify_sources(args.raw, args.sources)
    except (ValueError, KeyError) as error:
        sys.exit(str(error))
    chunks = load_corpus(args.raw)
    if not chunks:
        sys.exit(f"нет документов в {args.raw}")
    for chunk in chunks:
        source = sources[chunk["source"]]
        chunk.update(source_url=source["url"], source_sha256=source["sha256"],
                     downloaded_on=source["downloaded_on"])
    print(f"документов: {len({c['source'] for c in chunks})}, фрагментов: {len(chunks)}")

    t = time.perf_counter()
    vectors = embed([c["text"] for c in chunks], "passage", args.model)
    print(f"эмбеддинги {vectors.shape} за {time.perf_counter() - t:.1f} с")

    index = build_index(args.kind, vectors)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "vectors.npy", vectors)
    faiss.write_index(index, str(args.out / "faiss.index"))
    (args.out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "meta.json").write_text(json.dumps({"model": args.model, "kind": args.kind, "dim": vectors.shape[1]}), encoding="utf-8")
    print(f"индекс {args.kind} сохранён в {args.out}")


if __name__ == "__main__":
    main()
