"""Память, задержка поиска и ANN Recall относительно Flat на одинаковых векторах."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss
import numpy as np
from bankrag.index import KINDS, build_index, bytes_per_vector


def git_state(root: Path) -> tuple[str | None, bool | None]:
    """Снимок реального состояния репозитория до записи результатов."""
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"],
                                         cwd=root, text=True, stderr=subprocess.DEVNULL)
        return revision, bool(status)
    except (OSError, subprocess.SubprocessError):
        return None, None


def relative_parameter(value):
    if isinstance(value, Path):
        return Path(os.path.relpath(value.resolve(), ROOT)).as_posix()
    return value


def recall_at_k(expected: np.ndarray, actual: np.ndarray) -> float:
    recalls = []
    for ground, found in zip(expected, actual):
        target = {int(i) for i in ground if i >= 0}
        if target:
            recalls.append(len(target.intersection(int(i) for i in found if i >= 0)) / len(target))
    return float(np.mean(recalls)) if recalls else 0.0


def load_vectors(path: Path) -> np.ndarray:
    vectors = np.ascontiguousarray(np.load(path, allow_pickle=False), dtype="float32")
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError(f"{path}: нужна непустая конечная матрица векторов")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
        raise ValueError(f"{path}: ожидаются нормализованные векторы")
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=ROOT / "index/vectors.npy")
    parser.add_argument("--queries", type=Path, default=ROOT / "index/query_vectors.npy")
    parser.add_argument("--out", type=Path, default=ROOT / "results/benchmark.csv")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--pq-m", type=int, default=16)
    parser.add_argument("--require-clean", action="store_true", help="запуск только из чистого Git-коммита")
    args = parser.parse_args()
    if min(args.k, args.threads, args.repeats, args.hnsw_m, args.ef_search, args.pq_m) < 1 or args.warmup < 0:
        parser.error("параметры должны быть положительными; warmup может быть 0")
    revision, local_changes = git_state(ROOT)
    if args.require_clean and (revision is None or local_changes is not False):
        parser.error("для --require-clean нужен чистый Git-коммит; сначала зафиксируйте изменения")
    faiss.omp_set_num_threads(args.threads)
    vectors, queries = load_vectors(args.vectors), load_vectors(args.queries)
    if vectors.shape[1] != queries.shape[1]:
        parser.error("размерности корпуса и запросов не совпадают")
    k = min(args.k, len(vectors))
    flat = build_index("flat", vectors)
    expected = flat.search(queries, k)[1]
    rows = []
    rng = np.random.default_rng(42)
    order = [int(i) for _ in range(args.repeats) for i in rng.permutation(len(queries))]
    for kind in KINDS:
        row = {"index": kind, "bytes_per_vector": bytes_per_vector(kind, vectors.shape[1], args.pq_m),
               "status": "ok", "error": "", "corpus_vectors": len(vectors), "queries": len(queries),
               "effective_k": k}
        if kind == "ivf_pq" and (len(vectors) < 256 or vectors.shape[1] % args.pq_m):
            row.update(status="skipped", error="IVF-PQ requires >=256 vectors and dimension divisible by pq_m")
            rows.append(row)
            continue
        started = time.perf_counter()
        index = build_index(kind, vectors, hnsw_m=args.hnsw_m, pq_m=args.pq_m)
        row["build_seconds"] = time.perf_counter() - started
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = args.ef_search
        serialized_bytes = int(faiss.serialize_index(index).nbytes)
        row.update(index_bytes=serialized_bytes, index_mb=serialized_bytes / 1_000_000,
                   total_bytes_per_vector=serialized_bytes / len(vectors))
        actual = index.search(queries, k)[1]
        row[f"recall_at_{args.k}"] = recall_at_k(expected, actual)
        for _ in range(args.warmup):
            index.search(queries, k)
        samples = []
        for qi in order:
            query = queries[qi:qi+1]
            start = time.perf_counter_ns()
            index.search(query, k)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        row.update(latency_ms=float(np.mean(samples)), p50_ms=float(np.median(samples)),
                   p95_ms=float(np.percentile(samples, 95)), timed_queries=len(samples))
        rows.append(row)
        print(f"{kind}: {serialized_bytes} bytes, p50={row['p50_ms']:.4f} ms, recall@{args.k}={row[f'recall_at_{args.k}']:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    cpu = platform.processor()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "system": platform.platform(),
            "cpu": cpu, "python": platform.python_version(), "numpy": np.__version__, "faiss": faiss.__version__,
            "source_revision": revision, "local_changes": local_changes,
            "git_state_captured": "before benchmark output is written",
            "paths_relative_to": "repository root",
            "parameters": {key: relative_parameter(value) for key, value in vars(args).items()},
            "vectors_shape": list(vectors.shape), "queries_shape": list(queries.shape),
            "vectors_sha256": hashlib.sha256(args.vectors.read_bytes()).hexdigest(),
            "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
            "notes": ["Search-only single-query latency; excludes embedding, LLM and disk IO.",
                      "index_bytes is serialized index size, not peak RAM or process RSS.",
                      "bytes_per_vector is code payload only; total_bytes_per_vector includes index overhead.",
                      "ANN recall compares IDs with Flat, not answer correctness; ties can affect recall.",
                      "Four configurations do not constitute equal-memory SQ/PQ comparison.",
                      "Small-corpus results are exploratory; PQ training can be under-sampled."]}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Таблица: {args.out}")


if __name__ == "__main__":
    main()
