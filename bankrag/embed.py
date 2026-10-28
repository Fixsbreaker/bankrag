"""Эмбеддинги многоязычной моделью (русский, казахский, английский)."""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-small"

_models: dict = {}


def _model(name: str):
    if name not in _models:
        from sentence_transformers import SentenceTransformer

        _models[name] = SentenceTransformer(name)
    return _models[name]


def embed(texts: list[str], kind: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """kind = "query" или "passage": e5 ожидает этот префикс перед текстом."""
    prefixed = [f"{kind}: {t}" for t in texts]
    vectors = _model(model_name).encode(prefixed, normalize_embeddings=True, batch_size=32)
    return np.asarray(vectors, dtype="float32")
