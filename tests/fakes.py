"""Легкие заменители моделей для тестов: работают без скачивания bge / BM25 / реранкера.

Они лексические (хеширование слов), поэтому качество поиска с ними НЕ показательно —
тесты проверяют логику конвейера (фильтры, слияние, реранкинг, отбор контекста), а не метрики.
"""

import math
import re
import zlib
from dataclasses import dataclass
from typing import Iterable, List

from qdrant_client import QdrantClient, models

from src.retrieval.qdrant_store import DENSE_DIM, DENSE_NAME, SPARSE_NAME, create_collection, point_id

_STOP = {"the", "and", "for", "with", "that", "this", "was", "were", "are", "from", "what", "why", "how",
         "which", "did", "does", "when", "who", "into", "not", "but", "have", "has", "had", "its", "their"}


def tokens(text: str) -> List[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) < 3 or w in _STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def _h(token: str) -> int:
    return zlib.crc32(token.encode())


class FakeDense:
    """Хешированный мешок слов (+ биграммы) размерности 768, L2-нормированный."""

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * DENSE_DIM
        toks = tokens(text)
        for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            h = _h(t)
            v[h % DENSE_DIM] += 1.0 if (h >> 16) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]


@dataclass
class _Sparse:
    indices: List[int]
    values: List[float]


class FakeSparse:
    """BM25-подобный sparse-вектор: индекс = хеш слова, вес = 1 + log(tf); IDF добавляет Qdrant."""

    def _vec(self, text: str) -> _Sparse:
        tf = {}
        for t in tokens(text):
            idx = _h(t) % (1 << 24)
            tf[idx] = tf.get(idx, 0) + 1
        return _Sparse(indices=list(tf), values=[1.0 + math.log(c) for c in tf.values()])

    def embed_query(self, text: str) -> _Sparse:
        return self._vec(text)

    def embed_documents(self, texts: Iterable[str]) -> List[_Sparse]:
        return [self._vec(t) for t in texts]


class FakeReranker:
    """Доля слов вопроса, найденных в фрагменте, в шкале логитов примерно [-8, 4]."""

    def __init__(self):
        self.calls = 0

    def score(self, query: str, texts):
        self.calls += 1
        q = set(tokens(query))
        out = []
        for t in texts:
            d = set(tokens(t))
            out.append(12.0 * len(q & d) / max(len(q), 1) - 8.0)
        return out


def build_collection(chunks: list, client: QdrantClient = None, collection: str = "nasa_llis") -> QdrantClient:
    """In-memory коллекция Qdrant с той же схемой, что и боевая (dense + sparse IDF, payload как у LangChain)."""
    import src.retrieval.qdrant_store as qs

    client = client or QdrantClient(":memory:")
    old = qs.COLLECTION
    qs.COLLECTION = collection
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # локальный режим не поддерживает payload-индексы
            create_collection(client, recreate=True)
    finally:
        qs.COLLECTION = old
    dense, sparse = FakeDense(), FakeSparse()
    batch = []
    for c in chunks:
        meta = dict(c["metadata"], chunk_id=c["id"])
        sv = sparse.embed_query(c["page_content"])
        batch.append(models.PointStruct(
            id=point_id(c["id"]),
            vector={DENSE_NAME: dense.embed_query(c["page_content"]),
                    SPARSE_NAME: models.SparseVector(indices=sv.indices, values=sv.values)},
            payload={"page_content": c["page_content"], "metadata": meta},
        ))
        if len(batch) == 512:
            client.upsert(collection, batch)
            batch = []
    if batch:
        client.upsert(collection, batch)
    return client
