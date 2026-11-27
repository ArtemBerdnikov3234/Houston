"""Первичный поиск по Qdrant: семантический (dense), лексический (BM25) и гибридный.

В отличие от QdrantVectorStore.similarity_search, здесь поиск идет напрямую через Query API Qdrant:
- запрос векторизуется один раз и переиспользуется (dense и sparse считаются отдельно);
- в гибридном режиме размер выборки каждого поиска (prefetch_k) и итоговое число кандидатов
  (candidate_k) задаются независимо, а слияние выбирается: RRF или DBSF;
- результат — SearchHit со score, рангом и источником, который удобно передавать в реранкер.

Модели и клиент берутся из qdrant_store.py, поэтому поиск работает с той же коллекцией и теми же
эмбеддингами, что и индексация.

Пример:
    from src.retrieval.search import Searcher
    searcher = Searcher()
    hits = searcher.search("battery thermal runaway", k=10)                     # hybrid RRF
    hits = searcher.search("hydrazine", k=10, mode="dense", filters={"org_code": "KSC"})
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient, models

from src.retrieval.qdrant_store import COLLECTION, DENSE_NAME, SPARSE_NAME

FilterSpec = Union[None, models.Filter, Dict[str, Any]]


@dataclass
class SearchHit:
    """Найденный фрагмент (чанк) с оценками на разных этапах поиска."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float                       # score первичного поиска (cosine / BM25 / RRF / DBSF)
    rank: int                          # позиция в первичной выдаче, с 1
    source: str = "hybrid"             # dense | sparse | hybrid
    rerank_score: Optional[float] = None

    @property
    def lesson_id(self) -> str:
        return str(self.metadata.get("lesson_id", ""))

    @property
    def lesson_ids(self) -> List[str]:
        """Все уроки, где встречается текст чанка (одинаковые фрагменты хранятся одним чанком)."""
        return [str(x) for x in self.metadata.get("lesson_ids") or [self.lesson_id]]

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score

    def to_document(self):
        from langchain_core.documents import Document

        meta = dict(self.metadata, chunk_id=self.chunk_id, score=self.score, rank=self.rank,
                    rerank_score=self.rerank_score)
        return Document(id=self.chunk_id, page_content=self.text, metadata=meta)


# ==================== ФИЛЬТРЫ ====================

def build_filter(filters: FilterSpec) -> Optional[models.Filter]:
    """Фильтр Qdrant из словаря условий по полям metadata.

    Значение может быть:
      - скаляром:        {"org_code": "KSC"}                 -> точное совпадение;
      - списком:         {"org_code": ["KSC", "JSC"]}        -> любое из значений;
      - диапазоном:      {"year": {"gte": 1990, "lte": 1999}} -> Range (для чисел).
    Готовый models.Filter возвращается как есть.
    """
    if filters is None or isinstance(filters, models.Filter):
        return filters
    must = []
    for field_name, value in filters.items():
        key = field_name if field_name.startswith("metadata.") else f"metadata.{field_name}"
        if isinstance(value, dict):
            must.append(models.FieldCondition(key=key, range=models.Range(**value)))
        elif isinstance(value, (list, tuple, set)):
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(value))))
        else:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=must) if must else None


def merge_filters(*specs: FilterSpec) -> Optional[models.Filter]:
    """Объединяет несколько фильтров через AND."""
    built = [build_filter(s) for s in specs]
    built = [f for f in built if f is not None]
    if not built:
        return None
    if len(built) == 1:
        return built[0]
    return models.Filter(must=built)


# ==================== ПОИСК ====================

class Searcher:
    """Dense / sparse / hybrid поиск по коллекции Qdrant через Query API.

    embedding — объект с методом embed_query(text) -> List[float] (BGEEmbeddings из qdrant_store);
    sparse_embedding — объект с методом embed_query(text) -> SparseVector-подобный объект
    с полями indices / values (FastEmbedSparse из langchain_qdrant).
    """

    def __init__(self, client: Optional[QdrantClient] = None, embedding=None, sparse_embedding=None,
                 collection: str = COLLECTION):
        if client is None or embedding is None or sparse_embedding is None:
            from langchain_qdrant import FastEmbedSparse

            from src.retrieval.qdrant_store import CACHE_DIR, SPARSE_MODEL, BGEEmbeddings, get_client
            client = client or get_client()
            embedding = embedding or BGEEmbeddings()
            sparse_embedding = sparse_embedding or FastEmbedSparse(model_name=SPARSE_MODEL, cache_dir=CACHE_DIR)
        self.client = client
        self.embedding = embedding
        self.sparse_embedding = sparse_embedding
        self.collection = collection
        self.last_timings: Dict[str, float] = {}

    # --- векторизация запроса ---
    def _dense_vector(self, text: str) -> List[float]:
        return list(self.embedding.embed_query(text))

    def _sparse_vector(self, text: str) -> models.SparseVector:
        vec = self.sparse_embedding.embed_query(text)
        return models.SparseVector(indices=list(vec.indices), values=list(vec.values))

    # --- поиск ---
    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "hybrid",
        filters: FilterSpec = None,
        fusion: str = "rrf",
        prefetch_k: Optional[int] = None,
        sparse_query: Optional[str] = None,
    ) -> List[SearchHit]:
        """Возвращает top-k чанков.

        query        — текст для dense-поиска (смысл вопроса);
        sparse_query — текст для BM25 (по умолчанию тот же; сюда попадает запрос с раскрытыми аббревиатурами);
        prefetch_k   — сколько кандидатов берет каждый поиск перед слиянием (по умолчанию = k).
        """
        qfilter = build_filter(filters)
        prefetch_k = max(prefetch_k or k, k)
        sparse_query = sparse_query or query
        t0 = time.perf_counter()

        if mode == "dense":
            dense = self._dense_vector(query)
            t1 = time.perf_counter()
            resp = self.client.query_points(self.collection, query=dense, using=DENSE_NAME, limit=k,
                                            query_filter=qfilter, with_payload=True)
        elif mode == "sparse":
            sparse = self._sparse_vector(sparse_query)
            t1 = time.perf_counter()
            resp = self.client.query_points(self.collection, query=sparse, using=SPARSE_NAME, limit=k,
                                            query_filter=qfilter, with_payload=True)
        elif mode == "hybrid":
            dense, sparse = self._dense_vector(query), self._sparse_vector(sparse_query)
            t1 = time.perf_counter()
            fusion_kind = {"rrf": models.Fusion.RRF, "dbsf": models.Fusion.DBSF}[fusion]
            resp = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense, using=DENSE_NAME, limit=prefetch_k, filter=qfilter),
                    models.Prefetch(query=sparse, using=SPARSE_NAME, limit=prefetch_k, filter=qfilter),
                ],
                query=models.FusionQuery(fusion=fusion_kind),
                limit=k,
                query_filter=qfilter,
                with_payload=True,
            )
        else:
            raise ValueError(f"Неизвестный режим поиска: {mode!r}")

        t2 = time.perf_counter()
        self.last_timings = {"embed_ms": (t1 - t0) * 1000, "search_ms": (t2 - t1) * 1000}
        return [self._to_hit(p, rank, mode) for rank, p in enumerate(resp.points, 1)]

    def semantic_search(self, query: str, k: int = 10, filters: FilterSpec = None) -> List[SearchHit]:
        """Чисто семантический поиск (bge-base-en-v1.5, cosine)."""
        return self.search(query, k=k, mode="dense", filters=filters)

    def keyword_search(self, query: str, k: int = 10, filters: FilterSpec = None) -> List[SearchHit]:
        """Лексический поиск BM25: точные термины, номера документов, аббревиатуры."""
        return self.search(query, k=k, mode="sparse", filters=filters)

    def hybrid_search(self, query: str, k: int = 10, filters: FilterSpec = None, fusion: str = "rrf",
                      prefetch_k: Optional[int] = None, sparse_query: Optional[str] = None) -> List[SearchHit]:
        """Гибридный поиск: dense + BM25, объединенные через RRF или DBSF."""
        return self.search(query, k=k, mode="hybrid", filters=filters, fusion=fusion,
                           prefetch_k=prefetch_k, sparse_query=sparse_query)

    @staticmethod
    def _to_hit(point, rank: int, mode: str) -> SearchHit:
        payload = point.payload or {}
        metadata = dict(payload.get("metadata") or {})
        chunk_id = str(metadata.get("chunk_id") or point.id)
        return SearchHit(chunk_id=chunk_id, text=payload.get("page_content", ""), metadata=metadata,
                         score=float(point.score), rank=rank, source=mode)
