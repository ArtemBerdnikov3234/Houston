"""Переранжирование кандидатов Cross-Encoder'ом.

Первичный поиск (bi-encoder bge + BM25) сравнивает вектор вопроса с заранее посчитанными векторами
чанков и быстро отбирает кандидатов. Cross-Encoder читает вопрос и фрагмент вместе и точнее оценивает,
отвечает ли именно этот фрагмент на вопрос. Он медленнее, поэтому применяется только к candidate_k
кандидатам. Это исправляет главный промах ретривера на Hard Gold Set: урок найден, а в топе его
Abstract или заголовок, а не секция Recommendation / Evidence с ответом.

Модель: BAAI/bge-reranker-base через FastEmbed (ONNX, CPU, без PyTorch), скачивается в
.fastembed_cache/ при первом запуске (~1 ГБ). Score — логит релевантности (больше — лучше, без
фиксированной шкалы; ориентировочно > 0 — релевантно, < -5 — нет).

Пример:
    from src.retrieval.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    hits = reranker.rerank("why did the gauge rupture?", hits)     # hits отсортированы по rerank_score
"""

import time
from typing import List, Optional, Protocol, Sequence

from src.retrieval.config import RERANK_MODEL
from src.retrieval.search import SearchHit


class Reranker(Protocol):
    """Любой реранкер: оценивает пары (вопрос, текст)."""

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        ...


class CrossEncoderReranker:
    """BAAI/bge-reranker-base (или другая модель FastEmbed TextCrossEncoder)."""

    def __init__(self, model_name: str = RERANK_MODEL, batch_size: int = 16, cache_dir: Optional[str] = None,
                 threads: Optional[int] = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        from src.retrieval.qdrant_store import CACHE_DIR

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir or CACHE_DIR, threads=threads)
        self.last_ms = 0.0

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        t0 = time.perf_counter()
        scores = [float(s) for s in self.model.rerank(query, list(texts), batch_size=self.batch_size)]
        self.last_ms = (time.perf_counter() - t0) * 1000
        return scores

    def rerank(self, query: str, hits: List[SearchHit]) -> List[SearchHit]:
        return rerank_hits(self, query, hits)


def rerank_hits(reranker: Reranker, query: str, hits: List[SearchHit]) -> List[SearchHit]:
    """Проставляет rerank_score и сортирует по нему; при равенстве сохраняется порядок первичного поиска."""
    scores = reranker.score(query, [h.text for h in hits])
    for h, s in zip(hits, scores):
        h.rerank_score = s
    return sorted(hits, key=lambda h: (-h.rerank_score, h.rank))
