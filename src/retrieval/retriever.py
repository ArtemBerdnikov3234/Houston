"""HoustonRetriever — полный конвейер поиска для RAG (LangChain BaseRetriever).

    вопрос
      -> process_query: нормализация, аббревиатуры для BM25, подсказки-фильтры (центр, период, номер урока)
      -> явные фильтры (program расширяется на все варианты написания)
      -> гибридный поиск: dense top-prefetch_k + BM25 top-prefetch_k -> RRF -> candidate_k
         + если в вопросе есть подсказки: тот же поиск с подсказками-фильтрами, выдачи сливаются (RRF)
      -> реранкер bge-reranker-base: сортировка candidate_k кандидатов
      -> отбор контекста: порог, final_k уроков по max_per_lesson фрагментов, small-to-big, бюджет токенов

Использование:
    from src.retrieval.retriever import HoustonRetriever
    retriever = HoustonRetriever.from_config()                    # модели загружаются один раз
    docs = retriever.invoke("How to avoid hydrogen leaks on the pad?")             # LangChain Documents
    docs = retriever.invoke("...", filters={"org_code": "KSC"})
    result = retriever.retrieve("...")                            # все этапы: кандидаты, ранги, контекст, тайминги
    print(result.context_text)

CLI:
    python -m src.retrieval.retriever "Why did the Mars Climate Orbiter fail?"
    python -m src.retrieval.retriever "..." --filter org_code=JPL --no-rerank
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from src.retrieval.config import RetrieverConfig
from src.retrieval.context import ContextBlock, LessonStore, default_lesson_store, format_context, select_context
from src.retrieval.query_processing import ProcessedQuery, ProgramNormalizer, get_abbreviations, process_query
from src.retrieval.reranker import Reranker, rerank_hits
from src.retrieval.search import FilterSpec, SearchHit, Searcher, merge_filters

RRF_K = 60   # константа RRF при слиянии основной выдачи и выдачи с подсказками-фильтрами


@dataclass
class RetrievalResult:
    """Все промежуточные результаты одного поиска (для отладки, оценки и цепочки RAG)."""

    query: ProcessedQuery
    filters: Optional[Dict[str, Any]]           # явные фильтры после нормализации программ
    candidates: List[SearchHit]                 # после первичного поиска (до реранкера)
    ranked: List[SearchHit]                     # после реранкера (или те же кандидаты без него)
    context: List[ContextBlock]                 # отобранный контекст для LLM
    timings: Dict[str, float] = field(default_factory=dict)
    used_hints: bool = False

    @property
    def context_text(self) -> str:
        return format_context(self.context)

    @property
    def documents(self) -> List[Document]:
        return [b.to_document() for b in self.context]


def rrf_merge(lists: List[List[SearchHit]], limit: int, k: int = RRF_K) -> List[SearchHit]:
    """Слияние нескольких выдач через Reciprocal Rank Fusion; дубли по chunk_id объединяются."""
    scores: Dict[str, float] = {}
    first: Dict[str, SearchHit] = {}
    for hits in lists:
        for rank, h in enumerate(hits, 1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
            first.setdefault(h.chunk_id, h)
    order = sorted(scores, key=lambda cid: -scores[cid])[:limit]
    out = []
    for rank, cid in enumerate(order, 1):
        h = first[cid]
        out.append(SearchHit(chunk_id=h.chunk_id, text=h.text, metadata=h.metadata, score=scores[cid],
                             rank=rank, source=h.source))
    return out


class HoustonRetriever(BaseRetriever):
    """Ретривер Houston. Совместим с LangChain (invoke / batch / LCEL)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: RetrieverConfig = RetrieverConfig()
    searcher: Any = None                      # Searcher
    reranker: Optional[Any] = None            # Reranker (None — без переранжирования)
    lessons: Optional[Any] = None             # LessonStore для расширения контекста
    programs: Optional[Any] = None            # ProgramNormalizer
    abbreviations: Optional[Dict[str, str]] = None

    @classmethod
    def from_config(cls, config: Optional[RetrieverConfig] = None, *, searcher: Optional[Searcher] = None,
                    reranker: Optional[Reranker] = None, lessons: Optional[LessonStore] = None,
                    abbreviations: Optional[Dict[str, str]] = None) -> "HoustonRetriever":
        """Собирает ретривер с моделями по умолчанию (загружаются только нужные конфигурации)."""
        config = config or RetrieverConfig.from_env()
        if searcher is None:
            searcher = Searcher()
        if reranker is None and config.use_reranker:
            from src.retrieval.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker(config.rerank_model, batch_size=config.rerank_batch_size)
        if lessons is None and (config.expand_sections or config.normalize_programs):
            lessons = default_lesson_store()
        programs = ProgramNormalizer(lessons.programs()) if (lessons and config.normalize_programs) else None
        if abbreviations is None and config.expand_abbreviations:
            abbreviations = get_abbreviations()
        return cls(config=config, searcher=searcher, reranker=reranker, lessons=lessons, programs=programs,
                   abbreviations=abbreviations)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, filters: FilterSpec = None, config: Optional[RetrieverConfig] = None) -> RetrievalResult:
        cfg = config or self.config
        t0 = time.perf_counter()
        pq = process_query(query, abbreviations=self.abbreviations or {},
                           expand=cfg.expand_abbreviations, extract_filters=cfg.auto_filters)
        if isinstance(filters, dict) and cfg.normalize_programs and self.programs is not None:
            filters = self.programs.expand_filter(filters)

        search_kw = dict(k=cfg.candidate_k, mode=cfg.mode, fusion=cfg.fusion, prefetch_k=cfg.prefetch_k,
                         sparse_query=pq.sparse_text if cfg.expand_abbreviations else None)
        candidates = self.searcher.search(pq.text, filters=filters, **search_kw)
        timings = {f"main_{k}": v for k, v in self.searcher.last_timings.items()}

        used_hints = False
        if cfg.auto_filters and pq.hint_filters:
            hinted = self.searcher.search(pq.text, filters=merge_filters(filters, pq.hint_filters), **search_kw)
            if len(hinted) >= cfg.auto_filter_min_hits:
                candidates = rrf_merge([candidates, hinted], limit=cfg.candidate_k)
                used_hints = True
        t1 = time.perf_counter()

        if cfg.use_reranker and self.reranker is not None and candidates:
            ranked = rerank_hits(self.reranker, pq.text, [self._copy(h) for h in candidates])
        else:
            ranked = list(candidates)
        t2 = time.perf_counter()

        context = select_context(ranked, cfg, self.lessons)
        t3 = time.perf_counter()
        timings.update(search_ms=(t1 - t0) * 1000, rerank_ms=(t2 - t1) * 1000, context_ms=(t3 - t2) * 1000,
                       total_ms=(t3 - t0) * 1000)
        return RetrievalResult(query=pq, filters=filters if isinstance(filters, dict) else None,
                               candidates=candidates, ranked=ranked, context=context, timings=timings,
                               used_hints=used_hints)

    @staticmethod
    def _copy(h: SearchHit) -> SearchHit:
        return SearchHit(chunk_id=h.chunk_id, text=h.text, metadata=h.metadata, score=h.score, rank=h.rank,
                         source=h.source)

    # --- LangChain ---
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
                                filters: FilterSpec = None, **kwargs) -> List[Document]:
        """Документы контекста: по одному на урок, с метаданными lesson_id / title / url / score."""
        return self.retrieve(query, filters=filters).documents


def _parse_filters(items: List[str]) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    for item in items or []:
        key, _, value = item.partition("=")
        out[key.strip()] = int(value) if value.strip().isdigit() and key.strip() == "year" else value.strip()
    return out or None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Поиск по базе NASA LLIS (все этапы ретривера)")
    parser.add_argument("query")
    parser.add_argument("--filter", action="append", default=[], help="поле=значение, например org_code=KSC")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--final-k", type=int)
    parser.add_argument("--show-context", action="store_true", help="напечатать текст контекста для LLM")
    args = parser.parse_args()

    overrides = {"use_reranker": False} if args.no_rerank else {}
    if args.final_k:
        overrides["final_k"] = args.final_k
    retriever = HoustonRetriever.from_config(RetrieverConfig.from_env(**overrides))
    res = retriever.retrieve(args.query, filters=_parse_filters(args.filter))

    print(f"Запрос : {res.query.text}\nОбработка: {res.query.describe()}"
          + ("  (подсказки применены)" if res.used_hints else ""))
    print(f"\nTop-10 после {'реранкера' if retriever.config.use_reranker else 'гибридного поиска'}:")
    for i, h in enumerate(res.ranked[:10], 1):
        m = h.metadata
        rs = f"{h.rerank_score:7.2f}" if h.rerank_score is not None else "      -"
        print(f"  {i:>2}. {rs} (было #{h.rank:<2}) #{h.lesson_id:<6} {m.get('org_code', ''):<5} "
              f"{m.get('section', ''):<15} {m.get('title', '')[:60]}")
    print(f"\nКонтекст для LLM: {len(res.context)} урок(ов), ~{sum(b.tokens for b in res.context)} токенов")
    for b in res.context:
        print(f"  [{b.rank}] {b.header()[:100]}  секции: {', '.join(b.sections)}")
    print("\nВремя, мс: " + ", ".join(f"{k}={v:.0f}" for k, v in res.timings.items()))
    if args.show_context:
        print("\n" + "=" * 70 + "\n" + res.context_text)


if __name__ == "__main__":
    main()
