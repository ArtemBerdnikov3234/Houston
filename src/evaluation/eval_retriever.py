"""Оценка HoustonRetriever на Gold Set: вклад каждого этапа поиска.

Наборы конфигураций (--suite):
  ablation (по умолчанию) — лестница от поиска, который был в проекте, до полного конвейера
  (каждая строка добавляет один этап к предыдущей):
      baseline        hybrid RRF top-10, как в check_qdrant.py
      +abbreviations  + раскрытие аббревиатур в BM25-запросе
      +hints          + подсказки-фильтры из текста вопроса и варианты написания программ
      +candidates     + шире выборка: prefetch_k=50, candidate_k=30
      +rerank         + Cross-Encoder bge-reranker-base
      full            + отбор контекста (порог, группировка по урокам, small-to-big, бюджет) = конфигурация по умолчанию
  search — первичный поиск без реранкера: dense, sparse (BM25), hybrid RRF, hybrid DBSF.

Метрики ранжирования (по первым 10 фрагментам после всех этапов до отбора контекста):
  Lesson Hit@k, Quote Hit@k, MRR@10 — как в check_qdrant.py.
Метрики контекста (что реально получит LLM):
  ctx_lesson_recall — в контексте есть нужный урок;
  ctx_quote_recall  — в тексте контекста есть эталонная цитата;
  ctx_lessons, ctx_tokens — средний размер контекста.

Запуск:
    python -m src.evaluation.eval_retriever                                     # оба Gold Set, ablation
    python -m src.evaluation.eval_retriever --gold data/benchmarks/gold_set_hard.json
    python -m src.evaluation.eval_retriever --suite search
Результат: data/evaluation/retriever_eval_<suite>.json и таблица в консоли (Markdown).
"""

import argparse
import json
import sys
import time
from typing import Callable, Dict, List, Optional

from Houston.src.retrieval.config import BASELINE, DEFAULT, RetrieverConfig
from src.settings import BENCHMARKS_DIR, EVALUATION_DIR

K = 10
GOLD_SETS = [BENCHMARKS_DIR / "gold_set_50.json", BENCHMARKS_DIR / "gold_set_hard.json"]


def ablation_configs() -> Dict[str, RetrieverConfig]:
    abbr = BASELINE.with_(expand_abbreviations=True)
    query = abbr.with_(auto_filters=True, normalize_programs=True)
    wide = query.with_(prefetch_k=50, candidate_k=30)
    rerank = wide.with_(use_reranker=True)
    return {"baseline": BASELINE, "+abbreviations": abbr, "+hints": query, "+candidates": wide,
            "+rerank": rerank, "full": DEFAULT}


def search_configs() -> Dict[str, RetrieverConfig]:
    base = BASELINE.with_(prefetch_k=10, candidate_k=10)
    return {
        "dense": base.with_(mode="dense"),
        "sparse": base.with_(mode="sparse"),
        "hybrid_rrf": base,
        "hybrid_dbsf": base.with_(fusion="dbsf"),
        "hybrid_rrf_p50": base.with_(prefetch_k=50),
        "hybrid_dbsf_p50": base.with_(fusion="dbsf", prefetch_k=50),
    }


SUITES: Dict[str, Callable[[], Dict[str, RetrieverConfig]]] = {"ablation": ablation_configs, "search": search_configs}


# ==================== МЕТРИКИ ====================

def first_rank(items, predicate) -> Optional[int]:
    for i, item in enumerate(items, 1):
        if predicate(item):
            return i
    return None


def question_metrics(g: dict, result) -> dict:
    relevant = set(g.get("relevant_lesson_ids") or [g["lesson_id"]])
    quote, lid = g["context_quote"], g["lesson_id"]
    top = result.ranked[:K]
    lesson_rank = first_rank(top, lambda h: bool(relevant & set(h.lesson_ids)))
    quote_rank = first_rank(top, lambda h: lid in h.lesson_ids and quote in h.text)
    ctx_text = result.context_text
    return {
        "id": g["id"],
        "lesson_rank": lesson_rank,
        "quote_rank": quote_rank,
        "ctx_lesson": any(b.lesson_id in relevant for b in result.context),
        "ctx_quote": quote in ctx_text,
        "ctx_lessons": len(result.context),
        "ctx_tokens": sum(b.tokens for b in result.context),
        "ms": result.timings.get("total_ms", 0.0),
        "used_hints": result.used_hints,
    }


def aggregate(rows: List[dict]) -> dict:
    n = len(rows)
    hit = lambda key, k: sum(r[key] is not None and r[key] <= k for r in rows) / n
    return {
        **{f"lesson_hit@{k}": hit("lesson_rank", k) for k in (1, 3, 5, 10)},
        **{f"quote_hit@{k}": hit("quote_rank", k) for k in (1, 5, 10)},
        "mrr@10": sum(1 / r["lesson_rank"] for r in rows if r["lesson_rank"]) / n,
        "ctx_lesson_recall": sum(r["ctx_lesson"] for r in rows) / n,
        "ctx_quote_recall": sum(r["ctx_quote"] for r in rows) / n,
        "ctx_lessons": sum(r["ctx_lessons"] for r in rows) / n,
        "ctx_tokens": sum(r["ctx_tokens"] for r in rows) / n,
        "latency_ms": sum(r["ms"] for r in rows) / n,
        "hints_used": sum(r["used_hints"] for r in rows),
        "missed_question_ids": [r["id"] for r in rows if r["lesson_rank"] is None],
    }


def evaluate(retriever, gold: List[dict], configs: Dict[str, RetrieverConfig], verbose: bool = True) -> dict:
    """Прогоняет все конфигурации на одном Gold Set. retriever — HoustonRetriever (модели общие)."""
    results = {}
    for name, cfg in configs.items():
        t0 = time.time()
        rows = [question_metrics(g, retriever.retrieve(g["question"], filters=g.get("filter") or None, config=cfg))
                for g in gold]
        results[name] = {**aggregate(rows), "per_question": rows}
        if verbose:
            r = results[name]
            print(f"  {name:<15} Hit@1={r['lesson_hit@1']:.2f}  Hit@5={r['lesson_hit@5']:.2f}  "
                  f"QuoteHit@5={r['quote_hit@5']:.2f}  MRR={r['mrr@10']:.2f}  ctx_quote={r['ctx_quote_recall']:.2f}  "
                  f"({time.time() - t0:.0f} с)", flush=True)
    return results


RANK_COLS = ["lesson_hit@1", "lesson_hit@5", "lesson_hit@10", "quote_hit@1", "quote_hit@5", "mrr@10"]
CTX_COLS = ["ctx_lesson_recall", "ctx_quote_recall", "ctx_lessons", "ctx_tokens", "latency_ms"]


def markdown_table(results: dict) -> str:
    cols = RANK_COLS + CTX_COLS
    head = "| Конфигурация | " + " | ".join(cols) + " |\n|:---|" + "---:|" * len(cols)
    lines = [head]
    for name, r in results.items():
        cells = [f"{r[c]:.0f}" if c in ("ctx_tokens", "latency_ms") else
                 f"{r[c]:.1f}" if c == "ctx_lessons" else f"{r[c]:.2f}" for c in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", action="append", help="Gold Set JSON (можно несколько); по умолчанию оба")
    parser.add_argument("--suite", choices=sorted(SUITES), default="ablation")
    parser.add_argument("--only", help="только эти конфигурации через запятую, например baseline,full")
    parser.add_argument("--out", help="файл результатов (по умолчанию data/evaluation/retriever_eval_<suite>.json)")
    args = parser.parse_args()

    from src.retrieval.retriever import HoustonRetriever

    configs = SUITES[args.suite]()
    if args.only:
        configs = {k: v for k, v in configs.items() if k in args.only.split(",")}
    need_rerank = any(c.use_reranker for c in configs.values())
    retriever = HoustonRetriever.from_config(DEFAULT.with_(use_reranker=need_rerank))

    report = {"suite": args.suite, "configs": {k: v.__dict__ for k, v in configs.items()}, "gold_sets": {}}
    for path in args.gold or [str(p) for p in GOLD_SETS]:
        with open(path, encoding="utf-8") as f:
            gold = json.load(f)
        print(f"\n=== {path} ({len(gold)} вопросов) ===")
        res = evaluate(retriever, gold, configs)
        report["gold_sets"][str(path)] = res
        print("\n" + markdown_table(res))

    out = args.out or str(EVALUATION_DIR / f"retriever_eval_{args.suite}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nРезультаты сохранены в {out}")


if __name__ == "__main__":
    main()
