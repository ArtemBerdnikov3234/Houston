"""Подбор Top-K и порога реранкера на Gold Set.

Перебираются:
  prefetch_k   — сколько кандидатов берет каждый поиск (dense и BM25) перед слиянием;
  candidate_k  — сколько кандидатов после слияния идет в реранкер (главный параметр скорости);
  final_k      — сколько уроков уходит в контекст LLM.
Для каждого вопроса поиск выполняется один раз на максимальную глубину, реранкер оценивает всех
кандидатов один раз; меньшие candidate_k — префиксы той же выдачи, поэтому перебор быстрый.
Время реранкера для candidate_k оценивается по замеренному времени на одну пару.

Дополнительно считается распределение score реранкера для фрагментов с эталонной цитатой и для
остальных, и предлагается порог min_rerank_score, который сохраняет 95% эталонных фрагментов.

Запуск:
    python -m src.evaluation.tune_top_k                                   # Hard Gold Set
    python -m src.evaluation.tune_top_k --gold data/benchmarks/gold_set_50.json
    python -m src.evaluation.tune_top_k --rerank-model Xenova/ms-marco-MiniLM-L-6-v2
Результат: data/evaluation/top_k_tuning_<набор>.json и таблицы в консоли.
"""

import argparse
import json
import os
import sys
import time
from typing import List, Sequence

from Houston.src.retrieval.config import DEFAULT, RetrieverConfig
from Houston.src.retrieval.context import select_context
from Houston.src.retrieval.query_processing import ProgramNormalizer, get_abbreviations, process_query
from src.retrieval.search import SearchHit, merge_filters
from src.retrieval.retriever import rrf_merge
from src.settings import BENCHMARKS_DIR, EVALUATION_DIR

PREFETCH_GRID = (20, 50, 100)
CANDIDATE_GRID = (10, 20, 30, 50, 80)
FINAL_GRID = (2, 3, 4, 5, 6)


def _copy(h: SearchHit) -> SearchHit:
    return SearchHit(chunk_id=h.chunk_id, text=h.text, metadata=h.metadata, score=h.score, rank=h.rank,
                     source=h.source)


def collect(searcher, reranker, gold: List[dict], prefetch_k: int, max_candidates: int, cfg: RetrieverConfig,
            programs=None, abbreviations=None) -> List[dict]:
    """Для каждого вопроса: кандидаты глубины max_candidates + rerank score каждого (один проход)."""
    items, pair_ms, pairs = [], 0.0, 0
    for g in gold:
        pq = process_query(g["question"], abbreviations=abbreviations or {})
        filters = g.get("filter") or None
        if filters and programs is not None:
            filters = programs.expand_filter(filters)
        # каждая ветка (dense, BM25) дает до prefetch_k кандидатов, после слияния их не больше 2 * prefetch_k
        kw = dict(k=min(max_candidates, 2 * prefetch_k), mode="hybrid", fusion=cfg.fusion, prefetch_k=prefetch_k,
                  sparse_query=pq.sparse_text)
        hits = searcher.search(pq.text, filters=filters, **kw)
        if pq.hint_filters:
            hinted = searcher.search(pq.text, filters=merge_filters(filters, pq.hint_filters), **kw)
            if len(hinted) >= cfg.auto_filter_min_hits:
                hits = rrf_merge([hits, hinted], limit=kw["k"])
        t0 = time.perf_counter()
        scores = reranker.score(pq.text, [h.text for h in hits]) if hits else []
        pair_ms += (time.perf_counter() - t0) * 1000
        pairs += len(hits)
        items.append({"gold": g, "query": pq.text, "hits": hits, "scores": scores})
    for it in items:
        it["ms_per_pair"] = pair_ms / max(pairs, 1)
    return items


def ranked_prefix(item: dict, candidate_k: int) -> List[SearchHit]:
    hits = [_copy(h) for h in item["hits"][:candidate_k]]
    for h, s in zip(hits, item["scores"][:candidate_k]):
        h.rerank_score = s
    return sorted(hits, key=lambda h: (-h.rerank_score, h.rank))


def metrics_for(items: List[dict], candidate_k: int) -> dict:
    n = len(items)
    hit1 = hit5 = quote5 = rr = 0.0
    for it in items:
        g = it["gold"]
        relevant = set(g.get("relevant_lesson_ids") or [g["lesson_id"]])
        ranked = ranked_prefix(it, candidate_k)[:10]
        lr = next((i for i, h in enumerate(ranked, 1) if relevant & set(h.lesson_ids)), None)
        qr = next((i for i, h in enumerate(ranked, 1)
                   if g["lesson_id"] in h.lesson_ids and g["context_quote"] in h.text), None)
        hit1 += bool(lr and lr <= 1)
        hit5 += bool(lr and lr <= 5)
        quote5 += bool(qr and qr <= 5)
        rr += 1 / lr if lr else 0
    ms = items[0]["ms_per_pair"] * candidate_k if items else 0
    return {"lesson_hit@1": hit1 / n, "lesson_hit@5": hit5 / n, "quote_hit@5": quote5 / n, "mrr@10": rr / n,
            "rerank_ms_est": ms}


def context_metrics(items: List[dict], candidate_k: int, final_k: int, cfg: RetrieverConfig, lessons) -> dict:
    c = cfg.with_(final_k=final_k, candidate_k=max(candidate_k, final_k))  # candidate_k тут только для проверки
    n = len(items)
    lesson = quote = tokens = blocks = 0.0
    for it in items:
        g = it["gold"]
        relevant = set(g.get("relevant_lesson_ids") or [g["lesson_id"]])
        ctx = select_context(ranked_prefix(it, candidate_k), c, lessons)
        text = "\n".join(b.text for b in ctx)
        lesson += any(b.lesson_id in relevant for b in ctx)
        quote += g["context_quote"] in text
        tokens += sum(b.tokens for b in ctx)
        blocks += len(ctx)
    return {"ctx_lesson_recall": lesson / n, "ctx_quote_recall": quote / n, "ctx_tokens": tokens / n,
            "ctx_lessons": blocks / n}


def threshold_stats(items: List[dict], candidate_k: int) -> dict:
    """Score реранкера у эталонных фрагментов и у остальных; порог, сохраняющий 95% эталонных."""
    pos, neg = [], []
    for it in items:
        g = it["gold"]
        for h, s in zip(it["hits"][:candidate_k], it["scores"][:candidate_k]):
            (pos if (g["lesson_id"] in h.lesson_ids and g["context_quote"] in h.text) else neg).append(s)
    if not pos:
        return {}
    pos_sorted = sorted(pos)
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    thr = pos_sorted[int(0.05 * len(pos_sorted))]
    return {"positives": len(pos), "negatives": len(neg),
            "pos_p05": thr, "pos_median": q(pos, 0.5), "neg_median": q(neg, 0.5), "neg_p90": q(neg, 0.9),
            "suggested_min_rerank_score": round(thr - 0.5, 1),
            "neg_dropped_share": sum(s < thr - 0.5 for s in neg) / max(len(neg), 1)}


def pick(table: List[dict], key: str, metric: str, tol: float) -> int:
    """Наименьшее значение параметра, у которого метрика не хуже лучшей больше чем на tol."""
    best = max(r[metric] for r in table)
    return min(r[key] for r in table if r[metric] >= best - tol)


def run(searcher, reranker, lessons, gold: List[dict], cfg: RetrieverConfig = DEFAULT,
        prefetch_grid: Sequence[int] = PREFETCH_GRID, candidate_grid: Sequence[int] = CANDIDATE_GRID,
        final_grid: Sequence[int] = FINAL_GRID, programs=None, abbreviations=None) -> dict:
    report = {"prefetch": [], "candidates": [], "final": [], "threshold": {}}
    items_by_prefetch = {}
    for p in prefetch_grid:
        items = collect(searcher, reranker, gold, p, max(candidate_grid), cfg, programs, abbreviations)
        items_by_prefetch[p] = items
        for ck in candidate_grid:
            report["candidates" if p == cfg.prefetch_k else "prefetch"].append(
                {"prefetch_k": p, "candidate_k": ck, **metrics_for(items, ck)})
    base_items = items_by_prefetch.get(cfg.prefetch_k) or items_by_prefetch[max(prefetch_grid)]
    cand_rows = [r for r in report["candidates"]] or [r for r in report["prefetch"] if r["prefetch_k"] == max(prefetch_grid)]
    best_ck = pick(cand_rows, "candidate_k", "mrr@10", 0.01)
    for fk in final_grid:
        report["final"].append({"candidate_k": best_ck, "final_k": fk,
                                **context_metrics(base_items, best_ck, fk, cfg, lessons)})
    report["threshold"] = threshold_stats(base_items, best_ck)
    report["recommended"] = {
        "candidate_k": best_ck,
        "final_k": pick(report["final"], "final_k", "ctx_quote_recall", 0.02),
        "min_rerank_score": report["threshold"].get("suggested_min_rerank_score"),
    }
    return report


def print_report(rep: dict) -> None:
    print(f"\n{'prefetch':>9}{'cand_k':>8}{'Hit@1':>8}{'Hit@5':>8}{'QHit@5':>8}{'MRR':>7}{'rerank мс':>11}")
    for r in sorted(rep["prefetch"] + rep["candidates"], key=lambda r: (r["prefetch_k"], r["candidate_k"])):
        print(f"{r['prefetch_k']:>9}{r['candidate_k']:>8}{r['lesson_hit@1']:>8.2f}{r['lesson_hit@5']:>8.2f}"
              f"{r['quote_hit@5']:>8.2f}{r['mrr@10']:>7.2f}{r['rerank_ms_est']:>11.0f}")
    print(f"\n{'final_k':>8}{'ctx_lesson':>12}{'ctx_quote':>11}{'уроков':>8}{'токенов':>9}")
    for r in rep["final"]:
        print(f"{r['final_k']:>8}{r['ctx_lesson_recall']:>12.2f}{r['ctx_quote_recall']:>11.2f}"
              f"{r['ctx_lessons']:>8.1f}{r['ctx_tokens']:>9.0f}")
    if rep["threshold"]:
        t = rep["threshold"]
        print(f"\nScore реранкера: эталонные фрагменты медиана {t['pos_median']:.2f}, 5% {t['pos_p05']:.2f}; "
              f"остальные медиана {t['neg_median']:.2f}, 90% {t['neg_p90']:.2f}. "
              f"Порог {t['suggested_min_rerank_score']} отсекает {t['neg_dropped_share']:.0%} нерелевантных.")
    print(f"\nРекомендация: {rep['recommended']}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(BENCHMARKS_DIR / "gold_set_hard.json"))
    parser.add_argument("--rerank-model", default=DEFAULT.rerank_model)
    parser.add_argument("--out")
    args = parser.parse_args()

    from Houston.src.retrieval.context import default_lesson_store
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.search import Searcher

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)
    lessons = default_lesson_store()
    rep = run(Searcher(), CrossEncoderReranker(args.rerank_model), lessons, gold,
              programs=ProgramNormalizer(lessons.programs()), abbreviations=get_abbreviations())
    rep["gold"], rep["rerank_model"] = args.gold, args.rerank_model
    print_report(rep)
    stem = os.path.splitext(os.path.basename(args.gold))[0].removeprefix("gold_set_")
    model = args.rerank_model.split("/")[-1]
    out = args.out or str(EVALUATION_DIR / f"top_k_tuning_{stem}{'' if model == 'bge-reranker-base' else '_' + model}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в {out}")


if __name__ == "__main__":
    main()
