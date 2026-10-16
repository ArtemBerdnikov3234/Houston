"""Проверка готовности Qdrant и качества поиска по Gold Set.

1. Состояние коллекции: число точек, конфигурация векторов, payload-индексы.
2. Демонстрационные запросы (гибридный поиск, поиск с фильтром по метаданным).
3. Оценка ретривера на Gold Set в трех режимах: DENSE, SPARSE (BM25), HYBRID (RRF).
   Вопрос может задавать "filter" (поиск с фильтром по метаданным) и "relevant_lesson_ids"
   (ответ есть в нескольких уроках — засчитывается любой из них).
   - Lesson Hit@k — нужный урок есть среди top-k чанков;
   - Quote Hit@k  — среди top-k есть именно тот чанк, который содержит эталонную цитату;
   - MRR@10       — средний обратный ранг первого чанка нужного урока.

Запуск:  python check_qdrant.py            # всё
         python check_qdrant.py --eval-only
         python check_qdrant.py --eval-only --gold gold_set_hard.json
Результат оценки сохраняется в retrieval_eval.json (gold_set_50) или retrieval_eval_<набор>.json.
"""

import argparse
import json
import os
import sys
import time

from langchain_qdrant import FastEmbedSparse, RetrievalMode
from qdrant_client import models

from qdrant_store import (CACHE_DIR, COLLECTION, SPARSE_MODEL, BGEEmbeddings, field_filter,
                          get_client, get_vector_store)

K = 10
DEMO_QUERIES = [
    ("What caused the hydraulic pressure gauge failure at the launch equipment test facility?", None),
    ("lessons about battery thermal runaway", None),
    ("How should flight software be tested before launch?", ("org_code", "JPL")),
    ("How to prevent hydrogen leaks during launch pad operations?", ("sections", "recommendation")),
]


def show_collection(client) -> None:
    info = client.get_collection(COLLECTION)
    p = info.config.params
    print(f"Коллекция         : {COLLECTION} (статус: {info.status})")
    print(f"Точек             : {info.points_count}")
    print(f"Dense-векторы     : " + ", ".join(f"{n}: dim={v.size}, {v.distance}" for n, v in p.vectors.items()))
    print(f"Sparse-векторы    : " + ", ".join(f"{n}: modifier={v.modifier}" for n, v in p.sparse_vectors.items()))
    print(f"Payload-индексы   : " + ", ".join(f"{k} ({v.data_type})" for k, v in info.payload_schema.items()))


def show_demo(store) -> None:
    for query, flt in DEMO_QUERIES:
        kw = {"filter": field_filter(*flt)} if flt else {}
        label = f"  [фильтр {flt[0]}={flt[1]}]" if flt else ""
        print(f"\nQ: {query}{label}")
        for doc, score in store.similarity_search_with_score(query, k=5, **kw):
            m = doc.metadata
            print(f"  {score:.3f}  #{m['lesson_id']:<6} {m['org_code']:<5} {m['section']:<15} {m['title'][:70]}")


def gold_filter(g: dict):
    """Фильтр вопроса Gold Set: {"org_code": "KSC"} -> must-условия по полям metadata."""
    conds = [models.FieldCondition(key=f"metadata.{f}", match=models.MatchValue(value=v))
             for f, v in g.get("filter", {}).items()]
    return models.Filter(must=conds) if conds else None


def evaluate(stores: dict, gold: list) -> dict:
    """Метрики по режимам; урок засчитывается, если это любой из relevant_lesson_ids (по умолчанию — lesson_id)."""
    results = {}
    for mode, store in stores.items():
        lesson_hits = {1: 0, 5: 0, 10: 0}
        quote_hits = {1: 0, 5: 0, 10: 0}
        rr_sum = 0.0
        misses = []
        first_ranks = {}
        t0 = time.time()
        for g in gold:
            relevant = set(g.get("relevant_lesson_ids") or [g["lesson_id"]])
            docs = store.similarity_search(g["question"], k=K, filter=gold_filter(g))
            lesson_ranks = [i for i, d in enumerate(docs, 1) if relevant & set(d.metadata["lesson_ids"])]
            quote_ranks = [i for i, d in enumerate(docs, 1)
                           if g["lesson_id"] in d.metadata["lesson_ids"] and g["context_quote"] in d.page_content]
            for k in lesson_hits:
                lesson_hits[k] += bool(lesson_ranks and lesson_ranks[0] <= k)
                quote_hits[k] += bool(quote_ranks and quote_ranks[0] <= k)
            rr_sum += 1 / lesson_ranks[0] if lesson_ranks else 0
            first_ranks[g["id"]] = lesson_ranks[0] if lesson_ranks else None
            if not lesson_ranks:
                misses.append(g["id"])
        n = len(gold)
        results[mode] = {
            **{f"lesson_hit@{k}": v / n for k, v in lesson_hits.items()},
            **{f"quote_hit@{k}": v / n for k, v in quote_hits.items()},
            "mrr@10": rr_sum / n,
            "missed_question_ids": misses,
            "latency_ms": (time.time() - t0) / n * 1000,
            "groups": group_metrics(gold, first_ranks),
        }
    return results


def group_metrics(gold: list, first_ranks: dict) -> dict:
    """Lesson Hit@5 и MRR@10 по срезам: секция цитаты, вопросы с фильтром, вопросы с несколькими уроками."""
    groups = {}
    for g in gold:
        keys = [f"section={g['section']}"]
        keys.append("filtered" if g.get("filter") else "unfiltered")
        if len(g.get("relevant_lesson_ids") or []) > 1:
            keys.append("multi_lesson")
        for key in keys:
            groups.setdefault(key, []).append(first_ranks[g["id"]])
    return {
        key: {"n": len(ranks),
              "lesson_hit@5": sum(r is not None and r <= 5 for r in ranks) / len(ranks),
              "mrr@10": sum(1 / r for r in ranks if r) / len(ranks)}
        for key, ranks in sorted(groups.items())
    }


def print_eval(results: dict, n: int) -> None:
    cols = ["lesson_hit@1", "lesson_hit@5", "lesson_hit@10", "quote_hit@5", "quote_hit@10", "mrr@10"]
    print(f"\n{'Режим':<8}" + "".join(f"{c:>14}" for c in cols) + f"{'мс/запрос':>12}")
    for mode, r in results.items():
        print(f"{mode:<8}" + "".join(f"{r[c]:>14.2f}" for c in cols) + f"{r['latency_ms']:>12.0f}")
    for mode, r in results.items():
        print(f"  {mode}: не найден урок в top-{K} для {len(r['missed_question_ids'])}/{n} вопросов: {r['missed_question_ids']}")
    print(f"\n{'Срез (Lesson Hit@5 / MRR@10)':<34}{'n':>4}" + "".join(f"{m:>16}" for m in results))
    for key, first in next(iter(results.values()))["groups"].items():
        cells = "".join(f"{r['groups'][key]['lesson_hit@5']:>9.2f} / {r['groups'][key]['mrr@10']:.2f}" for r in results.values())
        print(f"{key:<34}{first['n']:>4}{cells}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--gold", default="gold_set_50.json", help="JSON Gold Set (например, gold_set_hard.json)")
    parser.add_argument("--out", help="файл результатов (по умолчанию retrieval_eval.json / retrieval_eval_<набор>.json)")
    args = parser.parse_args()
    stem = os.path.splitext(os.path.basename(args.gold))[0]
    out = args.out or ("retrieval_eval.json" if stem == "gold_set_50" else f"retrieval_eval_{stem.removeprefix('gold_set_')}.json")

    client = get_client()
    dense, sparse = BGEEmbeddings(), FastEmbedSparse(model_name=SPARSE_MODEL, cache_dir=CACHE_DIR)
    stores = {
        mode.name: get_vector_store(mode, client=client, embedding=dense, sparse_embedding=sparse)
        for mode in (RetrievalMode.DENSE, RetrievalMode.SPARSE, RetrievalMode.HYBRID)
    }

    if not args.eval_only:
        print("=" * 60 + "\n  Состояние Qdrant\n" + "=" * 60)
        show_collection(client)
        print("\n" + "=" * 60 + "\n  Демо-запросы (HYBRID, top-5)\n" + "=" * 60)
        show_demo(stores["HYBRID"])

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)
    print("\n" + "=" * 60 + f"\n  Оценка ретривера на {args.gold} ({len(gold)} вопросов, top-{K})\n" + "=" * 60)
    results = evaluate(stores, gold)
    print_eval(results, len(gold))

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {out}")


if __name__ == "__main__":
    main()
