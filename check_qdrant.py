"""Проверка готовности Qdrant и качества поиска по Gold Set.

1. Состояние коллекции: число точек, конфигурация векторов, payload-индексы.
2. Демонстрационные запросы (гибридный поиск, поиск с фильтром по метаданным).
3. Оценка ретривера на 50 вопросах Gold Set в трех режимах: DENSE, SPARSE (BM25), HYBRID (RRF).
   - Lesson Hit@k — нужный урок есть среди top-k чанков;
   - Quote Hit@k  — среди top-k есть именно тот чанк, который содержит эталонную цитату;
   - MRR@10       — средний обратный ранг первого чанка нужного урока.

Запуск:  python check_qdrant.py            # всё
         python check_qdrant.py --eval-only
Результат оценки также сохраняется в retrieval_eval.json.
"""

import argparse
import json
import sys
import time

from langchain_qdrant import FastEmbedSparse, RetrievalMode

from qdrant_store import (CACHE_DIR, COLLECTION, SPARSE_MODEL, BGEEmbeddings, field_filter,
                          get_client, get_vector_store)

K = 10
DEMO_QUERIES = [
    ("What caused the hydraulic pressure gauge failure at the launch equipment test facility?", None),
    ("lessons about battery thermal runaway", None),
    ("How should flight software be tested before launch?", ("org_code", "JPL")),
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


def evaluate(stores: dict, gold: list) -> dict:
    results = {}
    for mode, store in stores.items():
        lesson_hits = {1: 0, 5: 0, 10: 0}
        quote_hits = {1: 0, 5: 0, 10: 0}
        rr_sum = 0.0
        misses = []
        t0 = time.time()
        for g in gold:
            docs = store.similarity_search(g["question"], k=K)
            lesson_ranks = [i for i, d in enumerate(docs, 1) if d.metadata["lesson_id"] == g["lesson_id"]]
            quote_ranks = [i for i, d in enumerate(docs, 1)
                           if d.metadata["lesson_id"] == g["lesson_id"] and g["context_quote"] in d.page_content]
            for k in lesson_hits:
                lesson_hits[k] += bool(lesson_ranks and lesson_ranks[0] <= k)
                quote_hits[k] += bool(quote_ranks and quote_ranks[0] <= k)
            rr_sum += 1 / lesson_ranks[0] if lesson_ranks else 0
            if not lesson_ranks:
                misses.append(g["id"])
        n = len(gold)
        results[mode] = {
            **{f"lesson_hit@{k}": v / n for k, v in lesson_hits.items()},
            **{f"quote_hit@{k}": v / n for k, v in quote_hits.items()},
            "mrr@10": rr_sum / n,
            "missed_question_ids": misses,
            "latency_ms": (time.time() - t0) / n * 1000,
        }
    return results


def print_eval(results: dict, n: int) -> None:
    cols = ["lesson_hit@1", "lesson_hit@5", "lesson_hit@10", "quote_hit@5", "quote_hit@10", "mrr@10"]
    print(f"\n{'Режим':<8}" + "".join(f"{c:>14}" for c in cols) + f"{'мс/запрос':>12}")
    for mode, r in results.items():
        print(f"{mode:<8}" + "".join(f"{r[c]:>14.2f}" for c in cols) + f"{r['latency_ms']:>12.0f}")
    for mode, r in results.items():
        print(f"  {mode}: не найден урок в top-{K} для {len(r['missed_question_ids'])}/{n} вопросов: {r['missed_question_ids']}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--gold", default="gold_set_50.json")
    args = parser.parse_args()

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
    print("\n" + "=" * 60 + f"\n  Оценка ретривера на Gold Set ({len(gold)} вопросов, top-{K})\n" + "=" * 60)
    results = evaluate(stores, gold)
    print_eval(results, len(gold))

    with open("retrieval_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nРезультаты сохранены в retrieval_eval.json")


if __name__ == "__main__":
    main()
