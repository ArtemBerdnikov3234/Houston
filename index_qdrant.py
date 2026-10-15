"""Индексация чанков NASA LLIS в Qdrant (dense bge-base-en-v1.5 + sparse BM25).

Требует запущенный Qdrant:  docker compose up -d
Запуск:
    python index_qdrant.py              # загрузить llis_chunks.jsonl (идемпотентно, upsert)
    python index_qdrant.py --recreate   # удалить коллекцию и проиндексировать заново
    python index_qdrant.py --limit 200  # быстрый пробный прогон на первых N чанках
"""

import argparse
import sys
import time

from qdrant_store import COLLECTION, QDRANT_URL, create_collection, get_client, get_vector_store, point_id
from rag_loader import load_chunks

BATCH_SIZE = 128


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", default="llis_chunks.jsonl")
    parser.add_argument("--recreate", action="store_true", help="пересоздать коллекцию с нуля")
    parser.add_argument("--limit", type=int, default=None, help="проиндексировать только первые N чанков")
    args = parser.parse_args()

    # Сортировка по длине уменьшает паддинг в батчах эмбеддера (~1.6x быстрее на CPU)
    docs = sorted(load_chunks(args.chunks)[: args.limit], key=lambda d: len(d.page_content))
    for d in docs:
        d.metadata["chunk_id"] = d.id

    print("=" * 60)
    print(f"  Индексация в Qdrant: {QDRANT_URL} / коллекция '{COLLECTION}'")
    print(f"  Чанков к загрузке: {len(docs)}")
    print("=" * 60)

    client = get_client()
    create_collection(client, recreate=args.recreate)
    print("Загрузка моделей эмбеддингов (при первом запуске скачиваются ~0.5 ГБ)...")
    store = get_vector_store(client=client)

    t0 = time.time()
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start:start + BATCH_SIZE]
        store.add_documents(batch, ids=[point_id(d.id) for d in batch], batch_size=BATCH_SIZE)
        done = start + len(batch)
        elapsed = time.time() - t0
        eta = elapsed / done * (len(docs) - done)
        print(f"  -> {done}/{len(docs)}  ({elapsed / 60:.1f} мин, осталось ~{eta / 60:.1f} мин)", flush=True)

    info = client.get_collection(COLLECTION)
    print("=" * 60)
    print(f"ГОТОВО за {(time.time() - t0) / 60:.1f} мин. Точек в коллекции: {info.points_count}, статус: {info.status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
