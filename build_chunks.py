"""Чанкинг датасета NASA LLIS для индексации в Qdrant (LangChain).

Стратегия:
1. Деление по смысловым секциям урока (Abstract, Driving Event, Lesson Learned, Recommendation),
   чтобы один чанк не смешивал «что произошло» и «что рекомендовано».
2. Длинные секции дополнительно режутся RecursiveCharacterTextSplitter
   (по абзацам -> строкам -> предложениям) с перекрытием.
   Заглушки («See report», «[D]») и обрывки без слов отбрасываются, куски короче MIN_CHUNK
   приклеиваются к соседнему чанку той же секции.
3. В начало каждого чанка добавляется контекстный заголовок (номер и название урока, секция),
   чтобы фрагмент был понятен поиску и LLM без остального документа.
4. Каждому чанку присваивается детерминированный ID «<lesson_id>:<section>:<index>»,
   поэтому повторная индексация в Qdrant не создает дублей.

Установка:  pip install langchain-core langchain-text-splitters
Запуск:     python build_chunks.py                 # llis_full.json -> llis_chunks.jsonl
            python build_chunks.py <input> <output>
"""

import json
import re
import statistics
import sys
from collections import Counter
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================== КОНФИГУРАЦИЯ ====================
INPUT_PATH = "llis_full.json"
OUTPUT_PATH = "llis_chunks.jsonl"
CHUNK_SIZE = 1200       # Символов на чанк (~300 токенов; лимит bge-base — 512 токенов)
CHUNK_OVERLAP = 150     # Перекрытие соседних чанков внутри секции
MIN_CHUNK = 200         # Более короткие куски приклеиваются к соседнему чанку той же секции

# Заглушки, которые в LLIS стоят вместо текста секции
PLACEHOLDERS = {"unavailable", "not available", "[d]"}
# Короткие отсылки вместо текста: «See attached PDF», «(see Lessons above)», «This is only a test»
PLACEHOLDER_RE = re.compile(r"^\(?see\b|^same as\b|^no description|^this is (only )?a test", re.IGNORECASE)

# Ключ секции в датасете -> человекочитаемое название в чанке
SECTIONS = {
    "abstract": "Abstract",
    "driving_event": "Driving Event",
    "lesson_learned": "Lesson Learned",
    "recommendation": "Recommendation",
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    keep_separator="end",
)


def is_junk(text: str) -> bool:
    """Фрагмент без смысловой нагрузки: заглушка, номер пункта, пустая шапка таблицы."""
    t = text.strip().lower().rstrip(".")
    if t in PLACEHOLDERS or (len(t) < 60 and PLACEHOLDER_RE.match(t)):
        return True
    return len(re.findall(r"[a-z]{2,}", t)) < 2


def split_section(text: str) -> List[str]:
    """Режет секцию на части, выбрасывает мусор и приклеивает слишком короткие куски к соседям."""
    parts = [p.strip() for p in splitter.split_text(text) if not is_junk(p)]
    merged: List[str] = []
    for p in parts:
        if merged and (len(p) < MIN_CHUNK or len(merged[-1]) < MIN_CHUNK):
            merged[-1] = f"{merged[-1]}\n\n{p}"
        else:
            merged.append(p)
    return merged


def lesson_to_chunks(rec: Dict[str, Any]) -> List[Document]:
    """Разбивает один урок на LangChain Document-чанки с метаданными."""
    meta = rec["metadata"]
    base_meta = {
        "lesson_id": rec["id"],
        "title": rec["title"],
        "date": rec["date"],
        "year": rec["year"],
        "org_code": rec["org_code"] or "NASA",
        "org_name": rec["organization"],
        "mission_directorate": rec["mission_directorate"],
        "categories": rec["categories_list"],
        "url": rec["url"],
        "lesson_tokens": meta["estimated_tokens"],
    }

    docs: List[Document] = []
    for key, name in SECTIONS.items():
        text = (rec["sections"].get(key) or "").strip()
        if not text:
            continue
        parts = split_section(text)
        for idx, part in enumerate(parts):
            header = f"[NASA LLIS #{rec['id']}] {rec['title']}\nSection: {name}"
            if len(parts) > 1:
                header += f" (part {idx + 1}/{len(parts)})"
            docs.append(Document(
                id=f"{rec['id']}:{key}:{idx}",
                page_content=f"{header}\n\n{part}",
                metadata={
                    **base_meta,
                    "section": key,
                    "chunk_index": idx,
                    "section_chunks": len(parts),
                },
            ))
    return docs


def build_chunks(input_path: str = INPUT_PATH) -> List[Document]:
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    chunks: List[Document] = []
    for rec in records:
        chunks.extend(lesson_to_chunks(rec))
    return chunks


def save_chunks(chunks: List[Document], output_path: str = OUTPUT_PATH) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for d in chunks:
            row = {"id": d.id, "page_content": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_stats(chunks: List[Document]) -> None:
    lens = sorted(len(d.page_content) for d in chunks)
    lessons = Counter(d.metadata["lesson_id"] for d in chunks)
    sections = Counter(d.metadata["section"] for d in chunks)
    q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]

    print(f"Чанков всего        : {len(chunks)}")
    print(f"Уроков              : {len(lessons)}")
    print(f"Чанков на урок      : медиана {statistics.median(lessons.values()):.0f}, макс {max(lessons.values())}")
    print("По секциям          : " + ", ".join(f"{k}={v}" for k, v in sections.items()))
    print(f"Длина чанка (симв.) : мин {lens[0]} | 25% {q(.25)} | медиана {q(.5)} | 75% {q(.75)} | макс {lens[-1]}")
    bodies = [len(d.page_content.split("\n\n", 1)[1]) for d in chunks]
    print(f"Текст чанка < {MIN_CHUNK}   : {sum(1 for n in bodies if n < MIN_CHUNK)} (короткие секции целиком)")
    print(f"Уникальных ID       : {len({d.id for d in chunks})}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH

    print("=" * 60)
    print(f"  Чанкинг NASA LLIS: {input_path} -> {output_path}")
    print(f"  chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print("=" * 60)

    chunks = build_chunks(input_path)
    save_chunks(chunks, output_path)
    print_stats(chunks)

    print("\n=== Пример чанка ===")
    sample = chunks[0]
    print(f"id: {sample.id}")
    print(json.dumps(sample.metadata, ensure_ascii=False))
    print(sample.page_content[:600])


if __name__ == "__main__":
    main()
