"""Чанкинг датасета NASA LLIS для индексации в Qdrant (LangChain).

1. Деление по смысловым секциям урока (Abstract, Driving Event, Lesson Learned, Recommendation,
   Evidence, References), чтобы один чанк не смешивал «что произошло» и «что рекомендовано».
2. Длинные секции дополнительно режутся RecursiveCharacterTextSplitter
   (по абзацам -> строкам -> предложениям) с перекрытием.
   Обрывки без слов и заглушки отбрасываются, куски короче MIN_CHUNK приклеиваются
   к соседнему чанку той же секции.
3. Каждый чанк проверяется токенизатором модели эмбеддингов: длиннее MAX_TOKENS режется дальше,
   чтобы модель ничего не обрезала.
4. В начало каждого чанка добавляется контекстный заголовок (номер и название урока, секция),
   чтобы фрагмент был понятен поиску и LLM без остального документа.
5. Одинаковые фрагменты (повтор секции внутри урока, уроки-дубли в базе NASA) хранятся один раз:
   в метаданных чанка перечислены все уроки (lesson_ids) и секции (sections), где он встречается.
6. Каждому чанку присваивается детерминированный ID «<lesson_id>:<section>:<index>»,
   поэтому повторная индексация в Qdrant не создает дублей.

Установка:  pip install langchain-core langchain-text-splitters tokenizers
Запуск:     python build_chunks.py                 # llis_full.json -> llis_chunks.jsonl
            python build_chunks.py <input> <output>
"""

import glob
import json
import os
import re
import statistics
import sys
from collections import Counter
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tokenizers import Tokenizer

from llis_scraper import is_placeholder

# ==================== КОНФИГУРАЦИЯ ====================
INPUT_PATH = "llis_full.json"
OUTPUT_PATH = "llis_chunks.jsonl"
CHUNK_SIZE = 1200       # Символов на чанк (~300 токенов)
CHUNK_OVERLAP = 150     # Перекрытие соседних чанков внутри секции
MIN_CHUNK = 200         # Более короткие куски приклеиваются к соседнему чанку той же секции
MAX_TOKENS = 512        # Лимит контекста BAAI/bge-base-en-v1.5 (включая служебные токены)
CROSS_LESSON_MIN = 300  # Одинаковый текст разных уроков сливается, только если он не короче этого
TOKENIZER_MODEL = "BAAI/bge-base-en-v1.5"
CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fastembed_cache"))

# Ключ секции в датасете -> человекочитаемое название в чанке
SECTIONS = {
    "abstract": "Abstract",
    "driving_event": "Driving Event",
    "lesson_learned": "Lesson Learned",
    "recommendation": "Recommendation",
    "evidence": "Evidence of Recurrence Control",
    "references": "Related Documents and Policy",
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    keep_separator="end",
)


def load_tokenizer() -> Tokenizer:
    paths = [p for p in glob.glob(os.path.join(CACHE_DIR, "**", "tokenizer.json"), recursive=True) if "bge-base" in p]
    tok = Tokenizer.from_file(paths[0]) if paths else Tokenizer.from_pretrained(TOKENIZER_MODEL)
    tok.no_truncation()
    return tok


tokenizer = load_tokenizer()


def n_tokens(text: str) -> int:
    return len(tokenizer.encode(text).ids)


def is_junk(text: str) -> bool:
    """Фрагмент без смысловой нагрузки: заглушка, номер пункта, пустая шапка таблицы."""
    t = text.strip()
    return is_placeholder(t) or len(re.findall(r"[a-z]{2,}", t.lower())) < 2


def fit_tokens(part: str, header_tokens: int) -> List[str]:
    """Режет фрагмент пополам (по границам абзацев/фраз), пока он с заголовком не влезет в MAX_TOKENS."""
    if header_tokens + n_tokens(part) <= MAX_TOKENS:
        return [part]
    half = RecursiveCharacterTextSplitter(
        chunk_size=len(part) // 2 + 1, chunk_overlap=0,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""], keep_separator="end",
    )
    return [p for piece in half.split_text(part) for p in fit_tokens(piece.strip(), header_tokens)]


def split_section(text: str, header_tokens: int) -> List[str]:
    """Режет секцию на части, выбрасывает мусор и приклеивает слишком короткие куски к соседям."""
    parts = [p.strip() for p in splitter.split_text(text) if not is_junk(p)]
    merged: List[str] = []
    for p in parts:
        if merged and (len(p) < MIN_CHUNK or len(merged[-1]) < MIN_CHUNK):
            merged[-1] = f"{merged[-1]}\n\n{p}"
        else:
            merged.append(p)
    return [p for m in merged for p in fit_tokens(m, header_tokens)]


def lesson_to_chunks(rec: Dict[str, Any]) -> List[Document]:
    """Разбивает один урок на LangChain Document-чанки с метаданными."""
    meta = rec["metadata"]
    base_meta = {
        "lesson_id": rec["id"],
        "lesson_ids": [rec["id"]],
        "title": rec["title"],
        "title_derived": meta["title_derived"],
        "date": rec["date"],
        "year": rec["year"],
        "org_code": rec["org_code"] or "NASA",
        "org_name": rec["organization"],
        "mission_directorate": rec["mission_directorate"],
        "program": rec["program"],
        "program_phase": rec["program_phase"],
        "categories": rec["categories_list"],
        "keywords": rec["keywords"],
        "attachments": rec["attachments"],
        "url": rec["url"],
        "lesson_tokens": meta["estimated_tokens"],
    }

    title = rec["title"]
    docs: List[Document] = []
    for key, name in SECTIONS.items():
        text = (rec["sections"].get(key) or "").strip()
        if not text:
            continue
        # Запас на « (part NN/NN)» в заголовке
        header_tokens = n_tokens(f"[NASA LLIS #{rec['id']}] {title}\nSection: {name} (part 99/99)\n\n")
        parts = split_section(text, header_tokens)
        for idx, part in enumerate(parts):
            header = f"[NASA LLIS #{rec['id']}] {title}\nSection: {name}"
            if len(parts) > 1:
                header += f" (part {idx + 1}/{len(parts)})"
            docs.append(Document(
                id=f"{rec['id']}:{key}:{idx}",
                page_content=f"{header}\n\n{part}",
                metadata={
                    **base_meta,
                    "section": key,
                    "sections": [key],
                    "chunk_index": idx,
                    "section_chunks": len(parts),
                },
            ))
    return docs


def body_key(doc: Document) -> str:
    body = doc.page_content.split("\n\n", 1)[1]
    return re.sub(r"\s+", " ", body).strip().lower()


def deduplicate(chunks: List[Document]) -> int:
    """Сливает чанки с одинаковым текстом: остается первый, в его метаданные добавляются
    уроки и секции остальных. Внутри урока сливаются любые повторы, между уроками — только
    длинные (от CROSS_LESSON_MIN символов): короткие общие фразы («Paragraph 4.3 of JPL Rules»)
    без заголовка своего урока теряют смысл. Возвращает число удаленных копий."""
    first: Dict[Any, Document] = {}
    kept: List[Document] = []
    for d in chunks:
        body = body_key(d)
        k = body if len(body) >= CROSS_LESSON_MIN else (d.metadata["lesson_id"], body)
        if k in first:
            m = first[k].metadata
            for field, value in (("lesson_ids", d.metadata["lesson_id"]), ("sections", d.metadata["section"])):
                if value not in m[field]:
                    m[field].append(value)
        else:
            first[k] = d
            kept.append(d)
    removed = len(chunks) - len(kept)
    chunks[:] = kept
    return removed


def build_chunks(input_path: str = INPUT_PATH) -> List[Document]:
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    # Сначала оригиналы, потом уроки, помеченные как полные дубли, — чтобы при слиянии оставался оригинал
    records.sort(key=lambda r: r["metadata"].get("duplicate_of") is not None)
    chunks: List[Document] = []
    for rec in records:
        chunks.extend(lesson_to_chunks(rec))
    build_chunks.removed = deduplicate(chunks)
    return chunks


def save_chunks(chunks: List[Document], output_path: str = OUTPUT_PATH) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for d in chunks:
            row = {"id": d.id, "page_content": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_stats(chunks: List[Document]) -> None:
    lens = sorted(len(d.page_content) for d in chunks)
    toks = sorted(n_tokens(d.page_content) for d in chunks)
    lessons = Counter(lid for d in chunks for lid in d.metadata["lesson_ids"])
    sections = Counter(d.metadata["section"] for d in chunks)
    q = lambda arr, p: arr[min(len(arr) - 1, int(p * len(arr)))]

    print(f"Чанков всего        : {len(chunks)} (слито одинаковых копий: {getattr(build_chunks, 'removed', 0)})")
    print(f"Уроков              : {len(lessons)}")
    print(f"Чанков на урок      : медиана {statistics.median(lessons.values()):.0f}, макс {max(lessons.values())}")
    print("По секциям          : " + ", ".join(f"{k}={v}" for k, v in sections.items()))
    print(f"Длина чанка (симв.) : мин {lens[0]} | 25% {q(lens, .25)} | медиана {q(lens, .5)} | 75% {q(lens, .75)} | макс {lens[-1]}")
    print(f"Длина (токены bge)  : медиана {q(toks, .5)} | 95% {q(toks, .95)} | макс {toks[-1]} (лимит {MAX_TOKENS})")
    print(f"Чанков из >1 урока  : {sum(1 for d in chunks if len(d.metadata['lesson_ids']) > 1)}")
    print(f"Чанков из >1 секции : {sum(1 for d in chunks if len(d.metadata['sections']) > 1)}")
    print(f"Уникальных ID       : {len({d.id for d in chunks})}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH

    print("=" * 60)
    print(f"  Чанкинг NASA LLIS: {input_path} -> {output_path}")
    print(f"  chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}, max_tokens={MAX_TOKENS}")
    print("=" * 60)

    chunks = build_chunks(input_path)
    save_chunks(chunks, output_path)
    print_stats(chunks)

    print("\n=== Пример чанка ===")
    sample = chunks[0]
    print(f"id: {sample.id}")
    print(json.dumps(sample.metadata, ensure_ascii=False)[:600])
    print(sample.page_content[:600])


if __name__ == "__main__":
    main()
