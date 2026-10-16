"""Вспомогательный модуль для удобной загрузки подготовленных данных NASA LLIS в RAG.

Примеры использования:
1. Загрузка в виде списка Python-словарей:
   from rag_loader import load_clean_data
   docs = load_clean_data("llis_all_1000.json")

2. Загрузка напрямую в объекты LangChain Document (с сохранением всех метаданных):
   from rag_loader import to_langchain_documents
   langchain_docs = to_langchain_documents("llis_all_1000.json")

3. Быстрая проверка и статистика:
   python rag_loader.py
"""

import json
import os
from typing import Any, Dict, List, Optional


def load_clean_data(filepath: str = "llis_all_1000.json") -> List[Dict[str, Any]]:
    """Загружает подготовленный датасет из JSON или JSONL файла."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    if filepath.endswith(".jsonl"):
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


def to_langchain_documents(filepath: str = "llis_all_1000.json", use_sections: bool = False):
    """Преобразует записи в список LangChain Document объектов.
    
    Если библиотека langchain_core не установлена, возвращает легковесный аналог Document.
    
    Параметры:
    - filepath: путь к JSON/JSONL файлу
    - use_sections: если True, возвращает отдельные документы для каждой секции с метаданными секции;
                    если False (по умолчанию), возвращает цельный Markdown-документ урока.
    """
    try:
        from langchain_core.documents import Document
    except ImportError:
        # Резервный класс, полностью совместимый по интерфейсу с LangChain Document
        class Document:
            def __init__(self, page_content: str, metadata: dict):
                self.page_content = page_content
                self.metadata = metadata

            def __repr__(self):
                return f"Document(page_content='{self.page_content[:60]}...', metadata={self.metadata})"

    raw_records = load_clean_data(filepath)
    docs = []

    for r in raw_records:
        if not use_sections:
            # Цельный документ урока с метаданными
            docs.append(Document(page_content=r["full_text"], metadata=r["metadata"]))
        else:
            # Посекционный режим (каждая секция — отдельный документ)
            base_meta = dict(r["metadata"])
            for sec_name, sec_text in r.get("sections", {}).items():
                if sec_text:
                    sec_meta = dict(base_meta)
                    sec_meta["section"] = sec_name
                    # Добавляем префикс контекста к тексту секции
                    content = f"[{r['title']} | Section: {sec_name.upper()}]\n{sec_text}"
                    docs.append(Document(page_content=content, metadata=sec_meta))

    return docs


if __name__ == "__main__":
    print("=" * 60)
    print("  Проверка готовности данных для RAG")
    print("=" * 60)
    docs = to_langchain_documents("llis_all_1000.json")
    print(f"Успешно загружено документов: {len(docs)}")
    print(f"Пример метаданных первого документа:")
    for k, v in docs[0].metadata.items():
        print(f"  - {k}: {v}")
    print("\nПервые 300 символов page_content:")
    print("-" * 40)
    print(docs[0].page_content[:300] + "...")
    print("-" * 40)
    print("\nДанные полностью готовы к передаче в любой TextSplitter / VectorStore!")
