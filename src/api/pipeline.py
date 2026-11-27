"""Логика ответа на вопрос.

Если в проекте появилась RAG-цепочка (по умолчанию модуль `analytics.rag_chain`, объект `rag_chain`),
используется она. Пока её нет, API работает в режиме «только поиск»: возвращает найденные чанки
без сгенерированного ответа.

Контракт цепочки (тот же, что в домашнем задании по Ragas):
    result = rag_chain.invoke({"question": "..."})
    result["answer"]    -> сообщение LangChain (с .content) или строка
    result["contexts"]  -> список найденных чанков (строки или Document)
    result["sources"]   -> список словарей {"path": ...} или строк

Другой модуль или имя можно задать переменными окружения RAG_CHAIN_MODULE и RAG_CHAIN_ATTR.
"""

import importlib
import os
from functools import lru_cache
from typing import Optional

CHAIN_MODULE = os.getenv("RAG_CHAIN_MODULE", "analytics.rag_chain")
CHAIN_ATTR = os.getenv("RAG_CHAIN_ATTR", "rag_chain")

# Поля метаданных, по которым разрешена фильтрация (есть payload-индексы в Qdrant)
ALLOWED_FILTERS = ("org_code", "program", "year")


@lru_cache(maxsize=1)
def get_store():
    """Гибридное хранилище Qdrant (создаётся один раз: загрузка моделей занимает время)."""
    from src.retrieval.qdrant_store import get_vector_store

    return get_vector_store()


@lru_cache(maxsize=1)
def get_chain():
    """RAG-цепочка или None, если модуль цепочки ещё не написан."""
    try:
        module = importlib.import_module(CHAIN_MODULE)
    except ModuleNotFoundError as exc:
        # Нет именно модуля цепочки -> режим «только поиск». Нет чего-то внутри неё -> настоящая ошибка.
        if exc.name and CHAIN_MODULE.startswith(exc.name):
            return None
        raise
    return getattr(module, CHAIN_ATTR)


def build_filter(filters: Optional[dict]):
    """{"org_code": "KSC"} -> фильтр Qdrant по metadata (как в check_qdrant.py). Пустой -> None."""
    conds = {k: v for k, v in (filters or {}).items() if k in ALLOWED_FILTERS and v not in (None, "")}
    if not conds:
        return None
    from qdrant_client import models

    return models.Filter(must=[
        models.FieldCondition(key=f"metadata.{field}", match=models.MatchValue(value=value))
        for field, value in conds.items()
    ])


def doc_to_context(doc) -> dict:
    """Document LangChain -> словарь для ответа API; строки из цепочки оборачиваются в {"text": ...}."""
    if not hasattr(doc, "page_content"):
        return {"text": str(doc)}
    m = doc.metadata or {}
    lesson_id = m.get("lesson_id")
    return {
        "text": doc.page_content,
        "lesson_id": str(lesson_id) if lesson_id is not None else None,
        "title": m.get("title"),
        "org_code": m.get("org_code"),
        "section": m.get("section"),
        "year": m.get("year"),
        "url": m.get("url"),
    }


def _unique(items: list) -> list:
    return list(dict.fromkeys(i for i in items if i))


def ask(question: str, k: int = 4, filters: Optional[dict] = None) -> dict:
    """Ответ на вопрос: {"mode", "answer", "contexts", "sources"}."""
    chain = get_chain()
    if chain is not None:
        # Цепочка принимает только вопрос: k и фильтры в ней заданы внутри.
        result = chain.invoke({"question": question})
        answer = result["answer"]
        answer = getattr(answer, "content", answer)
        contexts = [doc_to_context(c) for c in result.get("contexts", []) if c]
        sources = [s.get("path") if isinstance(s, dict) else s for s in result.get("sources", [])]
        return {"mode": "rag", "answer": answer, "contexts": contexts, "sources": _unique(sources)}

    docs = get_store().similarity_search(question, k=k, filter=build_filter(filters))
    contexts = [doc_to_context(d) for d in docs]
    return {
        "mode": "retrieval_only",
        "answer": None,
        "contexts": contexts,
        "sources": _unique(c["url"] for c in contexts),
    }
