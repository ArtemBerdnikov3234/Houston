"""REST API проекта Houston (FastAPI).

Запуск из корня репозитория:
    uvicorn src.api.main:app --reload --port 8000
Документация и проба запросов: http://localhost:8000/docs

Эндпоинты:
    GET  /health  — доступность Qdrant и подключена ли RAG-цепочка
    POST /ask     — вопрос -> ответ (если есть цепочка) + найденные чанки и источники
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api import pipeline

log = logging.getLogger("houston.api")


class Filters(BaseModel):
    org_code: Optional[str] = Field(None, description="Центр NASA, например KSC, JPL, JSC")
    program: Optional[str] = Field(None, description="Программа, например Cassini")
    year: Optional[int] = Field(None, description="Год урока")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Вопрос на английском")
    k: int = Field(4, ge=1, le=10, description="Сколько чанков вернуть (в режиме только поиска)")
    filters: Optional[Filters] = None


class Context(BaseModel):
    text: str
    lesson_id: Optional[str] = None
    title: Optional[str] = None
    org_code: Optional[str] = None
    section: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None


class AskResponse(BaseModel):
    mode: str = Field(..., description="rag — ответ сгенерирован цепочкой; retrieval_only — только поиск")
    answer: Optional[str] = None
    contexts: List[Context]
    sources: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Прогрев: модели эмбеддингов загружаются один раз при старте, а не на первом запросе.
    try:
        pipeline.get_store()
        pipeline.get_chain()
    except Exception as exc:  # Qdrant может подняться позже API: не падаем, /health покажет проблему
        log.warning("Прогрев не удался: %s", exc)
    yield


app = FastAPI(title="Houston RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    from src.retrieval.qdrant_store import COLLECTION, get_client

    try:
        client = get_client()
        exists = client.collection_exists(COLLECTION)
        points = client.count(COLLECTION).count if exists else 0
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Qdrant недоступен: {exc}")
    return {
        "status": "ok" if exists and points else "empty",
        "collection": COLLECTION,
        "points": points,
        "chain_connected": pipeline.get_chain() is not None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    filters = req.filters.model_dump(exclude_none=True) if req.filters else None
    try:
        return pipeline.ask(req.question, k=req.k, filters=filters)
    except Exception as exc:
        log.exception("Ошибка при обработке вопроса")
        raise HTTPException(status_code=503, detail=f"Не удалось получить ответ: {exc}")