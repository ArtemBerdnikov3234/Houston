"""Общая конфигурация векторного хранилища Qdrant для проекта Houston (LangChain + FastEmbed).

Коллекция хранит для каждого чанка два вектора:
- "dense"  — семантический эмбеддинг BAAI/bge-base-en-v1.5 (768 dim, cosine);
- "sparse" — BM25 (Qdrant/bm25, IDF считается на стороне Qdrant) для точных терминов,
             аббревиатур и номеров стандартов.
Гибридный поиск объединяет обе выдачи через Reciprocal Rank Fusion (RRF).

Использование:
    from qdrant_store import get_vector_store
    store = get_vector_store()                       # гибридный поиск
    store.similarity_search("hydraulic gauge failure", k=5,
                            filter=field_filter("sections", "recommendation"))
"""

import os
import uuid
from typing import List, Optional

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

# ==================== КОНФИГУРАЦИЯ ====================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "nasa_llis")
DENSE_MODEL = "BAAI/bge-base-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_DIM = 768
DENSE_NAME = "dense"
# Модели FastEmbed кэшируются в проекте, а не во временной папке ОС
CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fastembed_cache"))
SPARSE_NAME = "sparse"

# Инструкция BGE v1.5 для коротких запросов к длинным фрагментам (к документам не добавляется)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Пространство имен для детерминированных UUID точек из ID чанков ("34202:abstract:0")
ID_NAMESPACE = uuid.UUID("7c1e4a52-3f0b-4d1a-9a57-6f2b8c0d9e11")

# Поля метаданных, по которым строятся payload-индексы для фильтрации
PAYLOAD_INDEXES = {
    "metadata.lesson_id": models.PayloadSchemaType.KEYWORD,
    "metadata.lesson_ids": models.PayloadSchemaType.KEYWORD,   # все уроки, где встречается текст чанка
    "metadata.sections": models.PayloadSchemaType.KEYWORD,     # все секции, где встречается текст чанка
    "metadata.program": models.PayloadSchemaType.KEYWORD,
    "metadata.keywords": models.PayloadSchemaType.KEYWORD,
    "metadata.org_code": models.PayloadSchemaType.KEYWORD,
    "metadata.section": models.PayloadSchemaType.KEYWORD,
    "metadata.categories": models.PayloadSchemaType.KEYWORD,
    "metadata.year": models.PayloadSchemaType.INTEGER,
}


class BGEEmbeddings(Embeddings):
    """LangChain-обертка над FastEmbed (ONNX, CPU) для BAAI/bge-base-en-v1.5."""

    def __init__(self, model_name: str = DENSE_MODEL, batch_size: int = 32):
        self.model = TextEmbedding(model_name=model_name, cache_dir=CACHE_DIR)
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self.model.embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self.model.embed([BGE_QUERY_PREFIX + text]))).tolist()


def point_id(chunk_id: str) -> str:
    """Детерминированный UUID точки Qdrant по ID чанка (повторная загрузка не дублирует данные)."""
    return str(uuid.uuid5(ID_NAMESPACE, chunk_id))


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=60)


def create_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Создает коллекцию с dense + sparse векторами и payload-индексами."""
    if client.collection_exists(COLLECTION):
        if not recreate:
            return
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DENSE_NAME: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )
    for field, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)


def get_vector_store(
    mode: RetrievalMode = RetrievalMode.HYBRID,
    client: Optional[QdrantClient] = None,
    embedding: Optional[Embeddings] = None,
    sparse_embedding: Optional[FastEmbedSparse] = None,
) -> QdrantVectorStore:
    """LangChain VectorStore поверх существующей коллекции (режимы DENSE / SPARSE / HYBRID)."""
    return QdrantVectorStore(
        client=client or get_client(),
        collection_name=COLLECTION,
        embedding=embedding or BGEEmbeddings(),
        sparse_embedding=sparse_embedding or FastEmbedSparse(model_name=SPARSE_MODEL, cache_dir=CACHE_DIR),
        retrieval_mode=mode,
        vector_name=DENSE_NAME,
        sparse_vector_name=SPARSE_NAME,
    )


def field_filter(field: str, value) -> models.Filter:
    """Фильтр по полю метаданных, например field_filter("org_code", "JPL").

    Для секций и уроков используйте списочные поля "sections" и "lesson_ids": одинаковый текст
    хранится одним чанком, и в них перечислены все его вхождения (поле "section" — только первое).
    """
    return models.Filter(must=[
        models.FieldCondition(key=f"metadata.{field}", match=models.MatchValue(value=value)),
    ])
