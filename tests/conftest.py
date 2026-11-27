"""Общие фикстуры: небольшая in-memory коллекция Qdrant из реальных чанков и фейковые модели."""

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Houston.src.retrieval.config import RetrieverConfig  # noqa: E402
from src.retrieval.context import LessonStore  # noqa: E402
from src.retrieval.query_processing import ProgramNormalizer, get_abbreviations  # noqa: E402
from src.retrieval.retriever import HoustonRetriever  # noqa: E402
from src.retrieval.search import Searcher  # noqa: E402
from tests.fakes import FakeDense, FakeReranker, FakeSparse, build_collection  # noqa: E402

DATA = ROOT / "data"
COLLECTION = "test_llis"


@pytest.fixture(scope="session")
def gold_hard():
    return json.loads((DATA / "benchmarks" / "gold_set_hard.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def lessons():
    return LessonStore.from_file(str(DATA / "processed" / "llis_full.json"))


@pytest.fixture(scope="session")
def chunks(gold_hard):
    """Чанки уроков из Hard Gold Set + 400 случайных — достаточно для проверки логики."""
    rows = [json.loads(line) for line in (DATA / "processed" / "llis_chunks.jsonl").open(encoding="utf-8")]
    gold_lessons = {lid for g in gold_hard for lid in (g.get("relevant_lesson_ids") or [g["lesson_id"]])}
    keep = [r for r in rows if gold_lessons & set(r["metadata"]["lesson_ids"])]
    rest = [r for r in rows if not (gold_lessons & set(r["metadata"]["lesson_ids"]))]
    random.Random(0).shuffle(rest)
    return keep + rest[:400]


@pytest.fixture(scope="session")
def searcher(chunks):
    client = build_collection(chunks, collection=COLLECTION)
    return Searcher(client=client, embedding=FakeDense(), sparse_embedding=FakeSparse(), collection=COLLECTION)


@pytest.fixture()
def reranker():
    return FakeReranker()


@pytest.fixture()
def retriever(searcher, reranker, lessons):
    return HoustonRetriever(config=RetrieverConfig(), searcher=searcher, reranker=reranker, lessons=lessons,
                            programs=ProgramNormalizer(lessons.programs()), abbreviations=get_abbreviations())
