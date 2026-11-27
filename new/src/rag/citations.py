"""Ссылки на источники в ответе LLM и итоговая структура ответа.

LLM пишет ссылки в тексте как [LLIS #1234] (варианты [LLIS #1, #2], [LLIS #1; LLIS #2] тоже понимаются).
process_answer():
  1. находит все ссылки и приводит их к виду [LLIS #id] [LLIS #id];
  2. удаляет ссылки на уроки, которых не было в контексте (галлюцинации), и сохраняет их в
     invalid_citations — по этой метрике видно, выдумывает ли модель источники;
  3. собирает список источников из метаданных найденных уроков (название, URL, центр, год), а не
     из текста модели, поэтому URL всегда настоящие;
  4. распознает отказ (фраза NOT_FOUND_ANSWER из промпта).
"""

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.rag.prompts import NOT_FOUND_ANSWER

_GROUP_RE = re.compile(r"\[([^\[\]]*?LLIS[^\[\]]*?)\]", re.I)
_ID_RE = re.compile(r"(?:#\s*|LLIS\s+)(\d{1,7})", re.I)   # "#1234" или "LLIS 1234", но не "page 3"


class Source(BaseModel):
    lesson_id: str
    title: str = ""
    url: str = ""
    org_code: str = ""
    year: Optional[int] = None
    program: str = ""
    matched_sections: List[str] = Field(default_factory=list)
    retrieval_rank: Optional[int] = None
    score: Optional[float] = None
    cited: bool = False

    @property
    def label(self) -> str:
        return f"LLIS #{self.lesson_id}"


class RAGResponse(BaseModel):
    """Ответ RAG-цепочки: текст с проверенными ссылками + источники + служебная информация."""

    question: str
    standalone_question: str
    answer: str
    found: bool                                         # False — модель (или код) ответила, что ответа нет
    sources: List[Source] = Field(default_factory=list)            # процитированные уроки, по порядку ссылок
    context_sources: List[Source] = Field(default_factory=list)    # все уроки, переданные в LLM
    invalid_citations: List[str] = Field(default_factory=list)     # ссылки на уроки вне контекста (удалены)
    model: str = ""
    usage: Dict[str, int] = Field(default_factory=dict)
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    context_text: str = Field(default="", exclude=True)            # текст контекста (для Ragas и отладки)

    @property
    def cited_ids(self) -> List[str]:
        return [s.lesson_id for s in self.sources]

    def sources_markdown(self) -> str:
        if not self.sources:
            return ""
        lines = ["**Sources:**"]
        for i, s in enumerate(self.sources, 1):
            meta = ", ".join(str(x) for x in (s.org_code, s.year) if x)
            lines.append(f"{i}. [{s.label}] {s.title}" + (f" ({meta})" if meta else "") + (f" — {s.url}" if s.url else ""))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        src = self.sources_markdown()
        return self.answer + (f"\n\n{src}" if src else "")


# ==================== РАЗБОР ССЫЛОК ====================

def extract_citations(text: str) -> List[str]:
    """ID уроков в порядке первого упоминания (без повторов)."""
    seen: List[str] = []
    for group in _GROUP_RE.findall(text):
        for lid in _ID_RE.findall(group):
            if lid not in seen:
                seen.append(lid)
    return seen


def clean_citations(text: str, allowed: set) -> tuple:
    """Нормализует ссылки и удаляет ссылки на уроки вне allowed. Возвращает (текст, неверные id)."""
    invalid: List[str] = []

    def repl(m: re.Match) -> str:
        ids = _ID_RE.findall(m.group(1))
        good = [i for i in ids if i in allowed]
        invalid.extend(i for i in ids if i not in allowed and i not in invalid)
        return " ".join(f"[LLIS #{i}]" for i in dict.fromkeys(good))

    cleaned = _GROUP_RE.sub(repl, text)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)        # пробел перед точкой после удаленной ссылки
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), invalid


def is_refusal(text: str) -> bool:
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    return norm(NOT_FOUND_ANSWER) in norm(text)


def source_from_document(doc, cited: bool = False) -> Source:
    m = doc.metadata
    return Source(lesson_id=str(m.get("lesson_id", "")), title=m.get("title", ""), url=m.get("url", ""),
                  org_code=m.get("org_code", ""), year=m.get("year"), program=m.get("program", ""),
                  matched_sections=list(m.get("matched_sections") or []), retrieval_rank=m.get("rank"),
                  score=m.get("score"), cited=cited)


def process_answer(raw: str, documents: list) -> dict:
    """Проверка ссылок в ответе. documents — Document'ы контекста (по одному на урок)."""
    by_id = {str(d.metadata.get("lesson_id")): d for d in documents}
    answer, invalid = clean_citations(raw or "", set(by_id))
    cited = extract_citations(answer)
    refusal = is_refusal(answer)
    sources = [source_from_document(by_id[i], cited=True) for i in cited if i in by_id]
    context_sources = [source_from_document(d, cited=str(d.metadata.get("lesson_id")) in cited) for d in documents]
    return {"answer": answer, "found": not refusal, "sources": [] if refusal else sources,
            "context_sources": context_sources, "invalid_citations": invalid}
