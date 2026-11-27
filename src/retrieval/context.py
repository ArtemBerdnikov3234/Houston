"""Отбор релевантного контекста для LLM из переранжированных фрагментов.

Шаги select_context():
1. Порог: фрагменты с rerank_score ниже min_rerank_score отбрасываются (но keep_at_least лучших
   остаются всегда, чтобы LLM могла сама сказать, что ответа нет).
2. Группировка по урокам (select_hits): идем по ранжированной выдаче; фрагмент добавляется в блок
   своего урока, пока в нем меньше max_per_lesson фрагментов; новый урок — пока уроков меньше final_k;
   всего не больше max_chunks фрагментов. Так в контексте разные прецеденты, но и лучший урок
   не теряет свои сильные фрагменты.
3. Small-to-big: найденная секция короче full_section_max_chars подставляется целиком (длинные
   секции разрезаны на чанки, и ответ часто лежит в соседнем куске); к блоку добавляются соседние
   секции урока из llis_full.json (по умолчанию Driving Event, Lesson Learned, Recommendation),
   обрезанные до expansion_max_chars, чтобы LLM видела и что случилось, и что NASA рекомендовала.
4. Бюджет: блоки добавляются по убыванию релевантности, пока не исчерпан context_token_budget.

Результат — список ContextBlock; format_context() превращает его в текст с метками [LLIS #id],
на которые LLM ссылается в ответе.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

from src.retrieval.config import SECTION_ORDER, SECTION_TITLES, RetrieverConfig
from src.retrieval.search import SearchHit
from src.settings import PROCESSED_DATA_DIR

CHARS_PER_TOKEN = 4.5   # для англоязычных чанков LLIS: медиана ~660 символов / 135 токенов bge
_HEADER_RE = re.compile(r"^\[NASA LLIS #[^\]]+\][^\n]*\nSection:[^\n]*\n+", re.S)


def estimate_tokens(text: str) -> int:
    """Приблизительное число токенов (без загрузки токенизатора)."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def strip_chunk_header(text: str) -> str:
    """Убирает служебный заголовок чанка "[NASA LLIS #id] Title\\nSection: ..." — он есть в метаданных блока."""
    return _HEADER_RE.sub("", text, count=1).strip()


# ==================== УРОКИ ====================

class LessonStore:
    """Полные уроки по lesson_id (секции + метаданные) для расширения контекста."""

    def __init__(self, lessons: Iterable[dict]):
        self.lessons: Dict[str, dict] = {str(r["id"]): r for r in lessons}

    @classmethod
    def from_file(cls, path: str = str(PROCESSED_DATA_DIR / "llis_full.json")) -> "LessonStore":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def get(self, lesson_id: str) -> Optional[dict]:
        return self.lessons.get(str(lesson_id))

    def section(self, lesson_id: str, name: str) -> str:
        lesson = self.get(lesson_id)
        return ((lesson or {}).get("sections") or {}).get(name) or ""

    def programs(self) -> List[str]:
        return sorted({r.get("program", "") for r in self.lessons.values() if r.get("program")})


@lru_cache(maxsize=1)
def default_lesson_store() -> LessonStore:
    return LessonStore.from_file()


# ==================== БЛОК КОНТЕКСТА ====================

@dataclass
class ContextBlock:
    """Один урок в контексте LLM: найденные фрагменты + добавленные соседние секции."""

    lesson_id: str
    title: str
    url: str
    year: Optional[int]
    org_code: str
    program: str
    score: float                                     # лучший score фрагментов урока
    rank: int                                        # позиция блока в контексте, с 1
    sections: Dict[str, str] = field(default_factory=dict)      # секция -> текст (в порядке SECTION_ORDER)
    matched_sections: List[str] = field(default_factory=list)   # секции, найденные поиском
    chunk_ids: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        parts = []
        for name in SECTION_ORDER:
            if name in self.sections:
                mark = "" if name in self.matched_sections else " (added for context)"
                parts.append(f"## {SECTION_TITLES[name]}{mark}\n{self.sections[name]}")
        return "\n\n".join(parts)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    def header(self) -> str:
        meta = ", ".join(str(x) for x in (self.org_code, self.year, self.program) if x)
        return f"[LLIS #{self.lesson_id}] {self.title}" + (f" ({meta})" if meta else "")

    def to_document(self):
        from langchain_core.documents import Document

        return Document(
            id=f"lesson:{self.lesson_id}",
            page_content=self.text,
            metadata={
                "lesson_id": self.lesson_id, "title": self.title, "url": self.url, "year": self.year,
                "org_code": self.org_code, "program": self.program, "score": self.score, "rank": self.rank,
                "matched_sections": list(self.matched_sections), "chunk_ids": list(self.chunk_ids),
                "source_label": f"LLIS #{self.lesson_id}",
            },
        )


def format_context(blocks: List[ContextBlock]) -> str:
    """Текст контекста для промпта: каждый урок с меткой [LLIS #id], заголовком, метаданными и URL."""
    out = []
    for b in blocks:
        out.append(f"{b.header()}\nURL: {b.url}\n\n{b.text}")
    return "\n\n---\n\n".join(out)


# ==================== ОТБОР ====================

def _section_of(hit: SearchHit) -> str:
    sections = hit.metadata.get("sections") or [hit.metadata.get("section", "")]
    return sections[0] if sections else ""


def select_hits(hits: List[SearchHit], final_k: int, max_per_lesson: int, max_chunks: int) -> List[SearchHit]:
    """Отбор фрагментов с группировкой по урокам (порядок выдачи сохраняется).

    Слитый чанк (один текст в нескольких уроках) засчитывается своему основному lesson_id.
    """
    per_lesson: Dict[str, int] = {}
    out = []
    for h in hits:
        if len(out) >= max_chunks:
            break
        n = per_lesson.get(h.lesson_id)
        if n is None:
            if len(per_lesson) >= final_k:
                continue
            n = 0
        if n < max_per_lesson:
            out.append(h)
            per_lesson[h.lesson_id] = n + 1
    return out


def apply_threshold(hits: List[SearchHit], min_score: float, keep_at_least: int) -> List[SearchHit]:
    """Отбрасывает фрагменты с rerank_score ниже порога (без реранкера порог не применяется)."""
    if not hits or hits[0].rerank_score is None:
        return hits
    kept = [h for h in hits if h.rerank_score >= min_score]
    return kept if len(kept) >= keep_at_least else hits[:keep_at_least]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("\n"))
    return (cut[: end + 1] if end > max_chars * 0.6 else cut).rstrip() + " [...]"


def build_blocks(hits: List[SearchHit], config: RetrieverConfig,
                 lessons: Optional[LessonStore] = None) -> List[ContextBlock]:
    """Группирует фрагменты по урокам и расширяет соседними секциями."""
    blocks: Dict[str, ContextBlock] = {}
    for h in hits:
        lid = h.lesson_id
        m = h.metadata
        block = blocks.get(lid)
        if block is None:
            block = blocks[lid] = ContextBlock(
                lesson_id=lid, title=m.get("title", ""), url=m.get("url", f"https://llis.nasa.gov/lesson/{lid}"),
                year=m.get("year"), org_code=m.get("org_code", ""), program=m.get("program", ""),
                score=h.final_score, rank=len(blocks) + 1,
            )
        section = _section_of(h)
        body = strip_chunk_header(h.text)
        if section in block.sections and body not in block.sections[section]:
            block.sections[section] += "\n\n" + body      # второй фрагмент той же секции
        elif section not in block.sections:
            block.sections[section] = body
        if section not in block.matched_sections:
            block.matched_sections.append(section)
        block.chunk_ids.append(h.chunk_id)

    if lessons is not None:
        for block in blocks.values():
            for name in list(block.sections):           # найденная короткая секция — целиком
                full = lessons.section(block.lesson_id, name)
                if full and len(full) <= config.full_section_max_chars and len(full) > len(block.sections[name]):
                    block.sections[name] = full
            if not config.expand_sections:
                continue
            for name in config.expansion_sections:
                if name in block.sections:
                    continue
                text = lessons.section(block.lesson_id, name)
                if text:
                    block.sections[name] = _truncate(text, config.expansion_max_chars)

    for block in blocks.values():   # секции в естественном порядке урока
        block.sections = {k: block.sections[k] for k in SECTION_ORDER if k in block.sections}
    return list(blocks.values())


def fit_budget(blocks: List[ContextBlock], budget_tokens: int) -> List[ContextBlock]:
    """Оставляет блоки по порядку, пока помещаются в бюджет; первый блок остается всегда.

    Если блок не помещается целиком, из него сначала убираются добавленные (не найденные) секции.
    """
    out, used = [], 0
    for block in blocks:
        cost = block.tokens
        if used + cost > budget_tokens and out:
            trimmed = ContextBlock(**{**block.__dict__,
                                      "sections": {k: v for k, v in block.sections.items() if k in block.matched_sections}})
            if used + trimmed.tokens > budget_tokens:
                continue
            block, cost = trimmed, trimmed.tokens
        out.append(block)
        used += cost
    for i, b in enumerate(out, 1):
        b.rank = i
    return out


def select_context(hits: List[SearchHit], config: RetrieverConfig,
                   lessons: Optional[LessonStore] = None) -> List[ContextBlock]:
    """Полный отбор: порог -> группировка по урокам -> small-to-big -> бюджет."""
    kept = apply_threshold(hits, config.min_rerank_score, config.keep_at_least)
    kept = select_hits(kept, config.final_k, config.max_per_lesson, config.max_chunks)
    blocks = build_blocks(kept, config, lessons)
    return fit_budget(blocks, config.context_token_budget)
