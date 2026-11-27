"""Настройки поиска Houston: все параметры ретривера в одном месте.

Значения по умолчанию подобраны для коллекции nasa_llis (12 471 чанк) и CPU. Любой параметр можно
переопределить при создании объекта или через переменные окружения HOUSTON_* (см. RetrieverConfig.from_env).

Схема поиска и роль параметров:

    запрос -> обработка запроса (query_processing)
           -> dense top-prefetch_k + BM25 top-prefetch_k -> слияние (fusion) -> candidate_k кандидатов
           -> реранкер (use_reranker) -> отбор контекста: порог, final_k уроков, max_per_lesson фрагментов
              на урок, max_chunks всего -> короткие секции целиком + соседние секции, бюджет context_token_budget
"""

import os
from dataclasses import dataclass, fields, replace
from typing import Tuple

# Модель реранкера: Cross-Encoder, ONNX через FastEmbed (тот же стек, что и для эмбеддингов)
RERANK_MODEL = "BAAI/bge-reranker-base"

# Порядок секций урока при сборке контекста: от описания события к выводам и проверке
SECTION_ORDER = ("abstract", "driving_event", "lesson_learned", "recommendation", "evidence", "references")
SECTION_TITLES = {
    "abstract": "Abstract",
    "driving_event": "Driving Event",
    "lesson_learned": "Lesson Learned",
    "recommendation": "Recommendation",
    "evidence": "Evidence of Recurrence Control Effectiveness",
    "references": "Related Documents and Policy",
}


@dataclass(frozen=True)
class RetrieverConfig:
    # --- Режим первичного поиска ---
    mode: str = "hybrid"            # "dense" | "sparse" | "hybrid"
    fusion: str = "rrf"             # слияние для hybrid: "rrf" (по рангам) | "dbsf" (по нормированным score)
    prefetch_k: int = 50            # сколько кандидатов берет каждый из поисков (dense и BM25) перед слиянием
    candidate_k: int = 30           # сколько кандидатов после слияния идет дальше (в реранкер)

    # --- Обработка запроса ---
    expand_abbreviations: bool = True   # KSC -> Kennedy Space Center и обратно (только для BM25-части)
    auto_filters: bool = True           # извлекать фильтры (центр, годы) из текста вопроса
    auto_filter_min_hits: int = 3       # если с авто-фильтром нашлось меньше — повторить поиск без него
    normalize_programs: bool = True     # фильтр program расширяется на все варианты написания программы

    # --- Переранжирование ---
    use_reranker: bool = True
    rerank_model: str = RERANK_MODEL
    rerank_batch_size: int = 16

    # --- Отбор контекста ---
    final_k: int = 5                    # сколько уроков (блоков) передается в LLM
    max_per_lesson: int = 3             # сколько фрагментов одного урока можно взять в его блок
    max_chunks: int = 10                # общий лимит найденных фрагментов в контексте
    min_rerank_score: float = -6.0      # порог логита реранкера: ниже — фрагмент считается нерелевантным
    keep_at_least: int = 1              # сколько фрагментов оставить, даже если все ниже порога
    full_section_max_chars: int = 2400  # найденная секция короче этого подставляется целиком, а не кусками
    expand_sections: bool = True        # добавлять к блоку соседние секции урока
    expansion_sections: Tuple[str, ...] = ("driving_event", "lesson_learned", "recommendation")
    expansion_max_chars: int = 1200     # лимит на одну добавленную секцию (символов)
    context_token_budget: int = 6000    # общий бюджет контекста для LLM (приблизительно, в токенах)

    def with_(self, **kwargs) -> "RetrieverConfig":
        """Копия конфигурации с измененными полями: cfg.with_(use_reranker=False)."""
        return replace(self, **kwargs)

    @classmethod
    def from_env(cls, **overrides) -> "RetrieverConfig":
        """Читает переменные окружения HOUSTON_<ПОЛЕ> (например, HOUSTON_FINAL_K=4)."""
        values = {}
        for f in fields(cls):
            raw = os.getenv(f"HOUSTON_{f.name.upper()}")
            if raw is None:
                continue
            if f.type in (bool, "bool"):
                values[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif f.type in (int, "int"):
                values[f.name] = int(raw)
            elif f.type in (float, "float"):
                values[f.name] = float(raw)
            elif "Tuple" in str(f.type):
                values[f.name] = tuple(s.strip() for s in raw.split(",") if s.strip())
            else:
                values[f.name] = raw
        values.update(overrides)
        return cls(**values)

    def __post_init__(self):
        if self.mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(f"mode: ожидается dense/sparse/hybrid, получено {self.mode!r}")
        if self.fusion not in ("rrf", "dbsf"):
            raise ValueError(f"fusion: ожидается rrf/dbsf, получено {self.fusion!r}")
        if self.candidate_k < self.final_k:
            raise ValueError("candidate_k не может быть меньше final_k")
        if min(self.final_k, self.max_per_lesson, self.max_chunks) < 1:
            raise ValueError("final_k, max_per_lesson и max_chunks должны быть >= 1")
        unknown = set(self.expansion_sections) - set(SECTION_ORDER)
        if unknown:
            raise ValueError(f"expansion_sections: неизвестные секции {sorted(unknown)}")


# Базовая линия = поиск, который был в проекте до этой работы: hybrid RRF top-10, без обработки запроса
# и реранкера; в контекст идут просто 5 первых фрагментов, как в простейшем RAG
BASELINE = RetrieverConfig(
    expand_abbreviations=False, auto_filters=False, normalize_programs=False,
    use_reranker=False, candidate_k=10, prefetch_k=10,
    final_k=5, max_per_lesson=10, max_chunks=5, full_section_max_chars=0, expand_sections=False,
    min_rerank_score=float("-inf"), context_token_budget=10**9,
)
DEFAULT = RetrieverConfig()
