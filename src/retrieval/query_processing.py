"""Обработка поисковых запросов перед поиском.

Что делает process_query():
1. Нормализация текста: типографские кавычки и тире, лишние пробелы, невидимые символы.
2. Раскрытие аббревиатур NASA для BM25: "NSI" -> "NSI NASA Standard Initiator" и обратно
   ("Kennedy Space Center" -> + "KSC"). Словарь строится автоматически по самим урокам из шаблона
   "Полное Название (АББР)" и кэшируется в data/processed/abbreviations.json.
   Dense-запрос не меняется: bge понимает формулировку вопроса, а лишние слова сдвигают вектор.
3. Извлечение подсказок-фильтров из текста вопроса:
   - центр NASA:  "at Kennedy", "JPL", "Marshall Space Flight Center" -> org_code;
   - период:      "in the 1990s", "between 2000 and 2005", "since 2010" -> year;
   - номер урока: "lesson 5437", "LLIS #5437" -> lesson_ids.
   Это мягкие фильтры: ретривер добавляет отфильтрованную выдачу к обычной, а не заменяет ее
   (см. HoustonRetriever). Одиночный год ("the 1999 Mars orbiter") фильтром не считается:
   в вопросах это обычно год события, а не год публикации урока.
4. Нормализация программ: явный фильтр {"program": "Lunar Reconnaissance Orbiter"} расширяется
   на все варианты написания в базе ("... TWTA", "(LRO)" и т.п.).

Пересобрать словарь аббревиатур:  python -m src.retrieval.query_processing --rebuild
"""

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.settings import PROCESSED_DATA_DIR

ABBREVIATIONS_PATH = PROCESSED_DATA_DIR / "abbreviations.json"
LESSONS_PATH = PROCESSED_DATA_DIR / "llis_full.json"

# Служебные слова, которые не дают букву в аббревиатуре: "Failure Mode and Effects Analysis" -> FMEA
_CONNECTORS = {"of", "and", "for", "the", "on", "to", "in", "&", "at", "a", "an", "with"}
# Короткие аббревиатуры, совпадающие с обычными словами или слишком многозначные
_ABBR_BLOCKLIST = {"IT", "US", "OR", "AND", "AN", "AS", "AT", "BE", "IN", "IS", "NO", "ON", "TO", "UP", "WE",
                   "THE", "ALL", "NOT", "USE", "MAY", "CAN", "NASA", "ID", "OK", "PI"}

# ==================== ЦЕНТРЫ NASA ====================
# Шаблоны регистрозависимые для аббревиатур (ARC, HQ — обычные слова в нижнем регистре)
CENTER_PATTERNS: Dict[str, List[str]] = {
    "KSC": [r"\bKSC\b", r"\bKennedy\b"],
    "JPL": [r"\bJPL\b", r"\bJet Propulsion Lab(?:oratory)?\b"],
    "JSC": [r"\bJSC\b", r"\bJohnson Space Center\b"],
    "MSFC": [r"\bMSFC\b", r"\bMarshall Space Flight Center\b", r"\bat Marshall\b"],
    "GSFC": [r"\bGSFC\b", r"\bGoddard\b"],
    "ARC": [r"\bARC\b", r"\bAmes\b"],
    "GRC": [r"\bGRC\b", r"\bGlenn Research Center\b", r"\bat Glenn\b", r"\bLewis Research Center\b"],
    "LARC": [r"\bLaRC\b", r"\bLARC\b", r"\bLangley\b"],
    "AFRC": [r"\bAFRC\b", r"\bArmstrong Flight Research Center\b", r"\bDryden\b"],
    "SSC": [r"\bSSC\b", r"\bStennis\b"],
    "WSTF": [r"\bWSTF\b", r"\bWhite Sands\b"],
    "HQ": [r"\bNASA Headquarters\b", r"\bNASA HQ\b"],
    "NESC": [r"\bNESC\b", r"\bNASA Engineering and Safety Center\b"],
    "WFF": [r"\bWFF\b", r"\bWallops\b"],
}
_CENTER_RE = {code: [re.compile(p) for p in pats] for code, pats in CENTER_PATTERNS.items()}

_DECADE_RE = re.compile(r"\b(?:in\s+the\s+|during\s+the\s+|from\s+the\s+)?(?:(19|20)(\d)0|'(\d)0)s\b", re.I)
_RANGE_RE = re.compile(r"\bbetween\s+((?:19|20)\d\d)\s+and\s+((?:19|20)\d\d)\b", re.I)
_SINCE_RE = re.compile(r"\b(?:since|after)\s+((?:19|20)\d\d)\b", re.I)
_BEFORE_RE = re.compile(r"\bbefore\s+((?:19|20)\d\d)\b", re.I)
_LESSON_RE = re.compile(r"\b(?:LLIS\s*#?\s*|lesson\s+(?:number\s+|no\.?\s*|#\s*)?)(\d{2,6})\b", re.I)

_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                            "—": "-", " ": " ", "​": "", "﻿": ""})


@dataclass
class ProcessedQuery:
    """Результат обработки вопроса."""

    original: str
    text: str                                   # нормализованный вопрос (идет в dense и реранкер)
    sparse_text: str                            # вопрос + раскрытые аббревиатуры (идет в BM25)
    hint_filters: Dict[str, Any] = field(default_factory=dict)   # мягкие фильтры из текста вопроса
    expansions: List[Tuple[str, str]] = field(default_factory=list)  # (найдено, добавлено)

    def describe(self) -> str:
        parts = []
        if self.expansions:
            parts.append("аббревиатуры: " + ", ".join(f"{a} -> {b}" for a, b in self.expansions))
        if self.hint_filters:
            parts.append(f"подсказки-фильтры: {self.hint_filters}")
        return "; ".join(parts) or "без изменений"


# ==================== НОРМАЛИЗАЦИЯ ====================

def normalize_text(text: str) -> str:
    """Unicode NFKC, типографские символы -> ASCII, схлопывание пробелов."""
    text = unicodedata.normalize("NFKC", text or "").translate(_TRANSLATE)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return re.sub(r"\s+", " ", text).strip()


# ==================== АББРЕВИАТУРЫ ====================

_DEF_RE = re.compile(r"((?:[A-Za-z][\w\-/]*\s+){1,10}?)\(([A-Z][A-Za-z0-9&/\-]{1,9})\)")


def _match_definition(words: List[str], abbr: str) -> Optional[str]:
    """Находит в конце списка слов полное название, буквы которого дают аббревиатуру.

    Идем с конца: каждое значимое слово дает первую букву ("Mission" -> M), слово через дефис —
    несколько букв ("Electro-Static" -> ES), слитное составное слово — тоже несколько, если
    нужные буквы идут в нем по порядку ("Electrostatic" -> ES). Служебные слова пропускаются.
    """
    letters = [c for c in abbr if c.isupper()]
    if len(letters) < 2:
        return None
    idx, need = len(words) - 1, len(letters) - 1
    start = None
    while idx >= 0 and need >= 0:
        w = words[idx]
        if w.lower() in _CONNECTORS:
            idx -= 1
            continue
        parts = [p for p in re.split(r"[-/]", w) if p]
        initials = [p[0].upper() for p in parts]
        consumed = 0
        if len(initials) > 1 and letters[max(0, need - len(initials) + 1):need + 1] == initials:
            consumed = len(initials)
        elif w[0].upper() == letters[need]:
            consumed = 1
        else:
            for k in (2, 3):                   # слитное составное слово: Electrostatic -> E..S
                first = need - k + 1
                if first < 0 or w[0].upper() != letters[first]:
                    continue
                if re.search(".*".join(re.escape(c) for c in letters[first + 1:need + 1]), w[1:], re.I):
                    consumed = k
                    break
        if not consumed:
            return None
        need -= consumed
        start = idx
        idx -= 1
    if need >= 0 or start is None:
        return None
    phrase = " ".join(words[start:])
    return phrase if len(phrase) > len(abbr) + 2 else None


def _variant_key(phrase: str) -> str:
    """Ключ для объединения вариантов: "Failure Modes and Effects Analysis" ~ "failure mode and effect analysis"."""
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in phrase.lower().split())


def build_abbreviations(records: Iterable[dict], min_count: int = 2, min_share: float = 0.6) -> Dict[str, str]:
    """Словарь АББР -> полное название по шаблону "Полное Название (АББР)" в текстах уроков.

    Аббревиатура попадает в словарь, если встретилась не реже min_count раз и одно из названий
    составляет не меньше min_share всех определений (отсекает многозначные вроде SRB: Solid Rocket
    Booster / Standing Review Board — берется доминирующее значение, если оно явно преобладает).
    """
    counts: Dict[str, Counter] = defaultdict(Counter)     # АББР -> ключ варианта -> число
    surface: Dict[str, Counter] = defaultdict(Counter)    # ключ варианта -> написания
    for rec in records:
        for m in _DEF_RE.finditer(rec.get("full_text", "")):
            abbr = m.group(2)
            if abbr.upper() in _ABBR_BLOCKLIST or abbr.isdigit():
                continue
            phrase = _match_definition(m.group(1).split(), abbr)
            if phrase:
                phrase = phrase.lower().strip(" ,.;:")
                key = _variant_key(phrase)
                counts[abbr][key] += 1
                surface[key][phrase] += 1
    result = {}
    for abbr, c in counts.items():
        total = sum(c.values())
        key, n = c.most_common(1)[0]
        if total >= min_count and n / total >= min_share:
            result[abbr] = surface[key].most_common(1)[0][0]
    return dict(sorted(result.items()))


def load_lessons(path: Path = LESSONS_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_abbreviations(path: str = str(ABBREVIATIONS_PATH)) -> Dict[str, str]:
    """Словарь аббревиатур из кэша; если кэша нет — строится по llis_full.json и сохраняется."""
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    abbrs = build_abbreviations(load_lessons())
    p.write_text(json.dumps(abbrs, ensure_ascii=False, indent=1), encoding="utf-8")
    return abbrs


def expand_abbreviations(text: str, abbreviations: Dict[str, str]) -> Tuple[str, List[Tuple[str, str]]]:
    """Добавляет к запросу расшифровки найденных аббревиатур и аббревиатуры найденных названий.

    Исходный текст не меняется, добавки дописываются в конец — для BM25 важен только набор слов.
    """
    added: List[Tuple[str, str]] = []
    lower = text.lower()
    tokens = set(re.findall(r"\b[A-Z][A-Za-z0-9&/\-]{1,9}\b", text))
    for abbr in sorted(tokens):
        phrase = abbreviations.get(abbr)
        if phrase and phrase not in lower:
            added.append((abbr, phrase))
    for abbr, phrase in abbreviations.items():
        if len(phrase.split()) >= 2 and abbr not in tokens and re.search(rf"\b{re.escape(phrase)}s?\b", lower):
            added.append((phrase, abbr))
    if not added:
        return text, []
    return text + " " + " ".join(extra for _, extra in added), added


# ==================== ФИЛЬТРЫ ИЗ ТЕКСТА ====================

def extract_hint_filters(text: str) -> Dict[str, Any]:
    """Подсказки-фильтры из вопроса: центр, период, номер урока."""
    hints: Dict[str, Any] = {}
    centers = [code for code, pats in _CENTER_RE.items() if any(p.search(text) for p in pats)]
    if centers:
        hints["org_code"] = centers[0] if len(centers) == 1 else centers

    years: Optional[Dict[str, int]] = None
    if m := _RANGE_RE.search(text):
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        years = {"gte": a, "lte": b}
    elif m := _DECADE_RE.search(text):
        if m.group(3):                                   # '90s
            d = int(m.group(3))
            start = (1900 if d >= 5 else 2000) + d * 10
        else:
            start = int(m.group(1) + m.group(2) + "0")
        years = {"gte": start, "lte": start + 9}
    else:
        lo = _SINCE_RE.search(text)
        hi = _BEFORE_RE.search(text)
        if lo or hi:
            years = {}
            if lo:
                years["gte"] = int(lo.group(1))
            if hi:
                years["lte"] = int(hi.group(1)) - 1
    if years:
        hints["year"] = years

    lessons = _LESSON_RE.findall(text)
    if lessons:
        hints["lesson_ids"] = sorted(set(lessons))
    return hints


# ==================== ПРОГРАММЫ ====================

_PROGRAM_STOP = {"the", "program", "programs", "project", "projects", "mission", "missions", "nasa", "of", "and", "on"}


def _program_tokens(name: str) -> Tuple[set, set]:
    """(значимые слова, аббревиатуры в скобках) названия программы."""
    abbrs = {a.lower() for a in re.findall(r"\(([^)]+)\)", name)}
    words = re.findall(r"[a-z0-9]+", re.sub(r"\([^)]*\)", " ", name.lower()))
    return {w for w in words if w not in _PROGRAM_STOP}, abbrs


class ProgramNormalizer:
    """Сопоставляет название программы со всеми вариантами написания в базе.

    "Lunar Reconnaissance Orbiter" -> ["Lunar Reconnaissance Orbiter", "Lunar Reconnaissance Orbiter TWTA"]
    "MSL" / "Mars Science Laboratory" -> ["Mars Science Laboratory", "Mars Science Laboratory (MSL)"]
    Вариант подходит, если все значимые слова запроса есть в названии варианта (или запрос — его
    аббревиатура). Исходное значение всегда остается в списке, поэтому фильтр только расширяется.
    """

    def __init__(self, programs: Iterable[str]):
        self.programs = sorted({p for p in programs if p and p.strip()})
        self._index = [(p, *_program_tokens(p)) for p in self.programs]

    @classmethod
    def from_lessons(cls, records: Iterable[dict]) -> "ProgramNormalizer":
        return cls(r.get("program") or r.get("metadata", {}).get("program", "") for r in records)

    def variants(self, name: str) -> List[str]:
        words, abbrs = _program_tokens(name)
        query_abbr = name.strip().lower()
        out = {name}
        for prog, p_words, p_abbrs in self._index:
            if words and words <= (p_words | p_abbrs):
                out.add(prog)
            elif query_abbr in p_abbrs or (abbrs and abbrs & p_abbrs):
                out.add(prog)
        return sorted(out)

    def expand_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Заменяет program=<значение> на program=[все варианты]."""
        if not filters or "program" not in filters:
            return filters
        value = filters["program"]
        values = value if isinstance(value, (list, tuple, set)) else [value]
        expanded = sorted({v for item in values for v in self.variants(str(item))})
        return {**filters, "program": expanded}


# ==================== ОБЩАЯ ТОЧКА ВХОДА ====================

def process_query(query: str, abbreviations: Optional[Dict[str, str]] = None,
                  expand: bool = True, extract_filters: bool = True) -> ProcessedQuery:
    text = normalize_text(query)
    sparse_text, expansions = (text, [])
    if expand:
        sparse_text, expansions = expand_abbreviations(
            text, abbreviations if abbreviations is not None else get_abbreviations())
    hints = extract_hint_filters(text) if extract_filters else {}
    return ProcessedQuery(original=query, text=text, sparse_text=sparse_text, hint_filters=hints,
                          expansions=expansions)


def main():
    parser = argparse.ArgumentParser(description="Словарь аббревиатур и проверка обработки запросов")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать data/processed/abbreviations.json")
    parser.add_argument("query", nargs="*", help="вопрос для проверки обработки")
    args = parser.parse_args()
    if args.rebuild:
        abbrs = build_abbreviations(load_lessons())
        ABBREVIATIONS_PATH.write_text(json.dumps(abbrs, ensure_ascii=False, indent=1), encoding="utf-8")
        get_abbreviations.cache_clear()
        print(f"Аббревиатур: {len(abbrs)} -> {ABBREVIATIONS_PATH}")
    if args.query:
        pq = process_query(" ".join(args.query))
        print(f"Dense : {pq.text}\nBM25  : {pq.sparse_text}\nИтог  : {pq.describe()}")


if __name__ == "__main__":
    main()
