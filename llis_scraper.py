"""NASA Lessons Learned Information System (LLIS) Scraper & Preprocessor for RAG.

Скачивает уроки NASA LLIS, проводит глубокую очистку и семантическую нормализацию текста,
сохраняет метаданные и структуру документа в формате Markdown, готовом к чанкингу в LangChain.

Установка:  pip install requests beautifulsoup4
Запуск:     python llis_scraper.py                         # скачать всю базу и обработать
            python llis_scraper.py --from-raw llis_raw.json  # переобработать сохраненный ответ API
Результат:  llis_full.json (вся база LLIS), llis_raw.json (сырой ответ API)
"""

import argparse
import hashlib
import html
import json
import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# ==================== КОНФИГУРАЦИЯ ====================
URL = "https://llis.nasa.gov/llis/lesson/_search"
SITE = "https://llis.nasa.gov"
MAX_RECORDS = None      # Целевое число документов (None = вся база)
ORG = None              # None = все центры NASA; или "msfc", "ksc", "jpl", "jsc", "gsfc", "arc", "larc"
PAGE_SIZE = 50          # Размер пачки в одном API-запросе
PAUSE = 0.4             # Пауза между запросами (сек)
RETRIES = 4             # Число повторов запроса при сетевой ошибке
OUTPUT_PREFIX = "llis_full" if not (ORG or MAX_RECORDS) else f"llis_{ORG or 'all'}_{MAX_RECORDS or 'all'}"
RAW_PATH = "llis_raw.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
    "Content-Type": "text/plain;charset=UTF-8",
}

# Строковые маркеры отсутствия информации в NASA LLIS
DUMMY_VALUES = {
    "none", "none.", "n/a", "na", "null", "tbd", "unknown", "not specified", "various",
    "no recommendation", "no recommendations", "no recommendation.",
    "not applicable", "see recommendation", "see lesson", "n / a"
}
# Короткие отсылки вместо текста секции: «See "Recommendations" section above», «Same as Lesson Learned»
PLACEHOLDER_RE = re.compile(
    r"^\(?see (the )?\"?(recommendations?|lessons?( learned)?|driving event|abstract|above|below|attach\w*|report|pdf)\b"
    r"|^same as\b|^no description|^this is (only )?a test|^tbd\b|^click (here )?to view"
    r"|^the text has been redacted",
    re.IGNORECASE,
)
PLACEHOLDER_MAX_LEN = 100

# Служебный маркер «настоящего» переноса строки (из тегов). Сырые переводы строк внутри HTML — это пробелы.
NL = "\u2029"
HTML_STRUCTURE_RE = re.compile(r"<(p|br|div|li|ul|ol|table|tr|h[1-6])\b", re.IGNORECASE)
LIST_MARKER_RE = re.compile(r"^([-*•]|\d+[\.)]|[a-zA-Z][\.)])\s+")
ORPHAN_LINE_RE = re.compile(r"^[-*•_^|]+$")   # строки из одних маркеров: пустые пункты, осиротевшие ^/_
# Служебные строки веб-страницы, бессмысленные без самой страницы
BOILERPLATE_LINE_RE = re.compile(r"^click (here )?to view\b", re.IGNORECASE)


def build_payload(start: int, size: int) -> dict:
    """Формирует тело запроса с детерминированной сортировкой для стабильной пагинации."""
    payload = {
        "sort": [{"lesson_date": "desc"}, {"_id": "asc"}],
        "from": start,
        "size": size,
        "query": {"match_all": {}},
    }
    if ORG:
        payload["filter"] = {"bool": {"must": [{"term": {"organization.abr": ORG.lower()}}]}}
    return payload


def fetch_page(session: requests.Session, start: int, size: int) -> dict:
    """Запрашивает порцию документов из API NASA LLIS с таймаутом и повторами."""
    body = json.dumps(build_payload(start, size))
    for attempt in range(1, RETRIES + 1):
        try:
            resp = session.post(URL, data=body, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == RETRIES:
                raise
            wait = 2 ** attempt
            print(f"  !! Ошибка запроса (from={start}): {e}. Повтор через {wait} c...")
            time.sleep(wait)


def get_total(data: dict) -> int:
    """Число документов в базе (поддерживает int и формат ES 7+ {"value": N})."""
    total = data["hits"]["total"]
    return total["value"] if isinstance(total, dict) else int(total)


def is_dummy_value(text: str) -> bool:
    """Проверяет, не является ли текст служебной заглушкой (None, N/A и т.п.)."""
    cleaned = text.strip().lower().rstrip(".")
    return cleaned in DUMMY_VALUES or cleaned == ""


def is_placeholder(text: str) -> bool:
    """Заглушка или короткая отсылка к другой секции вместо содержательного текста."""
    t = text.strip()
    return is_dummy_value(t) or (len(t) < PLACEHOLDER_MAX_LEN and bool(PLACEHOLDER_RE.match(t)))


def _flat(node) -> str:
    """Текст узла: пробелы схлопнуты, маркеры переносов строк сохранены."""
    t = re.sub(r"[ \t\r\n\f\v]+", " ", node.get_text())
    t = re.sub(f" *{NL} *", NL, t)
    return t.strip(" " + NL)


def html_to_markdown(raw_html: Any) -> str:
    """Преобразует произвольный HTML в чистый, структурированный Markdown.

    1. Раскрывает HTML-сущности (&apos;, &quot;, &deg;, &plusmn;, &micro;, &reg;, &amp;).
    2. Сырые переводы строк внутри HTML считает пробелами (как браузер), переносы задают только теги;
       в простом тексте без разметки переводы строк сохраняются.
    3. Преобразует <table> в Markdown-таблицы, пропуская пустые строки.
    4. Преобразует <ol>/<ul> (включая вложенные) в списки, отбрасывая пустые пункты (<li><br/></li>).
    5. Преобразует непустые <sub>/<sup> в индексы (H_2O, 10^6).
    6. Чистит \xa0, ​, артефакт двойной кодировки «Â» и строки из одних маркеров.
    """
    if raw_html is None:
        return ""
    if isinstance(raw_html, list):
        cleaned_items = [html_to_markdown(item) for item in raw_html]
        valid = [item for item in cleaned_items if not is_dummy_value(item)]
        return "; ".join(valid)

    text = str(raw_html)
    if is_dummy_value(text):
        return ""

    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("​", "").replace(" ", " ").replace("\r", "")
    # Артефакты неверной кодировки: «contractorÂ¹s» -> «contractor's», «Â » -> « »
    text = text.replace("Â¹", "'")
    text = re.sub(r"Â(?=\s|$)", "", text)

    # Индексы: только непустые <sub>/<sup> без вложенных тегов, пустые удаляются
    text = re.sub(r"<sub[^>]*>\s*([^<>]+?)\s*</sub>", r"_\1", text, flags=re.IGNORECASE)
    text = re.sub(r"<sup[^>]*>\s*([^<>]+?)\s*</sup>", r"^\1", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(sub|sup)[^>]*>", "", text, flags=re.IGNORECASE)
    # Невалидный «</br>» парсер молча выбрасывает, склеивая слова — приводим все варианты к <br>
    text = re.sub(r"</?\s*br\s*/?\s*>", "<br>", text, flags=re.IGNORECASE)
    text = text.replace("</</", "</")  # битый закрывающий тег «</</TD>» ломает разбор таблицы
    # Курсив вплотную к соседнему слову («(<i>since</i>R = 0.98)») — отделяем пробелом
    text = re.sub(r"</(i|em)>(?=\w)", r"</\1> ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\w)<(i|em)>", r" <\1>", text, flags=re.IGNORECASE)
    # Конец фразы, отделенный от следующей только тегом («Counterfeiting.</b>Charter», «operation.<LO>Cutter»)
    text = re.sub(r"([.!?:;])((?:<[^>]+>)+)(?=[A-Z])", r"\1\2 ", text)

    if HTML_STRUCTURE_RE.search(text):
        text = re.sub(r"\s+", " ", text)
    else:
        text = text.replace("\n", NL)

    soup = BeautifulSoup(text, "html.parser")

    for br in soup.find_all("br"):
        br.replace_with(NL)

    # Блочные элементы -> абзацы (до таблиц и списков, чтобы абзацы внутри ячеек и пунктов не склеивались)
    for block in soup.find_all(["p", "div", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]):
        block.insert_before(NL * 2)
        block.insert_after(NL * 2)
        block.unwrap()

    # Таблицы HTML -> Markdown
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [_flat(c).replace(NL, " ") for c in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append("| " + " | ".join(cells) + " |")
        if len(rows) > 1:
            num_cols = rows[0].count(" | ") + 1
            rows.insert(1, "| " + " | ".join(["---"] * num_cols) + " |")
        table.replace_with(NL * 2 + NL.join(rows) + NL * 2 if rows else NL)

    # Списки: от самых вложенных к внешним, чтобы вложенные пункты получили отступ
    for lst in reversed(soup.find_all(["ul", "ol"])):
        idx = 0
        for li in lst.find_all("li", recursive=False):
            content = _flat(li)
            if not re.search(r"\w", content) or is_dummy_value(content):
                li.decompose()
                continue
            idx += 1
            lines = [ln for ln in content.split(NL) if ln.strip()]
            first = lines[0]
            if not LIST_MARKER_RE.match(first):
                first = (f"{idx}. " if lst.name == "ol" else "- ") + first
            else:
                first = re.sub(r"^[•*]\s*", "- ", first)
            li.replace_with(NL + NL.join([first] + ["  " + ln for ln in lines[1:]]) + NL)
        lst.replace_with(NL + lst.get_text() + NL)

    clean = soup.get_text().replace(NL, "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in clean.split("\n")]
    lines = [ln for ln in lines if not ORPHAN_LINE_RE.match(ln) and not BOILERPLATE_LINE_RE.match(ln)]
    clean = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    # В исходных данных NASA встречаются кавычки, потерявшие «&» и «;» («quotCorrectly orient...quot»)
    # или экранированные дважды («&quot)»); слова quota/quote/quotient/aliquot не затрагиваются
    clean = re.sub(r"&quot;?|(?<!ali)quot(?![aei])", '"', clean)
    return "" if is_dummy_value(clean) else clean


def clean_section(*raw_values: Any) -> str:
    """Очищает текст секции. Если в API есть несколько версий поля: версия, целиком входящая
    в другую, отбрасывается (берется более полная); действительно разные тексты объединяются."""
    texts = [t for t in (html_to_markdown(v) for v in raw_values) if t and not is_placeholder(t)]
    norm = lambda t: re.sub(r"\W+", " ", t).lower().strip()
    kept: List[str] = []
    for t in sorted(dict.fromkeys(texts), key=len, reverse=True):
        if not any(norm(t) in norm(k) for k in kept):
            kept.append(t)
    return "\n\n".join(kept)


def normalize_category(name: str) -> str:
    """Унифицирует написание: «Safety & Mission Assurance» -> «Safety and Mission Assurance»."""
    return re.sub(r"\s+", " ", name.replace("&", " and ")).strip()


def parse_categories(raw_categories: Any) -> tuple[List[str], str]:
    """Извлекает категории в виде списка строк и единой строки для метаданных."""
    if not raw_categories:
        return [], ""

    cat_names = []
    if isinstance(raw_categories, dict):
        val = raw_categories.get("name")
        if isinstance(val, list):
            cat_names = [str(x).strip() for x in val]
        elif val:
            cat_names = [str(val).strip()]
    elif isinstance(raw_categories, list):
        for item in raw_categories:
            if isinstance(item, dict):
                c = item.get("name")
                if isinstance(c, list):
                    cat_names.extend([str(x).strip() for x in c])
                elif c:
                    cat_names.append(str(c).strip())
            elif item:
                cat_names.append(str(item).strip())
    elif isinstance(raw_categories, str):
        cat_names = [c.strip() for c in raw_categories.split(";") if c.strip()]

    # Фильтруем пустые и заглушки, унифицируем «&»/«and»
    filtered = [normalize_category(c) for c in cat_names if not is_dummy_value(c)]
    # Дедупликация с сохранением порядка
    unique_cats = list(dict.fromkeys(filtered))
    return unique_cats, "; ".join(unique_cats)


def parse_keywords(s: dict) -> List[str]:
    """Ключевые слова из tags и classification (без служебного «Lesson Learned»)."""
    words = []
    for field in ("tags", "classification"):
        raw = html_to_markdown(s.get(field))
        words += [w.strip().lower() for w in raw.split(",") if w.strip()]
    return list(dict.fromkeys(w for w in words if w != "lesson learned" and not is_dummy_value(w)))


def parse_attachments(s: dict) -> List[Dict[str, str]]:
    """Вложения урока. URL есть только у поля attachments; у documentName* сохраняется лишь название."""
    items, seen = [], set()
    for a in s.get("attachments") or []:
        name = str(a.get("name") or "").strip()
        link = str(a.get("link") or "").strip()
        if name and name not in seen:
            seen.add(name)
            items.append({"name": name, "url": SITE + link if link.startswith("/") else link})
    for i in range(10):
        name = str(s.get(f"documentName{i}") or "").strip()
        if name and name not in seen:
            seen.add(name)
            items.append({"name": name, "url": ""})
    return items


def derive_title(sections: Dict[str, str]) -> str:
    """Название из первой фразы текста — для уроков, у которых в API title = «None»."""
    for key in ("abstract", "driving_event", "lesson_learned", "recommendation"):
        text = sections.get(key)
        if text:
            line = LIST_MARKER_RE.sub("", text.split("\n")[0]).strip()
            first = re.split(r"(?<=[.!?])\s", line)[0].rstrip(".")
            if len(first) < 30:  # обрыв на сокращении вроде «Feb.» — берем всю строку
                first = line.rstrip(".")
            if len(first) > 100:
                first = first[:100].rsplit(" ", 1)[0] + "…"
            return first
    return "Untitled lesson"


SECTION_HEADERS = {
    "abstract": "Abstract",
    "driving_event": "Driving Event / Background",
    "lesson_learned": "Lesson Learned",
    "recommendation": "Recommendation",
    "evidence": "Evidence of Recurrence Control Effectiveness",
    "references": "Related Documents and Policy",
}


def parse_hit(hit: dict) -> Optional[dict]:
    """Трансформирует сырой хит Elasticsearch в чистый документ, готовый к RAG."""
    s = hit.get("_source", {})
    lesson_id = hit["_id"]

    date_str = str(s.get("lesson_date") or "").strip()

    # Извлечение числового года для мета-фильтрации
    year: Optional[int] = None
    if date_str:
        year_match = re.match(r"^(\d{4})", date_str)
        if year_match:
            year = int(year_match.group(1))

    # Организация / Центр NASA
    org_dict = s.get("organization") or {}
    if isinstance(org_dict, dict):
        org_name = org_dict.get("name") or "NASA"
        org_code = (org_dict.get("abr") or "").upper()
    else:
        org_name = str(org_dict)
        org_code = ""

    # Направление миссии
    mission_raw = s.get("missionDirectorate")
    if isinstance(mission_raw, dict):
        mission_raw = mission_raw.get("name")
    mission_dir = html_to_markdown(mission_raw)

    categories_list, categories_str = parse_categories(s.get("categories"))
    program = html_to_markdown(s.get("programRelation"))
    program_phase = html_to_markdown(s.get("programPhase")).replace("»", ">")
    if program_phase.lower() in ("not applicable", "not specified"):
        program_phase = ""
    keywords = parse_keywords(s)
    attachments = parse_attachments(s)

    # Секции документа. У части уроков текст продублирован в двух полях API разной полноты.
    sections = {
        "abstract": clean_section(s.get("lessonAbstract")),
        "driving_event": clean_section(s.get("drivingEvent"), s.get("description_event")),
        "lesson_learned": clean_section(s.get("lesson_learned"), s.get("lesson")),
        "recommendation": clean_section(s.get("recommendation")),
        "evidence": clean_section(s.get("evidence")),
    }
    policy = clean_section(s.get("relatedPolicy"))
    documented = clean_section(s.get("lessonDocumented"))
    sections["references"] = "\n\n".join(dict.fromkeys(t for t in (policy, documented) if t))

    # Валидация качества: отбраковываем записи без основного содержания
    if not any(sections[k] for k in ("abstract", "driving_event", "lesson_learned", "recommendation")):
        return None

    title = html_to_markdown(s.get("title"))
    title_derived = not title
    if title_derived:
        title = derive_title(sections)

    # Построение структурированного Markdown-представления документа
    header_lines = [
        f"# [NASA LLIS #{lesson_id}] {title}",
        f"- **Date:** {date_str or 'Unknown'}",
        f"- **Organization:** {org_name} ({org_code if org_code else 'NASA'})",
    ]
    if mission_dir:
        header_lines.append(f"- **Mission Directorate:** {mission_dir}")
    if program:
        header_lines.append(f"- **Program:** {program}")
    if program_phase:
        header_lines.append(f"- **Program Phase:** {program_phase}")
    if categories_str:
        header_lines.append(f"- **Categories:** {categories_str}")
    if keywords:
        header_lines.append(f"- **Keywords:** {', '.join(keywords)}")
    header_lines.append(f"- **Source URL:** {SITE}/lesson/{lesson_id}")

    sections_md = [f"## {SECTION_HEADERS[k]}\n{v}" for k, v in sections.items() if v]
    if attachments:
        sections_md.append("## Attachments\n" + "\n".join(f"- {a['name']}" for a in attachments))

    full_markdown = "\n".join(header_lines) + "\n\n" + "\n\n".join(sections_md)

    word_count = len(full_markdown.split())
    estimated_tokens = int(word_count * 1.3)

    return {
        "id": lesson_id,
        "lesson_number": s.get("lesson_number") or lesson_id,
        "url": f"{SITE}/lesson/{lesson_id}",
        "title": title,
        "date": date_str,
        "year": year,
        "organization": org_name,
        "org_code": org_code,
        "mission_directorate": mission_dir,
        "program": program,
        "program_phase": program_phase,
        "categories": categories_str,
        "categories_list": categories_list,
        "keywords": keywords,
        "attachments": attachments,
        "sections": sections,
        "full_text": full_markdown,
        # Метаданные, готовые для передачи в LangChain Document
        "metadata": {
            "lesson_id": lesson_id,
            "title": title,
            "title_derived": title_derived,
            "date": date_str,
            "year": year,
            "org_code": org_code,
            "org_name": org_name,
            "mission_directorate": mission_dir,
            "program": program,
            "program_phase": program_phase,
            "categories": categories_str,
            "url": f"{SITE}/lesson/{lesson_id}",
            "has_abstract": bool(sections["abstract"]),
            "has_driving_event": bool(sections["driving_event"]),
            "has_lesson": bool(sections["lesson_learned"]),
            "has_recommendation": bool(sections["recommendation"]),
            "has_evidence": bool(sections["evidence"]),
            "duplicate_of": None,
            "char_count": len(full_markdown),
            "word_count": word_count,
            "estimated_tokens": estimated_tokens,
        },
    }


def mark_duplicates(records: List[dict]) -> int:
    """Помечает уроки, текст которых полностью совпадает с более ранним (по ID) уроком."""
    first_by_hash: Dict[str, str] = {}
    marked = 0
    for r in sorted(records, key=lambda r: int(r["id"]) if r["id"].isdigit() else r["id"]):
        body = "\n".join(r["sections"][k] for k in ("abstract", "driving_event", "lesson_learned", "recommendation"))
        key = hashlib.md5(re.sub(r"\s+", " ", body.lower()).encode()).hexdigest()
        if key in first_by_hash:
            r["metadata"]["duplicate_of"] = first_by_hash[key]
            marked += 1
        else:
            first_by_hash[key] = r["id"]
    return marked


def fetch_all() -> List[dict]:
    """Скачивает сырые хиты из API с пагинацией."""
    session = requests.Session()
    first = fetch_page(session, 0, PAGE_SIZE)
    total_in_db = get_total(first)
    target_count = min(total_in_db, MAX_RECORDS) if MAX_RECORDS else total_in_db
    print(f"Доступно в базе     : {total_in_db}")
    print(f"Будет выгружено     : {target_count}")
    print("-" * 60)

    hits: List[dict] = []
    seen = set()
    start = 0
    data = first
    while True:
        page = data["hits"]["hits"]
        if not page:
            break
        for h in page:
            if h["_id"] not in seen:
                seen.add(h["_id"])
                hits.append(h)
        print(f"  -> Скачано: {len(hits)} / {target_count}")
        if (MAX_RECORDS and len(hits) >= MAX_RECORDS) or start + PAGE_SIZE >= total_in_db:
            break
        start += PAGE_SIZE
        time.sleep(PAUSE)
        data = fetch_page(session, start, PAGE_SIZE)
    return hits[:MAX_RECORDS] if MAX_RECORDS else hits


def main():
    parser = argparse.ArgumentParser(description="NASA LLIS scraper & preprocessor")
    parser.add_argument("--from-raw", metavar="PATH", help="не обращаться к API, обработать сохраненный ответ")
    args = parser.parse_args()

    print("=" * 60)
    print("  NASA LLIS — Сбор и подготовка данных для RAG-системы")
    print("=" * 60)

    if args.from_raw:
        with open(args.from_raw, "r", encoding="utf-8") as f:
            hits = json.load(f)
        print(f"Сырые данные        : {args.from_raw} ({len(hits)} записей)")
    else:
        print(f"Целевой центр NASA : {ORG.upper() if ORG else 'Все центры'}")
        print(f"Лимит документов    : {MAX_RECORDS if MAX_RECORDS else 'Вся база'}")
        hits = fetch_all()
        with open(RAW_PATH, "w", encoding="utf-8") as f:
            json.dump(hits, f, ensure_ascii=False)

    records = [doc for doc in (parse_hit(h) for h in hits) if doc]
    rejected = len(hits) - len(records)
    duplicates = mark_duplicates(records)

    json_path = f"{OUTPUT_PREFIX}.json"

    print("\nСохранение файлов...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"УСПЕШНО СОХРАНЕНО: {len(records)} документов "
          f"(получено: {len(hits)}, отбраковано пустых: {rejected}, полных дублей помечено: {duplicates})")
    print(f"  - JSON  : {os.path.abspath(json_path)}")
    if not args.from_raw:
        print(f"  - RAW   : {os.path.abspath(RAW_PATH)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
