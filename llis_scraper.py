"""NASA Lessons Learned Information System (LLIS) Scraper & Preprocessor for RAG.

Скачивает уроки NASA LLIS, проводит глубокую очистку и семантическую нормализацию текста,
сохраняет метаданные и структуру документа в формате Markdown, готовом к чанкингу в LangChain / LlamaIndex.

Установка:  pip install requests beautifulsoup4
Запуск:     python llis_scraper.py
Результат:  llis_full.json и llis_full.jsonl (вся база LLIS)
"""

import html
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

# ==================== КОНФИГУРАЦИЯ ====================
URL = "https://llis.nasa.gov/llis/lesson/_search"
MAX_RECORDS = None      # Целевое число документов (None = вся база)
ORG = None              # None = все центры NASA; или "msfc", "ksc", "jpl", "jsc", "gsfc", "arc", "larc"
PAGE_SIZE = 50          # Размер пачки в одном API-запросе
PAUSE = 0.4             # Пауза между запросами (сек)
RETRIES = 4             # Число повторов запроса при сетевой ошибке
OUTPUT_PREFIX = "llis_full" if not (ORG or MAX_RECORDS) else f"llis_{ORG or 'all'}_{MAX_RECORDS or 'all'}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
    "Content-Type": "text/plain;charset=UTF-8",
}

# Строковые маркеры отсутствия информации в NASA LLIS
DUMMY_VALUES = {
    "none", "none.", "n/a", "na", "null", "tbd", "unknown",
    "no recommendation", "no recommendations", "no recommendation.",
    "not applicable", "see recommendation", "see lesson", "n / a"
}


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


def html_to_markdown(raw_html: Any) -> str:
    """Преобразует произвольный HTML в чистый, структурированный Markdown.
    
    1. Раскрывает HTML-сущности (&apos;, &quot;, &deg;, &plusmn;, &micro;, &reg;, &amp;).
    2. Преобразует <table> в аккуратные Markdown-таблицы.
    3. Преобразует <ol> и <ul> в списки (1. ... и - ...), предотвращая дублирование номеров.
    4. Преобразует <sub> и <sup> в математические индексы (H_2O, 10^6).
    5. Заменяет <p>, <div>, <br> на нормализованные переносы строк.
    6. Чистит неразрывные пробелы (\xa0), нулевые пробелы (\u200b) и лишние пробелы.
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

    # Раскрываем HTML-сущности (&nbsp; -> пробел, &deg; -> °, &apos; -> ' и т.д.)
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\u202f", " ")

    # Обработка математических и физических индексов до парсинга тегов
    text = re.sub(r"<sub[^>]*>(.*?)</sub>", r"_\1", text, flags=re.IGNORECASE)
    text = re.sub(r"<sup[^>]*>(.*?)</sup>", r"^\1", text, flags=re.IGNORECASE)

    soup = BeautifulSoup(text, "html.parser")

    # Преобразование таблиц HTML в Markdown
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            # Если найдена шапка или несколько строк, вставляем разделитель Markdown
            num_cols = len(table.find_all("tr")[0].find_all(["th", "td"]))
            if len(rows) > 1 and num_cols > 0:
                header_sep = "| " + " | ".join(["---"] * num_cols) + " |"
                rows.insert(1, header_sep)
            table.replace_with("\n\n" + "\n".join(rows) + "\n\n")

    # Преобразование списков <ul> (ненумерованные)
    for ul in soup.find_all("ul"):
        for li in ul.find_all("li", recursive=False):
            li_text = li.get_text(" ", strip=True)
            # Проверяем, не начинается ли элемент уже с маркера или цифры
            if not re.match(r"^([-*•]|\d+[\.)]|[a-zA-Z][\.)])\s+", li_text):
                li_text = "- " + li_text
            else:
                li_text = re.sub(r"^[•*]\s*", "- ", li_text)
            li.replace_with("\n" + li_text + "\n")

    # Преобразование списков <ol> (нумерованные)
    for ol in soup.find_all("ol"):
        for idx, li in enumerate(ol.find_all("li", recursive=False), 1):
            li_text = li.get_text(" ", strip=True)
            if not re.match(r"^(\d+[\.)]|[a-zA-Z][\.)])\s+", li_text):
                li_text = f"{idx}. " + li_text
            li.replace_with("\n" + li_text + "\n")

    # Обработка параграфов, заголовков и разрывов строк
    for block in soup.find_all(["p", "div", "blockquote"]):
        block_text = block.get_text(" ", strip=True)
        block.replace_with("\n\n" + block_text + "\n\n")

    for br in soup.find_all("br"):
        br.replace_with("\n")

    # Получаем чистый текст
    clean = soup.get_text()

    # Финальная нормализация пробелов и переносов строк
    clean = clean.replace("\r", "")
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"^[ \t]+|[ \t]+$", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"\n{3,}", "\n\n", clean)

    res = clean.strip()
    return "" if is_dummy_value(res) else res


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

    # Фильтруем пустые и заглушки
    filtered = [c for c in cat_names if not is_dummy_value(c)]
    # Дедупликация с сохранением порядка
    unique_cats = list(dict.fromkeys(filtered))
    return unique_cats, "; ".join(unique_cats)


def parse_hit(hit: dict) -> Optional[dict]:
    """Трансформирует сырой хит Elasticsearch в чистый документ, готовый к RAG."""
    s = hit.get("_source", {})
    lesson_id = hit["_id"]

    # Основные поля
    title = html_to_markdown(s.get("title"))
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

    # Категории
    categories_list, categories_str = parse_categories(s.get("categories"))

    # Секции документа
    abstract = html_to_markdown(s.get("lessonAbstract"))
    driving_event = html_to_markdown(s.get("drivingEvent") or s.get("description_event"))
    lesson_learned = html_to_markdown(s.get("lesson_learned") or s.get("lesson"))
    recommendation = html_to_markdown(s.get("recommendation"))

    # Валидация качества: отбраковываем пустые записи
    if not any([abstract, driving_event, lesson_learned, recommendation]):
        return None

    # Построение структурированного Markdown-представления документа
    header_lines = [
        f"# [NASA LLIS #{lesson_id}] {title}",
        f"- **Date:** {date_str or 'Unknown'}",
        f"- **Organization:** {org_name} ({org_code if org_code else 'NASA'})",
    ]
    if mission_dir:
        header_lines.append(f"- **Mission Directorate:** {mission_dir}")
    if categories_str:
        header_lines.append(f"- **Categories:** {categories_str}")
    header_lines.append(f"- **Source URL:** https://llis.nasa.gov/lesson/{lesson_id}")

    sections_md = []
    if abstract:
        sections_md.append(f"## Abstract\n{abstract}")
    if driving_event:
        sections_md.append(f"## Driving Event / Background\n{driving_event}")
    if lesson_learned:
        sections_md.append(f"## Lesson Learned\n{lesson_learned}")
    if recommendation:
        sections_md.append(f"## Recommendation\n{recommendation}")

    full_markdown = "\n".join(header_lines) + "\n\n" + "\n\n".join(sections_md)

    word_count = len(full_markdown.split())
    estimated_tokens = int(word_count * 1.3)

    return {
        "id": lesson_id,
        "lesson_number": s.get("lesson_number") or lesson_id,
        "url": f"https://llis.nasa.gov/lesson/{lesson_id}",
        "title": title,
        "date": date_str,
        "year": year,
        "organization": org_name,
        "org_code": org_code,
        "mission_directorate": mission_dir,
        "categories": categories_str,
        "categories_list": categories_list,
        "sections": {
            "abstract": abstract,
            "driving_event": driving_event,
            "lesson_learned": lesson_learned,
            "recommendation": recommendation,
        },
        "full_text": full_markdown,
        # Метаданные, готовые для передачи в LangChain Document или LlamaIndex Node
        "metadata": {
            "lesson_id": lesson_id,
            "title": title,
            "date": date_str,
            "year": year,
            "org_code": org_code,
            "org_name": org_name,
            "mission_directorate": mission_dir,
            "categories": categories_str,
            "url": f"https://llis.nasa.gov/lesson/{lesson_id}",
            "has_abstract": bool(abstract),
            "has_driving_event": bool(driving_event),
            "has_lesson": bool(lesson_learned),
            "has_recommendation": bool(recommendation),
            "char_count": len(full_markdown),
            "word_count": word_count,
            "estimated_tokens": estimated_tokens,
        },
    }


def main():
    print("=" * 60)
    print("  NASA LLIS — Сбор и подготовка данных для RAG-системы")
    print("=" * 60)
    print(f"Целевой центр NASA : {ORG.upper() if ORG else 'Все центры'}")
    print(f"Лимит документов    : {MAX_RECORDS if MAX_RECORDS else 'Вся база'}")

    session = requests.Session()
    first = fetch_page(session, 0, PAGE_SIZE)
    total_in_db = get_total(first)
    target_count = min(total_in_db, MAX_RECORDS) if MAX_RECORDS else total_in_db

    print(f"Доступно в базе     : {total_in_db}")
    print(f"Будет выгружено     : {target_count}")
    print("-" * 60)

    records: List[dict] = []
    seen = set()
    rejected = 0
    start = 0
    data = first

    while True:
        hits = data["hits"]["hits"]
        if not hits:
            break

        for h in hits:
            doc_id = h["_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                doc = parse_hit(h)
                if doc:
                    records.append(doc)
                else:
                    rejected += 1

                if MAX_RECORDS and len(records) >= MAX_RECORDS:
                    break

        print(f"  -> Обработано: {len(records)} / {target_count}")

        if (MAX_RECORDS and len(records) >= MAX_RECORDS) or start + PAGE_SIZE >= total_in_db:
            break

        start += PAGE_SIZE
        time.sleep(PAUSE)
        data = fetch_page(session, start, PAGE_SIZE)

    # Сохраняем в JSON и JSONL
    json_path = f"{OUTPUT_PREFIX}.json"
    jsonl_path = f"{OUTPUT_PREFIX}.jsonl"

    print("\nСохранение файлов...")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"УСПЕШНО СОХРАНЕНО: {len(records)} документов (уникальных ID: {len(seen)}, отбраковано пустых: {rejected})")
    print(f"  - JSON  : {os.path.abspath(json_path)}")
    print(f"  - JSONL : {os.path.abspath(jsonl_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
