"""Скрипт для экспресс-анализа и проверки качества подготовленных датасетов NASA LLIS.

Запуск:
    python inspect_dataset.py                    # Проверит самый свежий файл llis_*.json
    python inspect_dataset.py llis_full.json     # Проверит конкретный файл
"""

import glob
import json
import os
import sys

# Определяем файл для анализа: аргумент командной строки или самый свежий json
files = sys.argv[1:] or sorted(glob.glob("llis_*.json"), key=os.path.getmtime, reverse=True)
if not files:
    print("Не найдено файлов llis_*.json для анализа")
    sys.exit(0)

target = files[0]
print(f"=== Анализ датасета: {target} ===")

with open(target, "r", encoding="utf-8") as f:
    recs = json.load(f)

print(f"Всего документов: {len(recs)} | Уникальных ID: {len({r['id'] for r in recs})}")

# Распределение по центрам
orgs = {}
for r in recs:
    code = r.get("org_code") or r.get("organization") or "UNKNOWN"
    orgs[code] = orgs.get(code, 0) + 1
print("\nТоп центров NASA:")
for org, count in sorted(orgs.items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f"  - {org}: {count}")

# Наличие секций
print("\nЗаполненность разделов:")
sections = ["abstract", "driving_event", "lesson_learned", "recommendation", "evidence", "references"]
for s in sections:
    filled = sum(1 for r in recs if r.get("sections", {}).get(s) or r.get(s))
    print(f"  - {s}: {filled}/{len(recs)} ({filled*100//len(recs)}%)")

# Длины текстов
text_key = "full_text" if "full_text" in recs[0] else "text"
lens = sorted(len(r[text_key]) for r in recs)
print(f"\nДлина {text_key} (в символах):")
print(f"  Мин: {lens[0]} | 25%: {lens[len(lens)//4]} | Медиана: {lens[len(lens)//2]} | 75%: {lens[3*len(lens)//4]} | Макс: {lens[-1]}")
print(f"  Примерно токенов (~4 симв/токен): ~{lens[len(lens)//2]//4} медиана, ~{lens[-1]//4} макс")

# Пример подготовленного документа
print(f"\n=== Пример подготовленного документа [ID: {recs[0]['id']}] ===")
preview = recs[0][text_key]
print(preview[:500] + ("..." if len(preview) > 500 else ""))
