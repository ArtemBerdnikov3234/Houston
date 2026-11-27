import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from analytics.xlsx_utils import style_sheet

ROOT = Path(__file__).resolve().parents[1]


def gold_to_dataframe(gold: list) -> pd.DataFrame:
    """Gold Set (JSON) -> DataFrame. Columns: ID | Вопрос | Эталонный ответ | lesson_id | relevant_lesson_ids | context_quote | filter"""
    rows = []
    for g in gold:
        relevant = g.get("relevant_lesson_ids") or [g["lesson_id"]]
        rows.append({
            "ID": g["id"],
            "Вопрос": g["question"],
            "Эталонный ответ": g["ground_truth_answer"],
            "lesson_id": str(g["lesson_id"]),
            "relevant_lesson_ids": ",".join(str(x) for x in relevant),
            "context_quote": g["context_quote"],
            "filter": json.dumps(g.get("filter") or {}, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold Set JSON -> Excel")
    parser.add_argument("--src", default=str(ROOT / "data" / "benchmarks" / "gold_set_50.json"))
    parser.add_argument("--out", default=str(ROOT / "analytics" / "gold_set.xlsx"))
    args = parser.parse_args()

    gold = json.loads(Path(args.src).read_text(encoding="utf-8"))
    df = gold_to_dataframe(gold)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False, sheet_name="Gold Set")

    wb = load_workbook(out)
    style_sheet(wb["Gold Set"], widths={
        "ID": 6, "Вопрос": 55, "Эталонный ответ": 45, "lesson_id": 11,
        "relevant_lesson_ids": 16, "context_quote": 60, "filter": 20,
    })
    wb.save(out)
    print(f"Готово: {len(df)} вопросов -> {out}")


if __name__ == "__main__":
    main()
