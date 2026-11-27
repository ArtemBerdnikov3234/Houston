import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from openpyxl.utils import get_column_letter

from analytics.xlsx_utils import FONT, style_sheet

ROOT = Path(__file__).resolve().parents[1]
K = 10           # how many samples do we take from the search for metrics
N_CONTEXTS = 4   # how many chunks do we include in Resource_1..4 (for Ragas)


def make_filter(raw_filter: str):
    '''{”org_code“: ‘KSC’}’ -> Qdrant filter based on metadata (as in check_qdrant.py).'''
    conds = json.loads(raw_filter) if isinstance(raw_filter, str) and raw_filter.strip() else {}
    if not conds:
        return None
    from qdrant_client import models

    return models.Filter(must=[
        models.FieldCondition(key=f"metadata.{f}", match=models.MatchValue(value=v))
        for f, v in conds.items()
    ])


def first_rank(docs: list, predicate) -> Optional[int]:
    """The position (starting at 1) of the first document that meets the condition; otherwise, None."""
    for i, doc in enumerate(docs, 1):
        if predicate(doc):
            return i
    return None


def evaluate_row(row: dict, docs: list) -> dict:
    """Manual search metrics for a single question. docs — result of similarity_search (in descending relevance order)."""
    relevant = {s.strip() for s in str(row["relevant_lesson_ids"]).split(",") if s.strip()}
    lesson_id = str(row["lesson_id"])
    quote = str(row["context_quote"])

    def lesson_ids(doc) -> set:
        return {str(x) for x in doc.metadata.get("lesson_ids", [])}

    lesson_rank = first_rank(docs, lambda d: bool(relevant & lesson_ids(d)))
    quote_rank = first_rank(docs, lambda d: lesson_id in lesson_ids(d) and quote in d.page_content)

    return {
        "Lesson_Rank": lesson_rank,
        "Quote_Rank": quote_rank,
        "RR": 1 / lesson_rank if lesson_rank else 0.0,
        "Recall@4": int(bool(lesson_rank and lesson_rank <= 4)),
        "Recall@10": int(bool(lesson_rank and lesson_rank <= 10)),
        "QuoteHit@4": int(bool(quote_rank and quote_rank <= 4)),
    }


def build_row(row: dict, docs: list) -> dict:
    """Row in the summary table: question, reference, top-4 chanka, sources, metrics."""
    contexts = [d.page_content for d in docs[:N_CONTEXTS]]
    sources = [d.metadata.get("url", "") for d in docs[:N_CONTEXTS]]
    contexts += [""] * (N_CONTEXTS - len(contexts))   # add empty rows so that there are always 4 columns
    sources += [""] * (N_CONTEXTS - len(sources))

    result = {
        "ID": row["ID"],
        "Вопрос": row["Вопрос"],
        "Эталонный ответ": row["Эталонный ответ"],
        **{f"Resource_{i + 1}": contexts[i] for i in range(N_CONTEXTS)},
        "Sources": "\n".join(s for s in sources if s),
    }
    result.update(evaluate_row(row, docs))
    result["answer"] = ""   # will fill in the RAG sequence
    return result


def add_summary_sheet(workbook, results_ws, columns: list, gold_path: str, n_rows: int) -> None:
    """Summary Sheet: Summary metrics calculated using formulas from the Results sheet."""
    ws = workbook.create_sheet("Summary")
    last = n_rows + 1

    def col(name: str) -> str:
        letter = get_column_letter(columns.index(name) + 1)
        return f"Results!${letter}$2:${letter}${last}"

    ws.append(["Метрика", "Значение", "Как считается"])
    ws.append(["Вопросов", f"=COUNTA({col('ID')})", "Число строк в Gold Set"])
    ws.append(["MRR@10", f"=AVERAGE({col('RR')})", "Среднее 1/ранг первого чанка нужного урока (0, если не найден в top-10)"])
    ws.append(["Recall@4", f"=AVERAGE({col('Recall@4')})", "Доля вопросов, где нужный урок есть в top-4 чанках"])
    ws.append(["Recall@10", f"=AVERAGE({col('Recall@10')})", "Доля вопросов, где нужный урок есть в top-10 чанках"])
    ws.append(["Quote Hit@4", f"=AVERAGE({col('QuoteHit@4')})", "Доля вопросов, где в top-4 есть чанк с дословной эталонной цитатой"])
    ws.append([])
    ws.append(["Gold Set", gold_path, "Источник вопросов. Режим поиска: hybrid (dense + BM25, RRF); K=10"])

    style_sheet(ws, widths={"Метрика": 16, "Значение": 16, "Как считается": 80})
    for r in range(3, 7):
        ws.cell(row=r, column=2).number_format = "0.000"


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold Set -> поиск -> Excel с метриками поиска")
    parser.add_argument("--gold", default=str(ROOT / "analytics" / "gold_set.xlsx"))
    parser.add_argument("--out", default=str(ROOT / "analytics" / "results.xlsx"))
    args = parser.parse_args()

    from src.retrieval.qdrant_store import get_vector_store   

    gold = pd.read_excel(args.gold, dtype={"lesson_id": str, "relevant_lesson_ids": str}).fillna("")
    store = get_vector_store()   # HYBRID по умолчанию

    rows = []
    for i, row in enumerate(gold.to_dict("records"), 1):
        docs = store.similarity_search(row["Вопрос"], k=K, filter=make_filter(row["filter"]))
        rows.append(build_row(row, docs))
        print(f"[{i}/{len(gold)}] {row['Вопрос'][:70]}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
        ws = writer.book["Results"]
        style_sheet(ws, widths={
            "ID": 6, "Вопрос": 45, "Эталонный ответ": 35, "Resource_1": 50, "Resource_2": 50,
            "Resource_3": 50, "Resource_4": 50, "Sources": 32, "answer": 40,
        }, default_width=12)
        add_summary_sheet(writer.book, ws, list(df.columns), args.gold, len(df))

    print(f"Готово -> {out}")


if __name__ == "__main__":
    main()
