"""Оценка RAG-цепочки Houston (поиск + Mistral) на Gold Set и на вопросах вне базы.

Для каждого вопроса сохраняются ответ, источники и контекст (data/evaluation/rag_answers_<набор>.jsonl),
затем считаются метрики.

Собственные метрики (без дополнительных вызовов LLM):
  answered          — модель дала ответ, а не отказ;
  cites_relevant    — среди ссылок ответа есть правильный урок (lesson_id / relevant_lesson_ids);
  context_relevant  — правильный урок был в контексте (верхняя граница для cites_relevant);
  false_refusal     — отказ, хотя правильный урок был в контексте;
  invalid_citations — доля ответов со ссылками на уроки вне контекста (выдуманные источники);
  uncited_answers   — доля ответов без единой ссылки;
  correct_refusal   — для out_of_scope.json: доля вопросов вне базы, на которые модель честно отказала;
  latency_s, tokens — среднее время и токены на вопрос.

Ragas (--ragas, нужен pip install -r requirements-eval.txt), судья — Mistral через OpenAI-совместимый API:
  faithfulness       — утверждения ответа подтверждаются контекстом;
  answer_relevancy   — ответ отвечает именно на вопрос (эмбеддинги bge-base);
  context_precision  — релевантные уроки стоят в контексте выше (с эталонным ответом);
  context_recall     — контекст содержит то, что нужно для эталонного ответа.

Запуск:
    python -m src.evaluation.eval_rag --limit 10                  # быстрая проверка: 10 вопросов каждого набора
    python -m src.evaluation.eval_rag                              # оба Gold Set + вопросы вне базы
    python -m src.evaluation.eval_rag --ragas --limit 20           # + метрики Ragas
    python -m src.evaluation.eval_rag --resume                     # продолжить после обрыва (лимиты API)
    python -m src.evaluation.eval_rag --ragas-only                 # Ragas по уже сохраненным ответам
Итог: data/evaluation/rag_eval.json и таблица в консоли.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.settings import BENCHMARKS_DIR, EVALUATION_DIR

GOLD_SETS = [BENCHMARKS_DIR / "gold_set_50.json", BENCHMARKS_DIR / "gold_set_hard.json"]
OUT_OF_SCOPE = BENCHMARKS_DIR / "out_of_scope.json"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
CONTEXT_SEPARATOR = "\n\n---\n\n"          # разделитель уроков в format_context


def set_name(path) -> str:
    return Path(path).stem.removeprefix("gold_set_")


def answers_path(path) -> Path:
    return EVALUATION_DIR / f"rag_answers_{set_name(path)}.jsonl"


# ==================== ПРОГОН ====================

def make_row(g: dict, resp, kind: str) -> dict:
    relevant = set(g.get("relevant_lesson_ids") or ([g["lesson_id"]] if g.get("lesson_id") else []))
    ctx_ids = [s.lesson_id for s in resp.context_sources]
    return {
        "id": g["id"], "kind": kind, "question": g["question"], "filter": g.get("filter"),
        "ground_truth": g.get("ground_truth_answer", ""), "relevant_lesson_ids": sorted(relevant),
        "answer": resp.answer, "found": resp.found, "standalone_question": resp.standalone_question,
        "cited_ids": resp.cited_ids, "context_ids": ctx_ids, "invalid_citations": resp.invalid_citations,
        "cites_relevant": bool(relevant & set(resp.cited_ids)), "context_relevant": bool(relevant & set(ctx_ids)),
        "contexts": [c for c in resp.context_text.split(CONTEXT_SEPARATOR) if c.strip()],
        "usage": resp.usage, "latency_s": resp.timings_ms.get("total_ms", 0) / 1000, "model": resp.model,
    }


def run_set(rag, path: Path, kind: str, limit: Optional[int], resume: bool, sleep: float) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)[:limit] if limit else json.load(f)
    out = answers_path(path)
    done = {}
    if resume and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[str(r["id"])] = r
    rows = []
    mode = "a" if resume else "w"
    with open(out, mode, encoding="utf-8") as f:
        for i, g in enumerate(items, 1):
            if str(g["id"]) in done:
                rows.append(done[str(g["id"])])
                continue
            try:
                resp = rag.ask(g["question"], filters=g.get("filter") or None)
            except Exception as exc:                  # лимиты, сеть: вопрос пропускается, можно --resume
                print(f"  [{i}/{len(items)}] {g['id']}: ОШИБКА {type(exc).__name__}: {str(exc)[:150]}", flush=True)
                continue
            row = make_row(g, resp, kind)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            mark = ("+" if row["cites_relevant"] else "-") if kind == "gold" else ("+" if not row["found"] else "-")
            print(f"  [{i}/{len(items)}] {mark} {g['id']}: {'ответ' if row['found'] else 'отказ'}, "
                  f"ссылки {row['cited_ids']}{' НЕВЕРНЫЕ ' + str(row['invalid_citations']) if row['invalid_citations'] else ''}"
                  f", {row['latency_s']:.1f} с", flush=True)
            if sleep:
                time.sleep(sleep)
    return rows


def load_rows(path: Path) -> List[dict]:
    p = answers_path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


# ==================== МЕТРИКИ ====================

def mean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows: List[dict], kind: str) -> dict:
    if not rows:
        return {"n": 0}
    base = {
        "n": len(rows),
        "answered": mean([r["found"] for r in rows]),
        "invalid_citations": mean([bool(r["invalid_citations"]) for r in rows]),
        "uncited_answers": mean([r["found"] and not r["cited_ids"] for r in rows]),
        "latency_s": mean([r["latency_s"] for r in rows]),
        "tokens": mean([r["usage"].get("total_tokens", 0) for r in rows]),
    }
    if kind == "gold":
        base.update(
            cites_relevant=mean([r["cites_relevant"] for r in rows]),
            context_relevant=mean([r["context_relevant"] for r in rows]),
            false_refusal=mean([(not r["found"]) and r["context_relevant"] for r in rows]),
        )
    else:
        base["correct_refusal"] = mean([not r["found"] for r in rows])
    return base


# ==================== RAGAS ====================

def bge_ragas_embedding(model=None):
    """bge-base-en-v1.5 проекта как эмбеддинги Ragas (для answer_relevancy). model — объект с embed_documents."""
    from ragas.embeddings.base import BaseRagasEmbedding

    if model is None:
        from src.retrieval.qdrant_store import BGEEmbeddings
        model = BGEEmbeddings()

    class BGERagasEmbedding(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs):
            return list(model.embed_documents([text])[0])

        async def aembed_text(self, text: str, **kwargs):
            return self.embed_text(text)

    return BGERagasEmbedding()


def ragas_judge(model: str):
    """Mistral как судья Ragas: OpenAI-совместимый клиент + instructor (JSON-режим)."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    from src.rag.llm import load_dotenv

    load_dotenv()
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise SystemExit("Для Ragas нужен MISTRAL_API_KEY")
    client = AsyncOpenAI(api_key=key, base_url=MISTRAL_BASE_URL, max_retries=8, timeout=120)
    return llm_factory(model, provider="openai", client=client, temperature=0.0, max_tokens=2048)


def run_ragas(rows: List[dict], judge_model: str, sleep: float) -> Dict[str, float]:
    """Ragas-метрики по сохраненным ответам (последовательно, чтобы не упираться в лимиты API)."""
    import asyncio

    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness

    llm = ragas_judge(judge_model)
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=bge_ragas_embedding()),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    async def score(row: dict) -> Dict[str, float]:
        args = {
            "faithfulness": dict(user_input=row["question"], response=row["answer"], retrieved_contexts=row["contexts"]),
            "answer_relevancy": dict(user_input=row["question"], response=row["answer"]),
            "context_precision": dict(user_input=row["question"], reference=row["ground_truth"],
                                      retrieved_contexts=row["contexts"]),
            "context_recall": dict(user_input=row["question"], retrieved_contexts=row["contexts"],
                                   reference=row["ground_truth"]),
        }
        out = {}
        for name, metric in metrics.items():
            if name in ("faithfulness", "answer_relevancy") and not row["found"]:
                continue                       # отказ: нечего проверять на верность / релевантность
            if not row["contexts"]:
                continue
            try:
                out[name] = float((await metric.ascore(**args[name])).value)
            except Exception as exc:
                print(f"    ragas {name} {row['id']}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            if sleep:
                await asyncio.sleep(sleep)
        return out

    async def run_all():
        results = []
        for i, row in enumerate(rows, 1):
            r = await score(row)
            row["ragas"] = r
            results.append(r)
            print(f"  ragas [{i}/{len(rows)}] {row['id']}: " + ", ".join(f"{k}={v:.2f}" for k, v in r.items()), flush=True)
        return results

    results = asyncio.run(run_all())
    return {name: mean([r.get(name) for r in results]) for name in metrics}


# ==================== ВЫВОД ====================

def print_table(summary: Dict[str, dict]) -> None:
    gold_cols = ["n", "answered", "cites_relevant", "context_relevant", "false_refusal", "invalid_citations",
                 "uncited_answers", "faithfulness", "answer_relevancy", "context_precision", "context_recall",
                 "latency_s", "tokens"]
    print("\n| Набор | " + " | ".join(gold_cols) + " |\n|:---|" + "---:|" * len(gold_cols))
    for name, s in summary.items():
        if name == "out_of_scope":
            continue
        cells = [("—" if (v := s.get(c)) is None or (isinstance(v, float) and math.isnan(v)) else
                  f"{v:.0f}" if c in ("n", "tokens") else f"{v:.2f}") for c in gold_cols]
        print(f"| {name} | " + " | ".join(cells) + " |")
    if "out_of_scope" in summary and summary["out_of_scope"].get("n"):
        s = summary["out_of_scope"]
        print(f"\nВопросы вне базы: n={s['n']}, верный отказ {s['correct_refusal']:.2f}, "
              f"ответы со ссылками вне контекста {s['invalid_citations']:.2f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", action="append", help="Gold Set JSON (можно несколько); по умолчанию оба")
    parser.add_argument("--limit", type=int, help="только первые N вопросов каждого набора")
    parser.add_argument("--no-oos", action="store_true", help="без вопросов вне базы")
    parser.add_argument("--resume", action="store_true", help="не задавать повторно вопросы из rag_answers_*.jsonl")
    parser.add_argument("--sleep", type=float, default=1.0, help="пауза между вопросами, с (лимиты бесплатного тарифа)")
    parser.add_argument("--ragas", action="store_true", help="посчитать метрики Ragas")
    parser.add_argument("--ragas-only", action="store_true", help="только Ragas по сохраненным ответам")
    parser.add_argument("--judge-model", default=os.getenv("MISTRAL_JUDGE_MODEL", "mistral-small-latest"))
    parser.add_argument("--out", default=str(EVALUATION_DIR / "rag_eval.json"))
    args = parser.parse_args()

    gold_paths = [Path(p) for p in (args.gold or GOLD_SETS)]
    runs = [(p, "gold") for p in gold_paths] + ([] if args.no_oos else [(OUT_OF_SCOPE, "oos")])

    rows_by_set: Dict[str, List[dict]] = {}
    if args.ragas_only:
        for p, _ in runs:
            rows_by_set[set_name(p)] = load_rows(p)[: args.limit] if args.limit else load_rows(p)
    else:
        from src.rag.chain import HoustonRAG

        rag = HoustonRAG.from_defaults()
        for p, kind in runs:
            print(f"\n=== {p.name} ===")
            rows_by_set[set_name(p)] = run_set(rag, p, kind, args.limit, args.resume, args.sleep)

    summary = {}
    for p, kind in runs:
        name = set_name(p)
        summary[name] = summarize(rows_by_set.get(name, []), kind)
        if (args.ragas or args.ragas_only) and kind == "gold" and rows_by_set.get(name):
            print(f"\n=== Ragas: {p.name} (судья {args.judge_model}) ===")
            summary[name].update(run_ragas(rows_by_set[name], args.judge_model, sleep=args.sleep / 2))
            with open(answers_path(p), "w", encoding="utf-8") as f:     # сохранить оценки Ragas по вопросам
                for r in rows_by_set[name]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print_table(summary)
    model = next((r["model"] for rows in rows_by_set.values() for r in rows), "")
    report = {"model": model, "judge_model": args.judge_model if (args.ragas or args.ragas_only) else None,
              "limit": args.limit, "summary": summary}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nОтветы: data/evaluation/rag_answers_*.jsonl; сводка: {args.out}")


if __name__ == "__main__":
    main()
