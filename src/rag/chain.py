"""RAG-цепочка Houston (LangChain LCEL).

    {"question", "filters"?, "chat_history"?}
      -> condense:  если есть история — LLM переписывает уточняющий вопрос в самостоятельный
      -> retrieve:  HoustonRetriever (обработка запроса, гибрид, реранкер, отбор контекста)
      -> context:   текст уроков с метками [LLIS #id] (format_context)
      -> generate:  ANSWER_PROMPT | LLM (Groq, Mistral, ... — см. llm.py); при пустом контексте LLM не вызывается
      -> finalize:  проверка ссылок, список источников, токены, время -> RAGResponse

Использование:
    from src.rag.chain import HoustonRAG
    rag = HoustonRAG.from_defaults()                     # HoustonRetriever + LLM из .env (LLM_PROVIDER)
    resp = rag.ask("Why did the Mars Climate Orbiter fail?")
    print(resp.to_markdown())                            # ответ + Sources
    rag.chain.invoke({"question": "...", "filters": {"org_code": "JPL"}})   # та же цепочка как Runnable
"""

import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from src.rag.citations import RAGResponse, process_answer
from src.rag.llm import get_llm, model_name, resolve_settings
from src.rag.prompts import ANSWER_PROMPT, CONDENSE_PROMPT, NOT_FOUND_ANSWER

HISTORY_WINDOW = 6      # сколько последних сообщений истории (вопросы + ответы) видит модель


def _usage(msg) -> Dict[str, int]:
    meta = getattr(msg, "usage_metadata", None) or {}
    return {k: int(meta.get(k, 0)) for k in ("input_tokens", "output_tokens", "total_tokens")}


def _add_usage(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def _flatten(key: str):
    """{"a": 1, key: {"b": 2}} -> {"a": 1, "b": 2}: результат шага цепочки разворачивается на верхний уровень."""
    return RunnableLambda(lambda x: {**{k: v for k, v in x.items() if k != key}, **x[key]}, name=f"flatten_{key}")


def format_documents(docs: List[Any]) -> str:
    """Контекст из произвольных Document'ов (если ретривер — не HoustonRetriever)."""
    parts = []
    for d in docs:
        m = d.metadata
        parts.append(f"[LLIS #{m.get('lesson_id')}] {m.get('title', '')}\nURL: {m.get('url', '')}\n\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


class HoustonRAG:
    """RAG по базе NASA LLIS: поиск HoustonRetriever + генерация LLM (Groq / Mistral / ...) со ссылками на уроки."""

    def __init__(self, retriever, llm: Runnable, history_window: int = HISTORY_WINDOW,
                 model: Optional[str] = None):
        self.retriever = retriever
        self.llm = llm
        self.history_window = history_window
        self.model = model or model_name(llm)
        self.condense_chain = CONDENSE_PROMPT | llm
        self.answer_chain = ANSWER_PROMPT | llm
        self.chain = self._build_chain()

    @classmethod
    def from_defaults(cls, retriever=None, llm: Optional[Runnable] = None, **kwargs) -> "HoustonRAG":
        if llm is None:                        # сначала ключ API: без него не грузим модели поиска
            llm = get_llm()
            kwargs.setdefault("model", resolve_settings().label)
        if retriever is None:
            from src.retrieval.retriever import HoustonRetriever
            retriever = HoustonRetriever.from_config()
        return cls(retriever, llm, **kwargs)

    # ------------------------------------------------------------------ шаги цепочки
    def _history(self, x: dict) -> List[BaseMessage]:
        return list(x.get("chat_history") or [])[-self.history_window:]

    def _condense(self, x: dict) -> dict:
        history = self._history(x)
        if not history:
            return {"standalone_question": x["question"], "usage": {}, "condense_ms": 0.0}
        t0 = time.perf_counter()
        msg = self.condense_chain.invoke({"question": x["question"], "chat_history": history})
        text = StrOutputParser().invoke(msg).strip().strip('"').splitlines()
        question = next((line.strip() for line in text if line.strip()), x["question"])
        return {"standalone_question": question, "usage": _usage(msg),
                "condense_ms": (time.perf_counter() - t0) * 1000}

    def _retrieve(self, x: dict) -> dict:
        t0 = time.perf_counter()
        query, filters = x["standalone_question"], x.get("filters")
        if hasattr(self.retriever, "retrieve"):            # HoustonRetriever: все этапы и готовый текст контекста
            result = self.retriever.retrieve(query, filters=filters)
            docs, context, timings = result.documents, result.context_text, dict(result.timings)
        else:                                              # любой LangChain-ретривер
            docs = self.retriever.invoke(query)
            context, timings = format_documents(docs), {}
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        return {"documents": docs, "context": context, "retrieval_timings": timings}

    def _generate(self, x: dict) -> dict:
        if not x["documents"]:                              # нечего передать в LLM — честный отказ без вызова модели
            return {"raw_answer": NOT_FOUND_ANSWER, "answer_usage": {}, "llm_ms": 0.0}
        t0 = time.perf_counter()
        msg = self.answer_chain.invoke({"context": x["context"], "question": x["standalone_question"],
                                        "chat_history": self._history(x)})
        return {"raw_answer": StrOutputParser().invoke(msg), "answer_usage": _usage(msg),
                "llm_ms": (time.perf_counter() - t0) * 1000}

    def _finalize(self, x: dict) -> RAGResponse:
        checked = process_answer(x["raw_answer"], x["documents"])
        timings = {"condense_ms": x["condense"]["condense_ms"], **x["retrieval_timings"], "llm_ms": x["llm_ms"]}
        timings["total_ms"] = timings["condense_ms"] + timings["retrieval_ms"] + timings["llm_ms"]
        return RAGResponse(
            question=x["question"], standalone_question=x["standalone_question"], model=self.model,
            usage=_add_usage(x["condense"]["usage"], x["answer_usage"]), timings_ms=timings,
            context_text=x["context"], **checked,
        )

    def _build_chain(self) -> Runnable:
        return (
            RunnablePassthrough.assign(condense=RunnableLambda(self._condense, name="condense_question"))
            | RunnablePassthrough.assign(standalone_question=lambda x: x["condense"]["standalone_question"])
            | RunnablePassthrough.assign(retrieved=RunnableLambda(self._retrieve, name="retrieve"))
            | _flatten("retrieved")
            | RunnablePassthrough.assign(generated=RunnableLambda(self._generate, name="generate_answer"))
            | _flatten("generated")
            | RunnableLambda(self._finalize, name="check_citations")
        ).with_config(run_name="houston_rag")

    # ------------------------------------------------------------------ API
    def ask(self, question: str, filters: Optional[Dict[str, Any]] = None,
            chat_history: Optional[List[BaseMessage]] = None) -> RAGResponse:
        return self.chain.invoke({"question": question, "filters": filters, "chat_history": chat_history or []})
