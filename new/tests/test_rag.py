import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.rag.ask import parse_filters, render
from src.rag.chain import HoustonRAG
from src.rag.citations import clean_citations, extract_citations, is_refusal, process_answer
from src.rag.llm import MissingAPIKeyError, RetryableLLMError, get_chat_model, load_dotenv, make_resilient
from src.rag.memory import ConversationalRAG
from src.rag.prompts import ANSWER_PROMPT, CONDENSE_PROMPT, NOT_FOUND_ANSWER
from tests.fakes import FakeChatLLM


# ---------- ссылки ----------

def test_extract_and_normalize_citations():
    text = "A [LLIS #12]. B [LLIS #34, #12]; C [LLIS #56; LLIS #78] and [LLIS 90]. Page [LLIS #12, page 3]."
    assert extract_citations(text) == ["12", "34", "56", "78", "90"]
    cleaned, invalid = clean_citations(text, {"12", "34", "56"})
    assert invalid == ["78", "90"]
    assert "[LLIS #34] [LLIS #12]" in cleaned and "#78" not in cleaned and "LLIS 90" not in cleaned
    assert "and." in cleaned or "and ." not in cleaned          # пустая ссылка не оставляет " ."


def test_refusal_detection():
    assert is_refusal(NOT_FOUND_ANSWER)
    assert is_refusal("  " + NOT_FOUND_ANSWER.upper())
    assert not is_refusal("The gauge failed [LLIS #1].")


def test_process_answer_sources_from_metadata():
    from langchain_core.documents import Document
    docs = [Document(page_content="x", metadata={"lesson_id": "1", "title": "T1", "url": "u1", "rank": 1}),
            Document(page_content="y", metadata={"lesson_id": "2", "title": "T2", "url": "u2", "rank": 2})]
    out = process_answer("Fact [LLIS #2]. Other [LLIS #7].", docs)
    assert out["found"] and [s.lesson_id for s in out["sources"]] == ["2"]
    assert out["sources"][0].url == "u2" and out["invalid_citations"] == ["7"]
    assert [s.cited for s in out["context_sources"]] == [False, True]
    assert process_answer(NOT_FOUND_ANSWER, docs)["sources"] == []


# ---------- LLM ----------

def test_missing_key_message(monkeypatch, tmp_path):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr("src.rag.llm.PROJECT_ROOT", tmp_path)
    import src.rag.llm as llm_mod
    monkeypatch.setattr(llm_mod, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(MissingAPIKeyError):
        get_chat_model()


def test_dotenv(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nMISTRAL_MODEL='mistral-medium-latest'\nexport HOUSTON_X=1\n", encoding="utf-8")
    monkeypatch.delenv("MISTRAL_MODEL", raising=False)
    monkeypatch.setenv("HOUSTON_X", "keep")
    load_dotenv(env)
    import os
    assert os.environ["MISTRAL_MODEL"] == "mistral-medium-latest" and os.environ["HOUSTON_X"] == "keep"


def test_retry_on_rate_limit_only():
    import httpx

    class Flaky:
        def __init__(self, status, fails):
            self.status, self.fails, self.calls = status, fails, 0

        def invoke(self, messages, config=None):
            self.calls += 1
            if self.calls <= self.fails:
                req = httpx.Request("POST", "https://api.mistral.ai")
                raise httpx.HTTPStatusError("err", request=req, response=httpx.Response(self.status, request=req))
            return AIMessage(content="ok")

    ok = Flaky(429, 2)
    assert make_resilient(ok, attempts=4).invoke("hi").content == "ok" and ok.calls == 3
    auth = Flaky(401, 1)
    with pytest.raises(httpx.HTTPStatusError):
        make_resilient(auth, attempts=4).invoke("hi")
    assert auth.calls == 1
    always = Flaky(503, 10)
    with pytest.raises(RetryableLLMError):
        make_resilient(always, attempts=2).invoke("hi")


# ---------- промпты ----------

def test_prompts_render():
    msgs = ANSWER_PROMPT.invoke({"context": "[LLIS #1] T", "question": "Q?"}).to_messages()
    assert msgs[0].type == "system" and "[LLIS #1234]" in msgs[0].content and NOT_FOUND_ANSWER in msgs[0].content
    assert msgs[-1].content.endswith("Question: Q?")
    hist = [HumanMessage("Why did MCO fail?"), AIMessage("Units [LLIS #641].")]
    msgs = CONDENSE_PROMPT.invoke({"question": "And after that?", "chat_history": hist}).to_messages()
    assert len(msgs) == 4 and "Follow-up question: And after that?" in msgs[-1].content


# ---------- цепочка ----------

@pytest.fixture()
def rag(retriever):
    return HoustonRAG(retriever, FakeChatLLM(hallucinate_id="999999"), model="fake")


def test_chain_answer_with_sources(rag, gold_hard):
    g = gold_hard[0]
    resp = rag.ask(g["question"])
    assert resp.found and resp.sources and resp.context_sources
    ctx_ids = {s.lesson_id for s in resp.context_sources}
    assert set(resp.cited_ids) <= ctx_ids
    assert resp.invalid_citations == ["999999"] and "999999" not in resp.answer
    assert "[LLIS #" in resp.answer and "#{0}, #{0}".format(resp.cited_ids[0]) not in resp.answer   # дубль схлопнут
    assert resp.usage["total_tokens"] == 120 and resp.timings_ms["llm_ms"] >= 0
    assert "Sources:" in resp.to_markdown() and resp.sources[0].url.startswith("https://llis.nasa.gov/")
    json.loads(resp.model_dump_json())                       # сериализуется (context_text исключен)


def test_chain_no_context_skips_llm(retriever):
    llm = FakeChatLLM()

    class Empty:
        def invoke(self, q):
            return []
    resp = HoustonRAG(Empty(), llm, model="fake").ask("anything")
    assert not resp.found and resp.answer == NOT_FOUND_ANSWER and llm.calls == []


def test_chain_refusal(retriever):
    resp = HoustonRAG(retriever, FakeChatLLM(refuse=True), model="fake").ask("What is the capital of France?")
    assert not resp.found and resp.sources == [] and resp.context_sources


def test_chain_filters_passed(rag):
    resp = rag.ask("battery fire", filters={"org_code": "JPL"})
    assert resp.context_sources and all(s.org_code == "JPL" for s in resp.context_sources)


def test_chain_is_runnable(rag):
    out = rag.chain.batch([{"question": "hydrazine spill"}, {"question": "gauge rupture"}])
    assert len(out) == 2 and all(o.found for o in out)


# ---------- история ----------

def test_conversation_history_and_condense(retriever, tmp_path):
    llm = FakeChatLLM()
    chat = ConversationalRAG(HoustonRAG(retriever, llm, model="fake", history_window=4))
    first = chat.ask("Why did the hydraulic gauge rupture?", session_id="a")
    assert first.standalone_question == first.question               # первый вопрос не переписывается
    second = chat.ask("What was recommended after that?", session_id="a")
    assert second.standalone_question.startswith("What was recommended after that? (about lesson")
    assert len(chat.messages("a")) == 4 and chat.messages("a")[1].content == first.answer
    other = chat.ask("Why did the hydraulic gauge rupture?", session_id="b")   # сессии не смешиваются
    assert other.standalone_question == other.question
    # окно истории: в промпт ответа попадает не больше history_window сообщений
    for _ in range(3):
        chat.ask("And then?", session_id="a")
    answer_prompt = [c for c in llm.calls if "Context — NASA LLIS lessons" in c[-1]][-1]
    assert len(answer_prompt) - 2 <= 4
    path = tmp_path / "s.json"
    chat.save(str(path), "a")
    chat.reset("a")
    assert chat.messages("a") == []
    chat.load(str(path), "a")
    assert len(chat.messages("a")) == 10


# ---------- CLI ----------

def test_cli_helpers(rag):
    assert parse_filters(["org_code=KSC", "year=1999"]) == {"org_code": "KSC", "year": 1999}
    with pytest.raises(SystemExit):
        parse_filters(["oops"])
    text = render(rag.ask("gauge rupture"), verbose=True)
    assert "Sources:" in text and "Удалены ссылки" in text


# ---------- оценка ----------

def test_eval_rag_rows_and_summary(rag, gold_hard):
    from src.evaluation.eval_rag import make_row, summarize
    g = gold_hard[0]
    row = make_row(g, rag.ask(g["question"]), "gold")
    assert row["contexts"] and set(row["cited_ids"]) <= set(row["context_ids"])
    assert row["cites_relevant"] == bool(set(row["relevant_lesson_ids"]) & set(row["cited_ids"]))
    s = summarize([row, {**row, "found": False, "cited_ids": [], "invalid_citations": []}], "gold")
    assert s["n"] == 2 and s["answered"] == 0.5 and s["false_refusal"] == (0.5 if row["context_relevant"] else 0.0)
    oos = summarize([{**row, "found": False}], "oos")
    assert oos["correct_refusal"] == 1.0
