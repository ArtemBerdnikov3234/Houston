import pytest
from langchain_core.documents import Document

from Houston.src.retrieval.config import BASELINE, RetrieverConfig
from src.retrieval.context import (ContextBlock, apply_threshold, build_blocks, fit_budget,
                                   format_context, select_hits,
                                   strip_chunk_header)
from src.retrieval.retriever import rrf_merge
from src.retrieval.search import SearchHit, build_filter


def hit(cid, lesson, score, rank, section="recommendation", rerank=None, text=None):
    return SearchHit(chunk_id=cid, text=text or f"[NASA LLIS #{lesson}] T\nSection: X\n\nbody {cid}",
                     metadata={"lesson_id": lesson, "lesson_ids": [lesson], "section": section,
                               "sections": [section], "title": f"Lesson {lesson}", "url": f"u/{lesson}",
                               "org_code": "KSC", "year": 2000, "program": ""},
                     score=score, rank=rank, rerank_score=rerank)


# ---------- конфигурация ----------

def test_config_validation_and_env(monkeypatch):
    with pytest.raises(ValueError):
        RetrieverConfig(mode="bm42")
    with pytest.raises(ValueError):
        RetrieverConfig(candidate_k=3, final_k=5)
    monkeypatch.setenv("HOUSTON_FINAL_K", "3")
    monkeypatch.setenv("HOUSTON_USE_RERANKER", "false")
    monkeypatch.setenv("HOUSTON_EXPANSION_SECTIONS", "recommendation,evidence")
    cfg = RetrieverConfig.from_env()
    assert cfg.final_k == 3 and cfg.use_reranker is False and cfg.expansion_sections == ("recommendation", "evidence")


# ---------- фильтры ----------

def test_build_filter_kinds():
    f = build_filter({"org_code": "KSC", "program": ["A", "B"], "year": {"gte": 1990, "lte": 1999}})
    kinds = [type(c.match).__name__ if c.match else "Range" for c in f.must]
    assert kinds == ["MatchValue", "MatchAny", "Range"]
    assert f.must[0].key == "metadata.org_code"
    assert build_filter(None) is None and build_filter({}) is None


# ---------- поиск ----------

@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_search_modes(searcher, mode):
    hits = searcher.search("hypergolic propellant spills and fires", k=5, mode=mode)
    assert 0 < len(hits) <= 5
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.text and h.metadata.get("lesson_id") for h in hits)


def test_search_filter_and_fusion(searcher):
    hits = searcher.search("propellant", k=10, filters={"org_code": "KSC"}, fusion="dbsf", prefetch_k=30)
    assert hits and all(h.metadata["org_code"] == "KSC" for h in hits)
    hits = searcher.search("propellant", k=10, filters={"year": {"gte": 1990, "lte": 1999}})
    assert all(1990 <= h.metadata["year"] <= 1999 for h in hits)


def test_rrf_merge_dedup_and_boost():
    a = [hit("x", "1", 0, 1), hit("y", "2", 0, 2), hit("z", "3", 0, 3)]
    b = [hit("z", "3", 0, 1), hit("w", "4", 0, 2)]
    merged = rrf_merge([a, b], limit=10)
    assert [h.chunk_id for h in merged][0] == "z"           # есть в обеих выдачах
    assert len({h.chunk_id for h in merged}) == len(merged) == 4


# ---------- отбор контекста ----------

def test_threshold_and_diversity():
    hits = [hit("a", "1", 0, 1, rerank=3), hit("b", "1", 0, 2, rerank=2), hit("c", "1", 0, 3, rerank=1),
            hit("d", "2", 0, 4, rerank=-9)]
    assert [h.chunk_id for h in apply_threshold(hits, -6, 1)] == ["a", "b", "c"]
    assert [h.chunk_id for h in apply_threshold([hits[3]], -6, 1)] == ["d"]        # keep_at_least
    hits.append(hit("e", "3", 0, 5, rerank=-9))
    assert [h.chunk_id for h in select_hits(hits, final_k=5, max_per_lesson=2, max_chunks=10)] == ["a", "b", "d", "e"]
    assert [h.chunk_id for h in select_hits(hits, final_k=2, max_per_lesson=3, max_chunks=10)] == ["a", "b", "c", "d"]
    assert [h.chunk_id for h in select_hits(hits, final_k=5, max_per_lesson=3, max_chunks=2)] == ["a", "b"]


def test_blocks_group_and_expand(lessons):
    lid = "2196"   # урок с секциями driving_event / lesson_learned / recommendation
    hits = [hit("2196:recommendation:0", lid, 1.0, 1, "recommendation", rerank=2.0,
                text="[NASA LLIS #2196] T\nSection: Recommendation\n\nfound text")]
    cfg = RetrieverConfig(expansion_max_chars=300, full_section_max_chars=0)
    [block] = build_blocks(hits, cfg, lessons)
    assert block.matched_sections == ["recommendation"]
    assert block.sections["recommendation"] == "found text"
    # короткая найденная секция подставляется целиком
    [full] = build_blocks(hits, cfg.with_(full_section_max_chars=10**6), lessons)
    assert full.sections["recommendation"] == lessons.section(lid, "recommendation")
    added = [s for s in block.sections if s != "recommendation"]
    assert added and all(len(block.sections[s]) <= 310 for s in added)
    assert list(block.sections) == [s for s in ("abstract", "driving_event", "lesson_learned", "recommendation",
                                                 "evidence", "references") if s in block.sections]
    assert "(added for context)" in block.text and "[LLIS #2196]" in format_context([block])
    doc = block.to_document()
    assert isinstance(doc, Document) and doc.metadata["lesson_id"] == lid


def test_budget_drops_added_sections_first():
    big = ContextBlock("1", "A", "u", 2000, "KSC", "", 1.0, 1, sections={"recommendation": "x" * 450},
                       matched_sections=["recommendation"])
    second = ContextBlock("2", "B", "u", 2000, "KSC", "", 0.5, 2,
                          sections={"driving_event": "y" * 4500, "recommendation": "z" * 450},
                          matched_sections=["recommendation"])
    out = fit_budget([big, second], budget_tokens=250)
    assert [b.lesson_id for b in out] == ["1", "2"]
    assert list(out[1].sections) == ["recommendation"]


def test_strip_header():
    assert strip_chunk_header("[NASA LLIS #1] Title\nSection: Abstract (part 1/2)\n\nBody") == "Body"


# ---------- полный ретривер ----------

def test_retriever_pipeline(retriever, gold_hard):
    g = next(x for x in gold_hard if x.get("filter"))
    res = retriever.retrieve(g["question"], filters=g["filter"])
    assert res.candidates and res.ranked and res.context
    assert all(h.rerank_score is not None for h in res.ranked)
    assert [h.rerank_score for h in res.ranked] == sorted((h.rerank_score for h in res.ranked), reverse=True)
    field, value = next(iter(g["filter"].items()))
    assert all(h.metadata[field] == value or value in str(h.metadata[field]) for h in res.candidates)
    assert len(res.context) <= retriever.config.final_k
    assert sum(b.tokens for b in res.context) <= retriever.config.context_token_budget
    assert {"search_ms", "rerank_ms", "total_ms"} <= set(res.timings)


def test_retriever_program_filter_is_normalized(retriever):
    res = retriever.retrieve("telescope radio amplifier failure", filters={"program": "Lunar Reconnaissance Orbiter"})
    assert set(res.filters["program"]) >= {"Lunar Reconnaissance Orbiter", "Lunar Reconnaissance Orbiter TWTA"}


def test_retriever_hints_are_soft(retriever):
    res = retriever.retrieve("At Goddard, where do anomalies seen in orbit get written up?")
    assert res.query.hint_filters == {"org_code": "GSFC"}
    assert res.used_hints
    assert any(h.metadata["org_code"] != "GSFC" for h in res.candidates)   # мягкий: другие центры остаются


def test_retriever_without_reranker(retriever, reranker):
    res = retriever.retrieve("hydraulic gauge rupture", config=BASELINE)
    assert reranker.calls == 0 and all(h.rerank_score is None for h in res.ranked)
    assert len(res.ranked) == 10


def test_langchain_interface(retriever):
    docs = retriever.invoke("battery thermal runaway", filters={"org_code": "JPL"})
    assert docs and all(isinstance(d, Document) for d in docs)
    assert all(d.metadata["org_code"] == "JPL" and d.metadata["source_label"].startswith("LLIS #") for d in docs)
