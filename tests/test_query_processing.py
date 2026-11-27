from Houston.src.retrieval.query_processing import (ProgramNormalizer, build_abbreviations, expand_abbreviations,
                                            extract_hint_filters, get_abbreviations, normalize_text, process_query)


def test_normalize_text():
    assert normalize_text("  What’s  the “NSI”—issue?​ ") == "What's the \"NSI\"-issue?"


def test_abbreviation_dictionary_from_data():
    abbrs = get_abbreviations()
    assert abbrs["KSC"] == "kennedy space center"
    assert abbrs["NSI"] == "nasa standard initiator"
    assert abbrs["ESD"] == "electrostatic discharge"          # слитное составное слово
    assert abbrs["FMEA"].startswith("failure mode")            # варианты Mode/Modes объединены
    assert "IT" not in abbrs and "NASA" not in abbrs


def test_build_abbreviations_rules():
    recs = [{"full_text": "The Solid Rocket Booster (SRB) failed. Solid Rocket Booster (SRB) again. "
                          "A Standing Review Board (SRB) met. Mars Climate Orbiter (MCO) lost."}]
    abbrs = build_abbreviations(recs, min_count=1, min_share=0.6)
    assert abbrs["SRB"] == "solid rocket booster"       # 2 из 3 определений
    assert abbrs["MCO"] == "mars climate orbiter"


def test_expand_both_directions():
    abbrs = {"NSI": "nasa standard initiator", "KSC": "kennedy space center"}
    text, added = expand_abbreviations("NSI failures at Kennedy Space Center", abbrs)
    assert text.startswith("NSI failures at Kennedy Space Center")
    assert ("NSI", "nasa standard initiator") in added and ("kennedy space center", "KSC") in added
    # уже раскрыто — повторно не добавляется
    _, added = expand_abbreviations("NASA Standard Initiator (NSI) test", abbrs)
    assert added == []


def test_hint_filters_centers_and_periods():
    assert extract_hint_filters("At Goddard, where are anomalies written up?") == {"org_code": "GSFC"}
    assert extract_hint_filters("KSC lessons from the 1990s")["year"] == {"gte": 1990, "lte": 1999}
    assert extract_hint_filters("issues in the '90s")["year"] == {"gte": 1990, "lte": 1999}
    assert extract_hint_filters("between 2005 and 2001")["year"] == {"gte": 2001, "lte": 2005}
    assert extract_hint_filters("since 2010, before 2015")["year"] == {"gte": 2010, "lte": 2014}
    assert extract_hint_filters("In NASA lesson 5437, what caused the fire?")["lesson_ids"] == ["5437"]
    assert extract_hint_filters("LLIS #1234 summary")["lesson_ids"] == ["1234"]


def test_no_false_hints():
    # одиночный год события, обычные слова "arc", "glenn" в нижнем регистре — не фильтры
    assert extract_hint_filters("What caused the loss of the 1999 Mars weather orbiter?") == {}
    assert extract_hint_filters("an electrical arc damaged the harness") == {}


def test_hints_on_gold_set_do_not_contradict_answers(gold_hard):
    """Подсказка-фильтр по центру никогда не должна противоречить центру правильного урока."""
    for g in gold_hard:
        hints = extract_hint_filters(g["question"])
        if "org_code" in hints:
            codes = hints["org_code"] if isinstance(hints["org_code"], list) else [hints["org_code"]]
            assert g["nasa_center"] in codes, g["question"]


def test_program_variants():
    norm = ProgramNormalizer(["Lunar Reconnaissance Orbiter", "Lunar Reconnaissance Orbiter TWTA",
                              "Mars Science Laboratory", "Mars Science Laboratory (MSL)", "Mars Phoenix",
                              "International Space Station (ISS)", "ISS program"])
    assert norm.variants("Lunar Reconnaissance Orbiter") == ["Lunar Reconnaissance Orbiter",
                                                            "Lunar Reconnaissance Orbiter TWTA"]
    assert "Mars Science Laboratory (MSL)" in norm.variants("MSL")
    assert "Mars Phoenix" not in norm.variants("Mars Science Laboratory")
    f = norm.expand_filter({"program": "ISS program", "org_code": "JSC"})
    assert "International Space Station (ISS)" in f["program"] and f["org_code"] == "JSC"


def test_process_query_dense_text_untouched():
    pq = process_query("How did the NSI  misfire?", abbreviations={"NSI": "nasa standard initiator"})
    assert pq.text == "How did the NSI misfire?"
    assert "nasa standard initiator" in pq.sparse_text
