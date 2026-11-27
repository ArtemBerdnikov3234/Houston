"""Веб-интерфейс Houston (Streamlit): вопрос -> ответ и найденные фрагменты NASA LLIS.

Запуск из корня репозитория (API должен работать на API_URL):
    streamlit run src/ui/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
CENTERS = ["Все центры", "KSC", "JPL", "JSC", "HQ", "MSFC", "GSFC", "ARC", "GRC", "LARC"]
EXAMPLES = [
    "What caused the failure of the analog hydraulic pressure gauge at the LETF Pump House?",
    "What lessons were learned from the loss of Mars Climate Orbiter?",
    "Why should spacecraft be tested as they will fly?",
]

st.set_page_config(page_title="Houston — NASA LLIS", page_icon="🚀", layout="wide")


@st.cache_data(ttl=10)
def api_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.ok else {"error": r.json().get("detail", r.text)}
    except requests.RequestException as exc:
        return {"error": str(exc)}


def ask_api(question: str, k: int, center: str) -> dict:
    filters = {} if center == CENTERS[0] else {"org_code": center}
    r = requests.post(f"{API_URL}/ask", json={"question": question, "k": k, "filters": filters}, timeout=180)
    r.raise_for_status()
    return r.json()


# ---------- боковая панель ----------
with st.sidebar:
    st.header("Настройки")
    k = st.slider("Сколько фрагментов показывать", 1, 10, 4)
    center = st.selectbox("Центр NASA", CENTERS)
    st.divider()
    health = api_health()
    if "error" in health:
        st.error("API недоступен")
        st.caption(health["error"])
    else:
        st.success(f"API работает · чанков в базе: {health['points']}")
        st.caption("RAG-цепочка подключена" if health["chain_connected"] else "Режим «только поиск» (LLM не подключена)")

# ---------- основная часть ----------
st.title("🚀 Houston")
st.caption("Поиск по базе инженерных уроков NASA (Lessons Learned). Вопросы лучше задавать на английском.")

if "question" not in st.session_state:
    st.session_state.question = ""

st.write("Примеры:")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.question = example

with st.form("ask"):
    question = st.text_area("Ваш вопрос", key="question", height=80)
    submitted = st.form_submit_button("Спросить")

if submitted:
    if len(question.strip()) < 3:
        st.warning("Введите вопрос.")
    else:
        with st.spinner("Ищу ответ…"):
            try:
                data = ask_api(question.strip(), k, center)
            except requests.HTTPError as exc:
                st.error(f"Ошибка API: {exc.response.json().get('detail', exc.response.text)}")
                st.stop()
            except requests.RequestException as exc:
                st.error(f"Не удалось связаться с API ({API_URL}): {exc}")
                st.stop()

        if data["mode"] == "rag":
            st.subheader("Ответ")
            st.write(data["answer"])
        else:
            st.info("LLM пока не подключена, поэтому ниже только найденные фрагменты NASA LLIS.")

        st.subheader("Найденные фрагменты")
        if not data["contexts"]:
            st.write("Ничего не найдено.")
        for i, c in enumerate(data["contexts"], 1):
            meta = " · ".join(str(x) for x in (c.get("org_code"), c.get("section"), c.get("year")) if x)
            title = c.get("title") or "Фрагмент"
            with st.expander(f"{i}. {title}" + (f"  ({meta})" if meta else ""), expanded=(i == 1)):
                st.write(c["text"])
                if c.get("url"):
                    st.markdown(f"[Открыть урок на llis.nasa.gov]({c['url']})")