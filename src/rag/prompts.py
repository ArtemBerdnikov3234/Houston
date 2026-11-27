"""Промпты RAG-цепочки Houston.

ANSWER_PROMPT    — генерация ответа по контексту уроков NASA LLIS со ссылками [LLIS #id].
CONDENSE_PROMPT  — переформулировка уточняющего вопроса из диалога в самостоятельный вопрос для поиска.

Промпты на английском: данные, эмбеддинги и вопросы англоязычные, так модель точнее держит
термины. Правила ответа:
- только по контексту, без внешних знаний;
- ссылка [LLIS #id] после каждого утверждения, только на уроки из контекста;
- если ответа в контексте нет — фиксированная фраза NOT_FOUND_ANSWER (по ней ответ распознается
  как отказ в коде и в оценке);
- структура: короткий прямой ответ, затем (если есть в контексте) что произошло, причина, урок,
  рекомендация NASA; для фактических вопросов (число, название, документ) — коротко.
Список источников модель не пишет: его собирает код из метаданных (citations.py).
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

NOT_FOUND_ANSWER = "I could not find the answer to this question in the NASA LLIS lessons that were retrieved."

SYSTEM_PROMPT = f"""You are Houston, an assistant for spacecraft and ground-systems engineers. You answer questions \
using lessons from the NASA Lessons Learned Information System (LLIS).

Rules:
1. Use ONLY the lessons in the context. Do not add facts from your own knowledge, even if you are sure they are true.
2. Cite every factual statement with the identifier of the lesson that supports it, written exactly as in the \
context header, for example [LLIS #1234], with plain square brackets. Put the citation at the end of the sentence. If several lessons support \
a statement, cite each: [LLIS #1234] [LLIS #5678]. Never cite an identifier that is not in the context.
3. If the context does not contain the answer, reply with exactly this sentence and nothing else: "{NOT_FOUND_ANSWER}"
   If the context answers only part of the question, answer that part and say clearly what is not covered.
4. Format:
   - Start with a direct answer in 1-3 sentences.
   - For questions about incidents, failures or practices, then add only the parts the context supports, as short \
paragraphs with bold labels: **What happened**, **Root cause**, **Lesson learned**, **NASA recommendation**.
   - For questions about a single fact (a number, a name, a document, a date), answer briefly without these sections.
   - When several lessons are relevant, say how they relate (same failure mode, different programs).
5. Keep numbers, units, part names, mission names and document numbers exactly as written in the context.
6. Sections marked "(added for context)" come from the same lesson and may be used.
7. Write in English. Do not write a list of sources or URLs at the end; it is added automatically."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "Context — NASA LLIS lessons:\n\n{context}\n\n---\n\nQuestion: {question}"),
])

CONDENSE_SYSTEM = """You rewrite a follow-up question from a conversation into a standalone search query for a \
database of NASA engineering lessons learned.

- Resolve references such as "it", "that mission", "this failure", "after that" using the conversation.
- Keep every specific name needed to understand the question: mission, program, component, NASA center, year.
- Do not answer the question and do not add facts that are not in the conversation.
- If the question is already standalone, return it unchanged.
- Output only the rewritten question in English, on one line, without quotes."""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONDENSE_SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "Follow-up question: {question}\n\nStandalone question:"),
])
