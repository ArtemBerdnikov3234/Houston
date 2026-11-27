# Цепочка RAG и генерация ответов

- `llm.py` — подключение LLM (Groq по умолчанию, Mistral, OpenRouter, Ollama, любой OpenAI-совместимый API), загрузка `.env`, повторы при 429 / 5xx.
- `prompts.py` — промпт ответа по контексту и промпт переформулировки уточняющего вопроса.
- `citations.py` — разбор и проверка ссылок `[LLIS #id]`, список источников, модель ответа `RAGResponse`.
- `chain.py` — `HoustonRAG`: LCEL-цепочка поверх `HoustonRetriever`.
- `memory.py` — `ConversationalRAG`: история диалога по сессиям.
- `ask.py` — командная строка (один вопрос или диалог).

## Схема

```text
{"question", "filters"?, "chat_history"?}
  -> condense   есть история -> CONDENSE_PROMPT | LLM -> самостоятельный вопрос (иначе вопрос как есть)
  -> retrieve   HoustonRetriever: обработка запроса, гибрид, реранкер, отбор контекста (до 5 уроков)
  -> generate   ANSWER_PROMPT(контекст [LLIS #id] + последние 6 сообщений истории + вопрос) | LLM
                пустой контекст -> отказ без вызова LLM
  -> check      ссылки: нормализация, удаление ссылок вне контекста; источники из метаданных; отказ
  -> RAGResponse(answer, found, sources, context_sources, invalid_citations, usage, timings_ms)
```

## Настройка

1. Скопируйте `.env.example` в `.env` в корне проекта (файл `.env` не коммитится).
2. Впишите провайдера и ключ:

| `LLM_PROVIDER` | Ключ | Модель по умолчанию | Заметки |
| :--- | :--- | :--- | :--- |
| `groq` (по умолчанию) | https://console.groq.com/keys | `openai/gpt-oss-120b` | бесплатно; ~30 запросов, ~8 000 токенов в минуту, ~200 000 в день на модель |
| `mistral` | console.mistral.ai | `mistral-small-latest` | с 2026 г. API только с оплатой по факту |
| `openrouter` | openrouter.ai/keys | `openai/gpt-oss-120b:free` | бесплатные модели `:free`: 20 в минуту, 50 запросов в день |
| `ollama` | не нужен | `qwen2.5:7b-instruct` | локально, без лимитов; `ollama pull qwen2.5:7b-instruct` |
| `openai_compatible` | `LLM_API_KEY` | `LLM_MODEL` | любой сервис с протоколом OpenAI: задайте `LLM_BASE_URL` |

3. Необязательно: `LLM_MODEL`, `LLM_TEMPERATURE` (0.1), `LLM_MAX_TOKENS` (1000), `LLM_REASONING_EFFORT` (для gpt-oss: `low` по умолчанию — меньше токенов на рассуждения), `LLM_JUDGE_MODEL` (судья Ragas).
4. Для бесплатного Groq контекст уменьшен в `.env.example`: `HOUSTON_CONTEXT_TOKEN_BUDGET=3500`, `HOUSTON_FINAL_K=4` — иначе запрос (контекст + промпт + ответ) может не поместиться в лимит 8 000 токенов в минуту.

Ошибки ключа и оплаты не повторяются, а выводятся одной строкой с подсказкой (401 — неверный ключ, 402 — нет оплаты, 413 — слишком большой запрос и т.д.). Ответы 429 (лимит в минуту) повторяются с паузой до 60 с.

Нужны также запущенный Qdrant с коллекцией `nasa_llis` и модели поиска (см. `src/retrieval/README.md`).

## Запуск

```bash
python -m src.rag.ask "Why did the Mars Climate Orbiter fail?"
python -m src.rag.ask "How to prevent hydrogen leaks?" --filter org_code=KSC --show-context
python -m src.rag.ask "..." --json            # полный RAGResponse: источники, токены, время этапов
python -m src.rag.ask --chat                  # диалог
```

Команды диалога: `/sources` — все найденные уроки последнего ответа (`*` — процитированные), `/context` — текст, переданный в LLM, `/reset` — новая тема, `/save файл.json`, `/load файл.json`, `/exit`.

В коде:

```python
from src.rag.chain import HoustonRAG
from src.rag.memory import ConversationalRAG

rag = HoustonRAG.from_defaults()                       # модели поиска и LLM из .env
resp = rag.ask("Why did the Mars Climate Orbiter fail?", filters={"org_code": "JPL"})
print(resp.to_markdown())

chat = ConversationalRAG(rag)
chat.ask("Why did the Mars Climate Orbiter fail?", session_id="alice")
resp = chat.ask("What did NASA change afterwards?", session_id="alice")
print(resp.standalone_question)                       # вопрос, который реально ушел в поиск
```

## Правила ответа

- Только по контексту: внешние знания запрещены промптом; это проверяют вопросы вне базы (`data/benchmarks/out_of_scope.json`), в том числе о чужих миссиях, ответ на которые модель может знать сама.
- Ссылка `[LLIS #id]` после каждого утверждения. Ссылки на уроки, которых не было в контексте, удаляются из текста и попадают в `invalid_citations` (в CLI — предупреждение).
- Отказ — фиксированная фраза `NOT_FOUND_ANSWER`; по ней `found=False`, источники не показываются.
- Список источников и URL строит код по метаданным уроков, модель их не пишет.

## Оценка

`python -m src.evaluation.eval_rag` — см. `src/evaluation/README.md`. Тесты цепочки (`tests/test_rag.py`) работают на заглушке LLM без ключа и сети.
