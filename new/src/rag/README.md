# Цепочка RAG и генерация ответов

- `llm.py` — подключение Mistral (`ChatMistralAI`), загрузка `.env`, повторы при 429 / 5xx.
- `prompts.py` — промпт ответа по контексту и промпт переформулировки уточняющего вопроса.
- `citations.py` — разбор и проверка ссылок `[LLIS #id]`, список источников, модель ответа `RAGResponse`.
- `chain.py` — `HoustonRAG`: LCEL-цепочка поверх `HoustonRetriever`.
- `memory.py` — `ConversationalRAG`: история диалога по сессиям.
- `ask.py` — командная строка (один вопрос или диалог).

## Схема

```text
{"question", "filters"?, "chat_history"?}
  -> condense   есть история -> CONDENSE_PROMPT | Mistral -> самостоятельный вопрос (иначе вопрос как есть)
  -> retrieve   HoustonRetriever: обработка запроса, гибрид, реранкер, отбор контекста (до 5 уроков)
  -> generate   ANSWER_PROMPT(контекст [LLIS #id] + последние 6 сообщений истории + вопрос) | Mistral
                пустой контекст -> отказ без вызова LLM
  -> check      ссылки: нормализация, удаление ссылок вне контекста; источники из метаданных; отказ
  -> RAGResponse(answer, found, sources, context_sources, invalid_citations, usage, timings_ms)
```

## Настройка

1. Ключ: https://console.mistral.ai → API Keys (бесплатного тарифа Experiment достаточно).
2. Скопируйте `.env.example` в `.env` в корне проекта и впишите `MISTRAL_API_KEY=...` (файл `.env` не коммитится). Или задайте переменную окружения: PowerShell `$env:MISTRAL_API_KEY="..."`, Linux/Mac `export MISTRAL_API_KEY=...`.
3. Необязательно: `MISTRAL_MODEL` (по умолчанию `mistral-small-latest`; `mistral-medium-latest` сильнее, но медленнее и быстрее расходует лимит), `MISTRAL_TEMPERATURE` (0.1), `MISTRAL_MAX_TOKENS` (1200).

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

rag = HoustonRAG.from_defaults()                       # модели поиска и Mistral
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
