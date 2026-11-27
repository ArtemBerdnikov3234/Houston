# Поиск и Qdrant

- `qdrant_store.py` задаёт модели dense/sparse-векторов, схему коллекции, фильтры и фабрику `QdrantVectorStore`.
- `index_qdrant.py` читает подготовленные чанки и идемпотентно загружает их в Qdrant.
- `config.py` — все параметры конвейера поиска (`RetrieverConfig`), базовая линия `BASELINE`.
- `search.py` — семантический, BM25 и гибридный поиск через Query API Qdrant (`Searcher`, `SearchHit`, фильтры).
- `query_processing.py` — нормализация вопроса, аббревиатуры, подсказки-фильтры, варианты названий программ.
- `reranker.py` — Cross-Encoder `BAAI/bge-reranker-base`.
- `context.py` — отбор и сборка контекста для LLM (`ContextBlock`, `format_context`).
- `retriever.py` — `HoustonRetriever`: весь конвейер как LangChain `BaseRetriever`, CLI.

Перед индексацией запустите Qdrant командой `docker compose -f infra/docker-compose.yml up -d`, затем выполните `python -m src.retrieval.index_qdrant --recreate`.

## Как устроен поиск

Для каждого чанка используются два представления: dense-вектор BGE описывает смысл текста, sparse-вектор BM25 помогает находить точные слова и обозначения. Qdrant хранит оба представления вместе с текстом и метаданными. Гибридный режим объединяет результаты через RRF; фильтры позволяют ограничить выдачу, например, центром NASA или секцией урока.

`qdrant_store.py` содержит общие настройки коллекции `nasa_llis`, создание клиента и коллекции, обёртку модели BGE, генерацию стабильных ID точек и построение фильтров. Его функция `get_vector_store()` используется как индексацией, так и оценкой, чтобы они работали с одной конфигурацией.

`index_qdrant.py` читает `data/processed/llis_chunks.jsonl`, готовит документы и загружает их пакетами. Результат находится в Qdrant, а не в новой JSON-копии внутри этой папки. Стабильные идентификаторы позволяют обновлять существующие точки при повторной загрузке.

## Команды и настройки

Обычная загрузка: `python -m src.retrieval.index_qdrant`. Пробная загрузка первых 200 чанков: `python -m src.retrieval.index_qdrant --limit 200`. Параметр `--chunks` задаёт другой JSONL. Флаг `--recreate` удаляет существующую коллекцию и создаёт её заново, поэтому для обычного повторного запуска он не обязателен.

Адрес сервера задаётся через `QDRANT_URL` (по умолчанию `http://localhost:6333`), имя коллекции — через `QDRANT_COLLECTION`. Модели кэшируются в `.fastembed_cache/`; путь можно заменить переменной `FASTEMBED_CACHE_PATH`. Первый запуск моделей может потребовать скачивания файлов.


## Конвейер поиска HoustonRetriever

```text
вопрос
  -> query_processing: нормализация; BM25-запрос + расшифровки аббревиатур; подсказки (центр, период, № урока)
  -> явные фильтры; program -> все варианты написания
  -> search: dense top-50 + BM25 top-50 -> RRF -> 30 кандидатов
       (+ тот же поиск с подсказками-фильтрами, обе выдачи сливаются через RRF)
  -> reranker: bge-reranker-base переоценивает 30 кандидатов
  -> context: порог -> до 5 уроков по <= 3 фрагмента (всего <= 10) -> короткие секции целиком
              -> + Driving Event / Lesson Learned / Recommendation -> бюджет ~6000 токенов
  -> Documents (по одному на урок) / текст контекста с метками [LLIS #id]
```

Ручная проверка: `python -m src.retrieval.retriever "Why did the Mars Climate Orbiter fail?"`. Ключи: `--filter org_code=JPL` (можно несколько), `--no-rerank`, `--final-k 3`, `--show-context` (напечатать текст для LLM). Вывод показывает, как обработан запрос, top-10 после реранкера с исходным рангом гибрида, состав контекста и время каждого этапа.

В коде:

```python
from src.retrieval.retriever import HoustonRetriever
retriever = HoustonRetriever.from_config()                  # модели загружаются один раз
docs = retriever.invoke("...", filters={"program": "Mars Science Laboratory"})
res = retriever.retrieve("...")                             # все этапы: res.candidates, res.ranked, res.context, res.timings
res2 = retriever.retrieve("...", config=retriever.config.with_(use_reranker=False))   # другая конфигурация без перезагрузки моделей
```

### Параметры (`RetrieverConfig`)

| Параметр | По умолчанию | Смысл |
| :--- | :---: | :--- |
| `mode`, `fusion` | `hybrid`, `rrf` | режим первичного поиска (`dense` / `sparse` / `hybrid`) и слияние (`rrf` / `dbsf`) |
| `prefetch_k` | 50 | кандидатов из каждой ветви (dense, BM25) до слияния |
| `candidate_k` | 30 | кандидатов после слияния, идут в реранкер |
| `expand_abbreviations` | `True` | расшифровки аббревиатур в BM25-запросе |
| `auto_filters`, `auto_filter_min_hits` | `True`, 3 | мягкие подсказки-фильтры из текста; если с подсказкой найдено меньше 3 фрагментов, она игнорируется |
| `normalize_programs` | `True` | фильтр `program` расширяется на варианты написания |
| `use_reranker`, `rerank_model` | `True`, `BAAI/bge-reranker-base` | переранжирование; для быстрого варианта — `Xenova/ms-marco-MiniLM-L-6-v2` |
| `final_k` | 5 | уроков в контексте LLM |
| `max_per_lesson`, `max_chunks` | 3, 10 | фрагментов на урок и всего |
| `min_rerank_score`, `keep_at_least` | -6.0, 1 | порог логита реранкера; сколько лучших оставить в любом случае |
| `full_section_max_chars` | 2400 | найденная секция короче — подставляется целиком |
| `expand_sections`, `expansion_sections`, `expansion_max_chars` | `True`, DE/LL/Rec, 1200 | соседние секции урока и их лимит |
| `context_token_budget` | 6000 | приблизительный бюджет контекста (символы / 4.5) |

Любой параметр задается переменной окружения `HOUSTON_<ИМЯ>` (например, `HOUSTON_FINAL_K=4`, `HOUSTON_USE_RERANKER=false`) и читается через `RetrieverConfig.from_env()`. Значения `candidate_k`, `final_k` и `min_rerank_score` подбираются скриптом `python -m src.evaluation.tune_top_k`.

### Словарь аббревиатур

`data/processed/abbreviations.json` строится из `llis_full.json` при первом обращении (или командой `python -m src.retrieval.query_processing --rebuild`). Проверить обработку вопроса: `python -m src.retrieval.query_processing "NSI failures at KSC in the 1990s"`.
