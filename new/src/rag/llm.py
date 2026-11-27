"""Подключение LLM Mistral для Houston.

Настройки (переменные окружения или файл .env в корне проекта, см. .env.example):
    MISTRAL_API_KEY      — ключ API (console.mistral.ai -> API Keys), обязателен;
    MISTRAL_MODEL        — модель генерации ответов, по умолчанию mistral-small-latest;
    MISTRAL_TEMPERATURE  — по умолчанию 0.1 (ответы по фактам, без фантазии);
    MISTRAL_MAX_TOKENS   — лимит длины ответа, по умолчанию 1200;
    MISTRAL_TIMEOUT      — таймаут запроса в секундах, по умолчанию 120.

Повторы: ChatMistralAI сам повторяет запрос только при сетевых ошибках. Ответы 429 (лимит
запросов, частый на бесплатном тарифе) и 5xx здесь превращаются в RetryableLLMError и
повторяются с экспоненциальной паузой; 401/400 и прочие ошибки не повторяются.

Пример:
    from src.rag.llm import get_llm
    llm = get_llm()                              # Runnable: messages -> AIMessage
    print(llm.invoke("Say hello").content)
"""

import os
from typing import Optional

from langchain_core.runnables import Runnable, RunnableLambda

from src.settings import PROJECT_ROOT

DEFAULT_MODEL = "mistral-small-latest"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class RetryableLLMError(RuntimeError):
    """Временная ошибка API (лимит запросов, перегрузка) — запрос можно повторить."""


class MissingAPIKeyError(RuntimeError):
    pass


def load_dotenv(path=PROJECT_ROOT / ".env") -> None:
    """Минимальная загрузка .env (KEY=VALUE, # — комментарий); уже заданные переменные не перезаписываются."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def make_resilient(llm: Runnable, attempts: int = 6) -> Runnable:
    """Оборачивает чат-модель: повтор при 429/5xx с экспоненциальной паузой (1, 2, 4 ... до 60 с)."""
    import httpx

    def call(messages, config=None):
        try:
            return llm.invoke(messages, config)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code in RETRY_STATUS:
                raise RetryableLLMError(str(exc)) from exc
            raise

    return RunnableLambda(call, name="mistral").with_retry(
        retry_if_exception_type=(RetryableLLMError,), wait_exponential_jitter=True, stop_after_attempt=attempts)


def get_chat_model(model: Optional[str] = None, temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None, **kwargs):
    """Голая ChatMistralAI (без повторов 429) — для with_structured_output, Ragas и т.п."""
    from langchain_mistralai import ChatMistralAI

    load_dotenv()
    if not os.getenv("MISTRAL_API_KEY"):
        raise MissingAPIKeyError(
            "Не задан MISTRAL_API_KEY. Получите ключ на https://console.mistral.ai (API Keys) и задайте его "
            "переменной окружения или строкой MISTRAL_API_KEY=... в файле .env в корне проекта.")
    return ChatMistralAI(
        model=model or os.getenv("MISTRAL_MODEL", DEFAULT_MODEL),
        temperature=float(temperature if temperature is not None else os.getenv("MISTRAL_TEMPERATURE", 0.1)),
        max_tokens=int(max_tokens or os.getenv("MISTRAL_MAX_TOKENS", 1200)),
        timeout=int(os.getenv("MISTRAL_TIMEOUT", 120)),
        max_retries=3,
        **kwargs,
    )


def get_llm(model: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None,
            attempts: int = 6) -> Runnable:
    """ChatMistralAI с повторами при 429/5xx. Возвращает Runnable: messages -> AIMessage."""
    return make_resilient(get_chat_model(model, temperature, max_tokens), attempts=attempts)


def model_name(llm) -> str:
    """Название модели для отчетов (работает и для обертки, и для фейковой модели в тестах)."""
    load_dotenv()
    return getattr(llm, "model", None) or os.getenv("MISTRAL_MODEL", DEFAULT_MODEL)
