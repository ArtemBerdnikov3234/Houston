"""Подключение LLM для Houston: Groq, Mistral, OpenRouter, Ollama или любой OpenAI-совместимый API.

Провайдер выбирается в .env (шаблон — .env.example) или переменными окружения:
    LLM_PROVIDER     groq (по умолчанию) | mistral | openrouter | ollama | openai_compatible
    LLM_API_KEY      ключ провайдера (для ollama не нужен). Для совместимости читаются и
                     GROQ_API_KEY / MISTRAL_API_KEY / OPENROUTER_API_KEY
    LLM_MODEL        модель; по умолчанию — из таблицы PROVIDERS
    LLM_BASE_URL     адрес API; обязателен только для openai_compatible
    LLM_TEMPERATURE  по умолчанию 0.1 (ответы по фактам)
    LLM_MAX_TOKENS   лимит длины ответа, по умолчанию 1000
    LLM_REASONING_EFFORT  low | medium | high — для reasoning-моделей (gpt-oss на Groq: по умолчанию low,
                     чтобы рассуждения не съедали лимит токенов в минуту); "none" — не передавать
    LLM_TIMEOUT      таймаут запроса, с (120)
    LLM_JUDGE_MODEL  модель-судья Ragas в eval_rag.py (на Groq по умолчанию openai/gpt-oss-20b)

Все провайдеры, кроме mistral, подключаются через ChatOpenAI (протокол OpenAI). Mistral — через
ChatMistralAI. Повторы: 429 (лимит запросов / токенов в минуту) и 5xx повторяются с паузой до 60 с;
ошибки ключа, оплаты и слишком большого запроса (400/401/402/403/404/413) не повторяются и
превращаются в LLMRequestError с понятным текстом.

Пример:
    from src.rag.llm import get_llm
    llm = get_llm()                              # Runnable: messages -> AIMessage
    print(llm.invoke("Say hello").content)
"""

import os
from dataclasses import dataclass
from typing import Optional

from langchain_core.runnables import Runnable, RunnableLambda

from src.settings import PROJECT_ROOT

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
HINTS = {
    400: "запрос отклонен: проверьте LLM_MODEL (есть ли такая модель у провайдера) и параметры",
    401: "неверный API-ключ: проверьте LLM_API_KEY в .env",
    402: "у аккаунта нет оплаты или бюджета на API у этого провайдера",
    403: "доступ запрещен: ключ без прав на модель, или провайдер недоступен из вашего региона",
    404: "модель или адрес API не найдены: проверьте LLM_MODEL и LLM_BASE_URL",
    413: "запрос больше лимита токенов провайдера: уменьшите контекст (HOUSTON_CONTEXT_TOKEN_BUDGET, "
         "HOUSTON_FINAL_K) или LLM_MAX_TOKENS",
}


@dataclass(frozen=True)
class Provider:
    base_url: Optional[str]
    default_model: str
    key_env: Optional[str]          # дополнительная переменная с ключом (кроме LLM_API_KEY)
    reasoning_effort: Optional[str] = None
    judge_model: Optional[str] = None   # модель-судья Ragas (на Groq — другая модель: у нее своя квота токенов)


PROVIDERS = {
    "groq": Provider("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY", "low",
                     judge_model="openai/gpt-oss-20b"),
    "mistral": Provider("https://api.mistral.ai/v1", "mistral-small-latest", "MISTRAL_API_KEY"),
    "openrouter": Provider("https://openrouter.ai/api/v1", "openai/gpt-oss-120b:free", "OPENROUTER_API_KEY"),
    "ollama": Provider("http://localhost:11434/v1", "qwen2.5:7b-instruct", None),
    "openai_compatible": Provider(None, "", None),
}


class RetryableLLMError(RuntimeError):
    """Временная ошибка API (лимит запросов, перегрузка) — запрос можно повторить."""


class LLMRequestError(RuntimeError):
    """Ошибка, которую повтор не исправит (ключ, оплата, модель, размер запроса)."""


class MissingAPIKeyError(LLMRequestError):
    pass


# ==================== НАСТРОЙКИ ====================

def load_dotenv(path=PROJECT_ROOT / ".env") -> None:
    """Минимальная загрузка .env (KEY=VALUE, # — комментарий); уже заданные переменные не перезаписываются."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    base_url: Optional[str]
    api_key: Optional[str]
    temperature: float
    max_tokens: int
    reasoning_effort: Optional[str]
    timeout: int

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def resolve_settings(model: Optional[str] = None, temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> LLMSettings:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in PROVIDERS:
        raise LLMRequestError(f"LLM_PROVIDER={provider!r}: ожидается одно из {', '.join(PROVIDERS)}")
    p = PROVIDERS[provider]
    key = os.getenv("LLM_API_KEY") or (os.getenv(p.key_env) if p.key_env else None)
    if provider == "ollama":
        key = key or "ollama"                         # Ollama ключ не проверяет, но клиент OpenAI его требует
    if not key:
        raise MissingAPIKeyError(
            f"Не задан ключ для провайдера {provider}: впишите LLM_API_KEY=... в файл .env в корне проекта "
            f"(шаблон — .env.example)" + (f" или задайте {p.key_env}" if p.key_env else ""))
    base_url = os.getenv("LLM_BASE_URL") or p.base_url
    if not base_url:
        raise LLMRequestError("Для LLM_PROVIDER=openai_compatible задайте LLM_BASE_URL")
    model = model or os.getenv("LLM_MODEL") or (os.getenv("MISTRAL_MODEL") if provider == "mistral" else None) \
        or p.default_model
    if not model:
        raise LLMRequestError("Задайте LLM_MODEL")
    effort = os.getenv("LLM_REASONING_EFFORT", p.reasoning_effort if "gpt-oss" in model else "") or None
    if effort and effort.lower() == "none":
        effort = None
    return LLMSettings(
        provider=provider, model=model, base_url=base_url, api_key=key,
        temperature=float(temperature if temperature is not None else os.getenv("LLM_TEMPERATURE", 0.1)),
        max_tokens=int(max_tokens or os.getenv("LLM_MAX_TOKENS", 1000)),
        reasoning_effort=effort, timeout=int(os.getenv("LLM_TIMEOUT", 120)),
    )


# ==================== ПОВТОРЫ И ОШИБКИ ====================

def _status_of(exc: Exception) -> Optional[int]:
    """HTTP-статус из ошибок httpx (ChatMistralAI) и openai (ChatOpenAI)."""
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    return int(status) if isinstance(status, int) else None


def make_resilient(llm: Runnable, attempts: int = 8, provider: str = "") -> Runnable:
    """Повтор при 429/5xx с экспоненциальной паузой (1, 2, 4 ... до 60 с); прочие HTTP-ошибки — LLMRequestError."""

    def call(messages, config=None):
        try:
            return llm.invoke(messages, config)
        except Exception as exc:
            status = _status_of(exc)
            if status is None:
                raise
            if status in RETRY_STATUS:
                raise RetryableLLMError(f"{provider} HTTP {status}: {str(exc)[:300]}") from exc
            hint = HINTS.get(status, "см. текст ошибки")
            raise LLMRequestError(f"{provider or 'LLM'} вернул HTTP {status}: {hint}.\nОтвет API: {str(exc)[:500]}") from exc

    return RunnableLambda(call, name="llm").with_retry(
        retry_if_exception_type=(RetryableLLMError,), wait_exponential_jitter=True, stop_after_attempt=attempts,
        exponential_jitter_params={"max": 60})          # паузы 1, 2, 4 ... 60 с (+ случайный сдвиг)


# ==================== МОДЕЛИ ====================

def get_chat_model(model: Optional[str] = None, temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None, settings: Optional[LLMSettings] = None, **kwargs):
    """Чат-модель выбранного провайдера (без повторов 429)."""
    s = settings or resolve_settings(model, temperature, max_tokens)
    if s.provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=s.model, api_key=s.api_key, temperature=s.temperature,
                             max_tokens=s.max_tokens, timeout=s.timeout, max_retries=3, **kwargs)
    from langchain_openai import ChatOpenAI
    extra = {"reasoning_effort": s.reasoning_effort} if s.reasoning_effort else {}
    return ChatOpenAI(model=s.model, base_url=s.base_url, api_key=s.api_key, temperature=s.temperature,
                      max_tokens=s.max_tokens, timeout=s.timeout, max_retries=2, **extra, **kwargs)


def get_llm(model: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None,
            attempts: int = 8) -> Runnable:
    """Модель с повторами при 429/5xx. Возвращает Runnable: messages -> AIMessage."""
    s = resolve_settings(model, temperature, max_tokens)
    return make_resilient(get_chat_model(settings=s), attempts=attempts, provider=s.provider)


def model_name(llm) -> str:
    """Название модели для отчетов: provider:model."""
    label = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    if label:
        return str(label)
    try:
        return resolve_settings().label
    except LLMRequestError:
        return "unknown"


def judge_model_default() -> str:
    """Модель-судья Ragas: LLM_JUDGE_MODEL, иначе модель-судья провайдера, иначе модель ответов."""
    load_dotenv()
    if os.getenv("LLM_JUDGE_MODEL"):
        return os.environ["LLM_JUDGE_MODEL"]
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    p = PROVIDERS.get(provider)
    return (p.judge_model if p and p.judge_model else None) or resolve_settings().model
