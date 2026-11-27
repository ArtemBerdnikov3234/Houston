"""Командная строка Houston: вопрос -> ответ со ссылками на уроки NASA LLIS.

Один вопрос:
    python -m src.rag.ask "Why did the Mars Climate Orbiter fail?"
    python -m src.rag.ask "How to prevent hydrogen leaks?" --filter org_code=KSC --show-context
    python -m src.rag.ask "..." --json                 # полный ответ в JSON (источники, токены, время)

Диалог (с историей):
    python -m src.rag.ask --chat
    команды в диалоге: /sources — все найденные уроки последнего ответа, /context — текст контекста,
                       /reset — очистить историю, /save <файл>, /load <файл>, /exit

Нужен MISTRAL_API_KEY (переменная окружения или .env) и запущенный Qdrant с коллекцией nasa_llis.
"""

import argparse
import sys
from typing import Any, Dict, List, Optional

from src.rag.citations import RAGResponse


def parse_filters(items: List[str]) -> Optional[Dict[str, Any]]:
    """["org_code=KSC", "year=1999", "program=Mars Phoenix"] -> словарь фильтров."""
    out: Dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"Фильтр в формате поле=значение, получено: {item!r}")
        key, value = key.strip(), value.strip()
        out[key] = int(value) if key == "year" and value.isdigit() else value
    return out or None


def render(resp: RAGResponse, verbose: bool = False) -> str:
    lines = [resp.to_markdown()]
    if resp.standalone_question != resp.question:
        lines.insert(0, f"(вопрос для поиска: {resp.standalone_question})\n")
    if resp.invalid_citations:
        lines.append(f"\n[!] Удалены ссылки на уроки вне контекста: {', '.join(resp.invalid_citations)}")
    if verbose:
        t = resp.timings_ms
        lines.append(f"\n— {resp.model}; токены: {resp.usage.get('total_tokens', 0)}; "
                     f"поиск {t.get('retrieval_ms', 0):.0f} мс, LLM {t.get('llm_ms', 0):.0f} мс")
    return "\n".join(lines)


def render_context_sources(resp: RAGResponse) -> str:
    rows = []
    for s in resp.context_sources:
        mark = "*" if s.cited else " "
        rows.append(f" {mark} {s.retrieval_rank}. [{s.label}] {s.title[:80]} ({s.org_code}, {s.year}) "
                    f"секции: {', '.join(s.matched_sections)}")
    return "Найденные уроки (* — процитирован):\n" + "\n".join(rows) if rows else "Уроки не найдены."


def chat_loop(filters) -> None:
    from src.rag.memory import ConversationalRAG

    from src.rag.llm import MissingAPIKeyError

    print("Загрузка моделей поиска...")
    try:
        chat = ConversationalRAG.from_defaults()
    except MissingAPIKeyError as exc:
        raise SystemExit(str(exc))
    sid = "cli"
    print("Houston готов. Вопросы на английском; /exit — выход, /reset — новая тема, /sources, /context.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        cmd, _, arg = q.partition(" ")
        if cmd in ("/exit", "/quit"):
            break
        if cmd == "/reset":
            chat.reset(sid)
            print("История очищена.\n")
            continue
        if cmd in ("/sources", "/context"):
            last = chat.last_response.get(sid)
            print((render_context_sources(last) if cmd == "/sources" else last.context_text) if last else "Еще нет ответа.")
            print()
            continue
        if cmd == "/save" and arg:
            chat.save(arg, sid)
            print(f"История сохранена в {arg}\n")
            continue
        if cmd == "/load" and arg:
            chat.load(arg, sid)
            print(f"История загружена ({len(chat.messages(sid))} сообщений)\n")
            continue
        try:
            resp = chat.ask(q, session_id=sid, filters=filters)
        except Exception as exc:                       # сеть, лимиты API — не роняем диалог
            print(f"[ошибка] {type(exc).__name__}: {exc}\n")
            continue
        print("\nhouston> " + render(resp, verbose=True) + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Houston: ответы по базе NASA LLIS")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="диалог с историей")
    parser.add_argument("--filter", action="append", default=[], help="поле=значение, например org_code=KSC")
    parser.add_argument("--show-context", action="store_true", help="показать найденные уроки и текст контекста")
    parser.add_argument("--json", action="store_true", help="вывести полный ответ в JSON")
    args = parser.parse_args()
    filters = parse_filters(args.filter)

    if args.chat:
        chat_loop(filters)
        return
    if not args.question:
        parser.error("укажите вопрос или --chat")

    from src.rag.chain import HoustonRAG
    from src.rag.llm import MissingAPIKeyError

    try:
        rag = HoustonRAG.from_defaults()
    except MissingAPIKeyError as exc:
        raise SystemExit(str(exc))
    resp = rag.ask(args.question, filters=filters)
    if args.json:
        print(resp.model_dump_json(indent=2))
        return
    print(render(resp, verbose=True))
    if args.show_context:
        print("\n" + render_context_sources(resp) + "\n\n" + "=" * 70 + "\n" + resp.context_text)


if __name__ == "__main__":
    main()
