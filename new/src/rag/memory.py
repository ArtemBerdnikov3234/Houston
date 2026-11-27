"""История диалога для RAG Houston.

ConversationalRAG хранит историю по session_id и передает ее в цепочку HoustonRAG:
- разные пользователи (сессии) не смешиваются;
- в модель уходят только последние history_window сообщений (окно задается в HoustonRAG, чтобы промпт
  не рос бесконечно);
- уточняющий вопрос ("What did NASA recommend after that?") сначала переписывается в самостоятельный
  (CONDENSE_PROMPT), и уже он идет в поиск — иначе поиск не понял бы, о чем "that";
- в историю записываются вопрос пользователя и ответ (со ссылками [LLIS #id], без списка источников);
- длина хранимой истории ограничена max_messages, историю сессии можно сохранить в JSON и загрузить.

Свое хранилище вместо RunnableWithMessageHistory / InMemoryChatMessageHistory: в langchain-core 1.6
они объявлены устаревшими (будут удалены в 2.0), а LangGraph ради истории — лишняя зависимость.
Сообщения — стандартные HumanMessage / AIMessage LangChain.

Пример:
    from src.rag.memory import ConversationalRAG
    chat = ConversationalRAG.from_defaults()
    chat.ask("Why did the Mars Climate Orbiter fail?", session_id="alice")
    chat.ask("What did NASA change afterwards?", session_id="alice")     # вопрос поймется в контексте
    chat.reset("alice")
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, messages_from_dict, messages_to_dict

from src.rag.chain import HoustonRAG
from src.rag.citations import RAGResponse

MAX_STORED_MESSAGES = 50


class ConversationalRAG:
    def __init__(self, rag: HoustonRAG, max_messages: int = MAX_STORED_MESSAGES):
        self.rag = rag
        self.max_messages = max_messages
        self.sessions: Dict[str, List[BaseMessage]] = {}
        self.last_response: Dict[str, RAGResponse] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_defaults(cls, **kwargs) -> "ConversationalRAG":
        return cls(HoustonRAG.from_defaults(**kwargs))

    def ask(self, question: str, session_id: str = "default", filters: Optional[Dict[str, Any]] = None) -> RAGResponse:
        history = self.messages(session_id)
        resp = self.rag.ask(question, filters=filters, chat_history=history)
        with self._lock:
            stored = self.sessions.setdefault(session_id, [])
            stored.extend([HumanMessage(question), AIMessage(resp.answer)])
            del stored[:-self.max_messages]
            self.last_response[session_id] = resp
        return resp

    def messages(self, session_id: str = "default") -> List[BaseMessage]:
        return list(self.sessions.get(session_id, []))

    def reset(self, session_id: str = "default") -> None:
        with self._lock:
            self.sessions.pop(session_id, None)
            self.last_response.pop(session_id, None)

    # --- сохранение ---
    def save(self, path: str, session_id: str = "default") -> None:
        data = {"session_id": session_id, "messages": messages_to_dict(self.messages(session_id))}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str, session_id: Optional[str] = None) -> str:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sid = session_id or data.get("session_id", "default")
        with self._lock:
            self.sessions[sid] = messages_from_dict(data["messages"])[-self.max_messages:]
        return sid
