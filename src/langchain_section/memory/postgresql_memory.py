import os

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from sqlalchemy import create_engine, text

from src.langchain_section.memory.base import BaseMemoryBackend


class PostgreSQLMemoryBackend(BaseMemoryBackend):
    """Backend de memoria usando PostgreSQL"""

    def __init__(self, database_url: str | None = None):
        self._database_url = database_url or os.getenv("DATABSE_URL")

        if not self._database_url:
            raise ValueError(
                "DATABASE_URL no esta configurado\n"
                "agrega una url a tu .env\n"
                "postgresql://user:password@host:port/db"
            )

        if not self._database_url.startswith("postgresql"):
            raise ValueError("DATABASE_URL debe empezar con 'postgresql://' ")

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retorna el historial de mensajes para una sessión"""
        return SQLChatMessageHistory(
            session_id=session_id, connection=self._database_url
        )

    def clean_history(self, session_id: str) -> None:
        """Elimina el historial de usa sesión"""
        history = self.get_history(session_id=session_id)
        history.clear()

    def list_sessions(self) -> list[str]:
        """Lista todos los session_id disponibles"""
        engine = create_engine(self._database_url)  # type: ignore

        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT DISTINT session_id FROM message_store")
                )

            return [row[0] for row in result]

        except ImportError:
            return []

        except Exception as e:
            print(e)
            return []
