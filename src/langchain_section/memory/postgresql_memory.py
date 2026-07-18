import os
from urllib.parse import urlparse

import psycopg2
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from psycopg2 import sql
from sqlalchemy import create_engine, text

from src.langchain_section.memory.base import BaseMemoryBackend


class PostgreSQLMemoryBackend(BaseMemoryBackend):
    """Backend de memoria usando PostgreSQL"""

    def __init__(self, database_url: str | None = None):
        self._database_url = database_url or os.getenv("DATABASE_URL")

        if not self._database_url:
            raise ValueError(
                "DATABASE_URL no esta configurado\n"
                "agrega una url a tu .env\n"
                "postgresql://user:password@host:port/db"
            )

        if not self._database_url.startswith("postgresql"):
            raise ValueError("DATABASE_URL debe empezar con 'postgresql://' ")

        self._ensure_database_exists()

    def _ensure_database_exists(self) -> None:
        parsed_url = urlparse(self._database_url)
        database_name = parsed_url.path.lstrip("/") or "postgres"

        if not database_name:
            return

        admin_conn = psycopg2.connect(
            host=parsed_url.hostname,
            port=parsed_url.port or 5432,
            dbname="postgres",
            user=parsed_url.username,
            password=parsed_url.password,
        )
        admin_conn.autocommit = True

        try:
            with admin_conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (database_name,),
                )
                exists = cursor.fetchone()

                if not exists:
                    cursor.execute(
                        sql.SQL("CREATE DATABASE {}").format(
                            sql.Identifier(database_name)
                        )
                    )
        finally:
            admin_conn.close()

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
                    text("SELECT DISTINCT session_id FROM message_store")
                )

            return [row[0] for row in result]

        except ImportError:
            return []

        except Exception as e:
            print(e)
            return []
