from abc import ABC, abstractmethod

from langchain_core.chat_history import BaseChatMessageHistory


class BaseMemoryBackend(ABC):
    """Contrato que debecumplir cualquier backend de memoria"""

    @abstractmethod
    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retorna el historial de mensajes para una sessión"""

    @abstractmethod
    def clean_history(self, session_id: str) -> None:
        """Elimina el historial de usa sesión"""

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """Lista todos los session_id disponibles"""
