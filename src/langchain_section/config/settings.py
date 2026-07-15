import os

from dotenv import load_dotenv

load_dotenv()


class LangChainSettings:
    """Configuración"""

    # Modelos
    CHAT_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Parámetros LLM
    DEFAULT_TEMPERATURE: float = 0.7
    LOW_TEMPERATURE: float = 0.1
    MAX_RETRIES: int = 3

    # Rutas de persistencia
    SQLITE_DB_PATH: str = "data/chat_history.db"
    CHROMA_PATH: str = "./data/langchain_chroma"

    # RAG
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESUTS: int = 3

    # Costos Aproximados
    COST_INPUT_PER_NILLION: float = 0.15
    COST_OUTPUT_PER_NILLION: float = 0.60

    @classmethod
    def validate(cls) -> None:
        """Valida variables críticas"""

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPEN_API_KEY no está configurado",
                "Crea un archivo .env y pega tu APIkey",
            )


settings = LangChainSettings()


"""
Desde el punto de vista del desarrollo profesional, ¿por qué resulta mejor definir un 
archivo de configuración dedicado que contenga una clase validada (ej: settings.py) en 
vez de invocar a las variables del sistema (os.getenv) en múltiples scripts dispersos?
"""
"""
Para centralizar la parametrización de la app en un solo punto, facilitar modificaciones
rápidas de los modelos, evitar dependencias cruzadas y detectar tempranamente variables
críticas faltantes.
"""
