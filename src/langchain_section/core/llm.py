from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.langchain_section.config.settings import settings


@lru_cache(maxsize=1)
def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """Retorna cliente LLM"""
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        temperature=temperature or settings.DEFAULT_TEMPERATURE,
        max_retries=settings.MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Retorna el cliente de embeddings"""
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
