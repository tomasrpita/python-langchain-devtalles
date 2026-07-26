import uuid
from pathlib import Path

from langchain_core.vectorstores import VectorStore

from src.langchain_section.core.document_loader import load_directory, split_documents
from src.langchain_section.core.embeddings import get_or_create_vectorstore
from src.langchain_section.memory.base import BaseMemoryBackend
from src.langchain_section.memory.postgresql_memory import PostgreSQLMemoryBackend
from src.langchain_section.memory.sqlite_memory import SQLiteMemoryBackend

DOCUMENTS_DIR = Path("data/documents")
COLLECTION_NAME = "knwoledge_base"
CHROMA_PATH = "./data/chromadb_knwoledge"


def setup_vectorstore() -> VectorStore | None:
    """Carga o inicializa el vector store"""
    vectorstore = get_or_create_vectorstore(
        collection_name=COLLECTION_NAME, persist_path=CHROMA_PATH
    )

    count = vectorstore._collection.count()

    if count > 0:
        print(f" Base de conocimiento: {count} chunks indexados")
        return vectorstore

    print("Base de conocimiento vacía. Indexando documentos...")
    docs = load_directory(DOCUMENTS_DIR)

    if not docs:
        print(f"\n Agrega archivos .txt o .pdf en {DOCUMENTS_DIR}")
        return None

    chunks = split_documents(docs)
    vectorstore.add_documents(chunks)

    print(f"{len(chunks)} chunks indexados")
    return vectorstore


def setup_memory_backend() -> BaseMemoryBackend:
    """Devuelve una conexción a db"""
    try:
        backend = PostgreSQLMemoryBackend()
        backend.list_sessions()
        print("Memoria: PostgreSQL")
        return backend
    except Exception as e:
        print(f"❌ Postgre no disponible {e} ")
        print(" Usando SQLite como fallback de desarrollo")
        return SQLiteMemoryBackend()


def select_session(backend: BaseMemoryBackend) -> str:
    """Seleccionar sesión o crear una nueva"""
    exists_sessions = backend.list_sessions()

    print("\n")
    print("=" * 55)
    print("Gestión de sesiones")
    print("=" * 55)

    if exists_sessions:
        print(f"\nConversaciones guardadas ({len(exists_sessions)}): ")
        for index, session_id in enumerate(exists_sessions, 1):
            messages = backend.get_history(session_id).messages
            last_message = ""

            if messages:
                last_message = f" - último: '{messages[-1].content[:40]}...'"
            print(f" {index}. {session_id}{last_message}")

        print("\nOpciones:")
        print("  Enter o n -> Nueva conversación")
        print("  1,2,3... -> Retomar conversación existente")
        print("  ID -> Escribir un session_id específico")

        choice = input("\nElige: ").strip().lower()

        if not choice or choice == "n":
            session_id = str(uuid.uuid4())
            print(f"\nNueva sesión: {session_id}")
            return session_id

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(exists_sessions):
                session_id = exists_sessions[idx]
                messages = backend.get_history(session_id).messages
                print(f"\n Retomando sesión: {session_id}")
                print(f" {len(messages)} mensajes previos")
                return session_id
            else:
                print("Número inválido. Creando nueva sesión")

        else:
            # En tal caso que se ingrese el session_id
            session_id = choice
            messages = backend.get_history(session_id).messages
            if messages:
                print(f"\n Retomando sesión: {session_id}")
                print(f" {len(messages)} mensajes previos")
                return session_id
            else:
                print("Id de session no válido. Creando nueva sesión")

    session_id = str(uuid.uuid4())
    print(f"\nNueva sesión: {session_id}")
    return session_id


def load_history(backend: BaseMemoryBackend, session_id: str) -> list:
    """Carga mensajes previos de una sesión"""
    history = backend.get_history(session_id)
    return history.messages
