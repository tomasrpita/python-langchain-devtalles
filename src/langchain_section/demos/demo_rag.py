from pathlib import Path

from src.langchain_section.core.document_loader import load_directory, split_documents
from src.langchain_section.core.embeddings import get_or_create_vectorstore

DOCUMENTS_DIR = Path("data/documents")
COLLECTION_NAME = "demo_rag_langchain"
CHOROMA_PATH = "./data/chromadb_rag_demo"


def index_documents() -> tuple:
    """Carga archivos reales desde disco y divide en chunks"""

    docs = load_directory(DOCUMENTS_DIR)

    if not docs:
        return None, 0  # (vectorstore, num_chunks)

    chunks = split_documents(docs)

    if not chunks:
        print("❌ No se generaron chunks, Los archivos podrían estar vacíos.")
        return None, 0

    vectorstore = get_or_create_vectorstore(
        collection_name=COLLECTION_NAME, persist_path=CHOROMA_PATH
    )

    current_count = vectorstore._collection.count()

    if current_count > 0:
        print(f"\n Ya hay {current_count} chunks indexados")
        answer = (
            input(
                f"¿Deseas eliminar los {current_count} chunks existentes "
                "y reemplazarlos? (s/N): "
            )
            .strip()
            .lower()
        )

        if answer == "s":
            vectorstore._client.delete_collection(COLLECTION_NAME)
            vectorstore = get_or_create_vectorstore(
                collection_name=COLLECTION_NAME, persist_path=CHOROMA_PATH
            )

            vectorstore.add_documents(chunks)
            print(f"Reindexado: {len(chunks)} chunks en ChormaDB")

        else:
            print(f"Usando indice existente ({len(chunks)} chunks)")

    else:
        vectorstore.add_documents(chunks)
        print(f"{len(chunks)} chunks indexados en ChomaDB")

    return vectorstore, vectorstore._collection.count()


if __name__ == "__main__":
    index_documents()
