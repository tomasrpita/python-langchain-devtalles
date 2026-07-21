from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def load_file(file_path: Path) -> list[Document]:
    """Carge un archivo y retorna una lita de Documents de Langchain"""

    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado {file_path}")

    extension = file_path.suffix.lower

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extensión de Archivo {extension} so soportada "
            f"Usa: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    print(f" Cargando: {file_path.name}", end="")

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_name"] = file_path.name
            doc.metadata["file_type"] = "pdf"

    elif extension == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_name"] = file_path.name
            doc.metadata["file_type"] = "txt"

    print(f"{len(docs)} sección/es")

    return docs
