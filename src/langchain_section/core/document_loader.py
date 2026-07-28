from pathlib import Path

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchain_section.config.settings import settings

SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def load_file(file_path: Path) -> list[Document]:
    """Carge un archivo y retorna una lita de Documents de Langchain"""

    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado {file_path}")

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extensión de Archivo {extension} so soportada "
            f"Usa: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    print(f" Cargando: {file_path.name}", end="")

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))
        # Sera un doc por cada pagina del PDF
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_name"] = file_path.name
            doc.metadata["file_type"] = "pdf"

    elif extension == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
        # En este caso sera un solo doc, pero se puede dividir en chunks luego
        docs = loader.load()
        # pero devuelve una lista de docs para mantener la consistencia, en este
        # caso pudimos haber agregado metadata con
        # docs[0].metadata["file_name"] = file_path.name
        for doc in docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_name"] = file_path.name
            doc.metadata["file_type"] = "txt"

    print(f" {len(docs)} sección/es")

    return docs


def load_directory(directory_path: Path) -> list[Document]:
    """Carga todos los archivos soportados en una carpeta"""
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Carpeta creada: {directory_path}")
        print("Agrega archivos .txt o .pdf y vuelve a ejecutar")
        return []

    all_files = []

    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(directory_path.glob(f"*{ext}"))
        all_files.extend(directory_path.glob(f"*{ext.upper()}"))

    all_files = list(set(all_files))

    if not all_files:
        print(f" No se encontraron archivos en: {directory_path}")
        print("Agrega archivos .txt o .pdf y vuelve a ejecutar")
        return []

    print(f"Archivos encontrados en {directory_path}: ")
    all_docs = []
    errors = []

    for file_path in sorted(all_files):
        try:
            docs = load_file(file_path)
            all_docs.extend(docs)
        except Exception as e:
            errors.append((file_path.name, str(e)))
            print(f"Error cargando {file_path.name}: {e}")

    if errors:
        print(
            f"\n ❌ {len(errors)} archivos(s) con error, ✅ {len(all_docs)} documento(s) cargados."
        )
    else:
        print(
            f"\n ✅ {len(all_files)} archivo(s) cargados -> {len(all_docs)} sección/es totales"
        )

    return all_docs


def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Divide los docomentos en chunks para indexación"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)

    print(
        f"{len(docs)} sección/es -> {len(chunks)} chunks "
        f"(tamaño: ~{chunk_size or settings.CHUNK_SIZE} chars, "
        f"Overlap: {chunk_overlap or settings.CHUNK_OVERLAP})"
    )

    return chunks
