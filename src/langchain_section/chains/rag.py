from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    """Convierte lista de documents en texto para el prompt"""
    return "\n\n---\n\n".join(
        [
            f"[Fuente: {doc.metadata.get('source', 'desconocido')},"
            f" Página: {doc.metadata.get('page', 'N/A')}]\n {doc.page_content}"
            for doc in docs
        ]
    )
