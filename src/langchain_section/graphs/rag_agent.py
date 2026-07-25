from functools import partial

from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

from src.langchain_section.graphs.nodes import (
    decide_retrieval_path,
    node_analyze,
    node_generate,
    node_retrieve,
)
from src.langchain_section.graphs.state import RAGAgenticState


def build_rag_agent(vectorestore: VectorStore):
    """Construir y compilar RAG agéntico"""
    builder = StateGraph(RAGAgenticState)

    node_retrieve_with_vs = partial(node_retrieve, vectorstore=vectorestore)

    builder.add_node("analyze", node_analyze)
    builder.add_node("retrieve", node_retrieve_with_vs)
    builder.add_node("generate", node_generate)

    builder.add_edge(START, "analyze")

    builder.add_conditional_edges(
        "analyze",
        decide_retrieval_path,
        {"retrieve": "retrieve", "generate": "generate"},
    )

    builder.add_edge("retrieve", "generate")

    builder.add_edge("generate", END)

    return builder.compile()
