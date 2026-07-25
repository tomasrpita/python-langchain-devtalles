from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class RAGAgenticState(TypedDict):
    "Estados"

    messages: Annotated[list, add_messages]
    question: str
    retrieved_docs: list[str]
    response: str
    needs_retrieval: bool
    sources: list[dict]
