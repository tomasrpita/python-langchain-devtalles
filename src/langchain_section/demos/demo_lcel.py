"""DEMO LCEL"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from src.langchain_section.core.llm import get_llm

llm = get_llm()


# código aquí
def demo_simple_chain() -> None:
    """Cadena Simple"""

    # ChatPromtTemplate (roel, contenido)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un experto {tema}. Responde de forma concisa, máximo tres oraciones",
            ),
            ("human", "{pregunta}"),
        ]
    )

    # StrOutParser
    parser = StrOutputParser()

    # Cadena (Chain)
    chain = prompt | llm | parser

    response = chain.invoke({"tema": "Python", "pregunta": "¿Qué es un decorador?"})

    print("Simple Chain")
    print(response)
    print()


def demo_steps_inspection() -> None:
    """Invoca cada componente"""
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Eres un asistente técnico."), ("human", "{pregunta}")]
    )

    parser = StrOutputParser()

    input_message = {"pregunta": "¿Qué es un API REST"}

    # Paso 1 prompt -> PromptValue
    messages = prompt.invoke(input_message)
    print("Paso 1 - Prompt output")
    print(f" Tipo: {type(messages).__name__}")
    print(f" Mensajes: {messages.to_messages}")

    # Paso 2 LLM -> AIMessage
    ai_message = llm.invoke(messages)
    print("Paso 2 - LLM output")
    print(f" Tipo: {type(ai_message).__name__}")
    print(f" AI_Message: {ai_message}")
    print(f" Mensajes: {ai_message.content[:100]}...")

    # Paso 3 AiMessage -> str (Parser extrae texto)
    text = parser.invoke(ai_message)
    print("Paso 3 - Parser output")
    print(f" Tipo: {type(text).__name__}")
    print(f" Mensajes: {text[:100]}...")
    print()


def demo_batch() -> None:
    """Bstch procesa varios inputs en PARALELO"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un asistente técnico evaluador,"
                " ante un comentario de usuario solo debes responder con una palabra"
                " POSITIVO, NEGATIVO o NEUTRO.",
            ),
            ("human", "{texto}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    inputs = [
        {"texto": "Me encanta este framework, es increíble."},
        {"texto": "El servidor estuvo caído 3 horas!! es inaceptable"},
        {"texto": "La versión 2.0 ya está disponible"},
        {"texto": "Perdí todos mis datos por un bug crítico."},
        {"texto": "La documentación es bastante clara."},
        {"texto": "Es una función."},
    ]

    results = chain.batch(inputs)

    print("BATCH PROCESSING:")
    for input_message, result in zip(inputs, results):
        print(f" [{result}] {input_message['texto']}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain LCEL - Fundamentos")
    print("=" * 60)
    # demo_simple_chain()
    # demo_steps_inspection()
    demo_batch()
