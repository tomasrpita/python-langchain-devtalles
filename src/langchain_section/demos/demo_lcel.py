"""DEMO LCEL"""

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.langchain_section.core.llm import get_llm

llm = get_llm(0.1)


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


def demo_stream() -> None:
    """Método streaming"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Explica conceptos técnicosde forma clara"),
            ("human", "Explica que es {concepto} en 2 párrafos"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    print("Streaming en tiempo REAL.")
    print("AI: ", end="", flush=True)

    # crea un generador en el cual iteramos
    for chunk in chain.stream({"concepto": "La ventana de contexto en LLMs"}):
        print(chunk, end="", flush=True)

    print()


def demo_passthrough() -> None:

    # Simula retriever
    def search_context(question: str) -> str:
        contexts = {
            "python": "Python fue creado por Guido Van Rossum en 1991.",
            "langchain": "LangChain es un framework para aplicaciones con LLMs.",
            "devtalles": "Una plataforma muy cool con instructores guapos.",
        }

        for (
            keywork,
            ctx,
        ) in contexts.items():
            if keywork.lower() in question.lower():
                return ctx

        return "No se encontro contexto relevante"

    # Coge una función y la convierte en un Runnable con LCEL
    retriever = RunnableLambda(search_context)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Response usando este contexto: '\n{context}'",
            ),
            ("human", "{question}"),
        ]
    )

    # RunnablePassthrough permite usar un runnable sin perder el prompt original
    # En elsiguiente caso podemos seguir manteniendo la pregunta
    # para luego poderla entregar con el contexto, ejemplo para un rag.
    """
    Asegurar que las variables clave de entrada (como la pregunta que hizo originalmente
    el usuario) puedan cruzar intactas ciertas etapas de la cadena sin ser sobrescritas 
    por otras operaciones.
    """
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke("¿Qué es TupiTupi?")
    print("PASSTHROUGH DEMO: ")
    print(response)
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain LCEL - Fundamentos")
    print("=" * 60)
    # demo_simple_chain()
    # demo_steps_inspection()
    # demo_batch()
    # demo_stream()
    demo_passthrough()
