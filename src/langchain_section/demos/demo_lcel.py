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


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain LCEL - Fundamentos")
    print("=" * 60)
    demo_simple_chain()
