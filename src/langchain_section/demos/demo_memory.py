from langchain_core.runnables import RunnableWithMessageHistory

from src.langchain_section.chains.base import build_assitant_chain
from src.langchain_section.memory.base import BaseMemoryBackend


def build_chatbot(backend: BaseMemoryBackend) -> RunnableWithMessageHistory:
    """Contruye ChatBot"""

    chain = build_assitant_chain()

    return RunnableWithMessageHistory(
        chain,
        backend.get_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def run_chat_session(
    chatbot: RunnableWithMessageHistory, backend: BaseMemoryBackend, session_id: str
) -> None:
    print(f"\nSesión activa: {session_id}")

    messages = backend.get_history(session_id).messages

    if messages:
        print(f"Retornando la conversación ({len(messages)}) mensajes previos")
    else:
        print("Nueva Conversación")

    print("Comandos: 'historial | 'limpiar' | 'sesiones' | 'salir' ")

    while True:
        try:
            user_input = input("Tú: ").strip().lower()

            if not user_input:
                continue

            if user_input == "salir":
                print("Hasta luego!")
                break

            if user_input == "historial":
                messages = backend.get_history(session_id).messages

                if not messages:
                    print("historial vacío")
                    continue
                print(f"Últimos mensajes de '{session_id}'")

                for message in messages[-6:]:
                    rol = "Tú" if message.type == "human" else "AI"
                    print(f" {rol}: {message.content[:70]}...")
                print()
                continue

            if user_input == "limpiar":
                backend.clean_history(session_id)
                print("HISTORIAL BORRADO.")
                continue

            if user_input.lower() == "sesiones":
                sessions = backend.list_sessions()
                print(f" Sesiones disponibles: {sessions}")
                continue

            response = chatbot.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            print(f"IA: {response}")

        except KeyboardInterrupt:
            print("Adios Lucas!")
            break

        except Exception as e:
            print(f"Error {e}")
