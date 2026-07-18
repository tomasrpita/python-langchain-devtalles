from langchain_core.runnables import RunnableWithMessageHistory

from src.langchain_section.chains.base import build_assitant_chain
from src.langchain_section.memory.base import BaseMemoryBackend
from src.langchain_section.memory.postgresql_memory import PostgreSQLMemoryBackend
from src.langchain_section.memory.sqlite_memory import SQLiteMemoryBackend


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
            print("👋🏻 Adios Luca!")
            break

        except Exception as e:
            print(f"Error {e}")


if __name__ == "__main__":
    print("\n¡Qué backend memory usar?")
    print("1.SQLite (archivo local y por defecto)")
    print("2.PostgresSQL (requiere docker ejecutandose)")

    choice = input("Elige(1/2): ").strip()

    if choice == "2":
        try:
            backend = PostgreSQLMemoryBackend()
            print("✅ Conectado a PostgreSQL")
        except Exception as e:
            print(f"❌ {e}")
            print("USando SQLite como fallback")
            backend = SQLiteMemoryBackend()
            print("✅ Conectando a SQLite")

    else:
        backend = SQLiteMemoryBackend()
        print("✅ Conectando a SQLite")

    chatbot = build_chatbot(backend=backend)
    run_chat_session(chatbot=chatbot, backend=backend, session_id="session_id_001")
