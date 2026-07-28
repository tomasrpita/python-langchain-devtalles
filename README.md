# LangChain & LangGraph — Python IA Aplicada

Ejercicios y demos prácticas del curso **[Python IA Aplicada](https://cursos.devtalles.com/courses/python-ia-aplicada)** de DevTalles, correspondientes a las **secciones 7, 8, 9 y 10**: fundamentos de LCEL, memoria conversacional persistente, RAG (Retrieval-Augmented Generation) sobre documentos reales y agentes con LangGraph.

## 📚 Contenido cubierto

| Sección del curso | Tema oficial | En este repo | Demo / código |
|---|---|---|---|
| 7 | Introducción a LangChain | **LCEL** (LangChain Expression Language): cliente LLM centralizado, chains, `RunnableLambda`, `RunnablePassthrough`, streaming y batch | [`demo_lcel.py`](src/langchain_section/demos/demo_lcel.py) |
| 8 | LangChain memoria persistente | **Memoria conversacional persistente**: interfaz base de memoria, backends SQLite / PostgreSQL (con Docker) y `RunnableWithMessageHistory` | [`demo_memory.py`](src/langchain_section/demos/demo_memory.py) |
| 9 | LangChain + RAG | **RAG** sobre documentos propios (PDF/TXT): document loaders, split de documentos, embeddings, indexación en vector store y fuentes consultadas | [`demo_rag.py`](src/langchain_section/demos/demo_rag.py) |
| 10 | LangGraph | **Agentes con LangGraph**: estados, nodos (analyze / retrieve / generate), edges condicionales y gestión de sesiones para un RAG agéntico | [`demo_langgraph.py`](src/langchain_section/demos/demo_langgraph.py) |

## 🗂️ Estructura del proyecto

```
.
├── main.py                          # Entry point base del proyecto
├── data/                            # (no versionado) documentos, DB SQLite y vector stores
│   └── documents/                   # Coloca aquí tus .txt / .pdf para los demos de RAG
└── src/langchain_section/
    ├── config/
    │   └── settings.py              # Configuración centralizada (modelos, paths, chunking)
    ├── core/
    │   ├── llm.py                   # Clientes cacheados de ChatOpenAI / OpenAIEmbeddings
    │   ├── embeddings.py            # Creación del vector store (Chroma)
    │   └── document_loader.py       # Carga y split de documentos (PDF/TXT)
    ├── chains/
    │   ├── base.py                  # Chain base del asistente con historial (LCEL)
    │   └── rag.py                   # Chain RAG (retriever + prompt + LLM)
    ├── memory/
    │   ├── base.py                  # Contrato BaseMemoryBackend
    │   ├── sqlite_memory.py         # Backend de memoria en SQLite
    │   └── postgresql_memory.py     # Backend de memoria en PostgreSQL
    ├── graphs/
    │   ├── state.py                 # Estado tipado del grafo (RAGAgenticState)
    │   ├── nodes.py                 # Nodos: analyze / retrieve / generate
    │   └── rag_agent.py             # Construcción y compilación del grafo (LangGraph)
    └── demos/
        ├── demo_lcel.py             # Sección 7
        ├── demo_memory.py           # Sección 8
        ├── demo_rag.py              # Sección 9
        └── demo_langgraph.py        # Sección 10
```

## ⚙️ Requisitos

- Python **3.13**
- [uv](https://docs.astral.sh/uv/) como gestor de dependencias y entornos
- Una **API Key de OpenAI** (los demos usan `gpt-4o-mini` por defecto y `text-embedding-3-small` para embeddings)
- *(Opcional, solo para memoria en PostgreSQL)* un servidor PostgreSQL accesible

## 🚀 Instalación

```bash
git clone <url-de-tu-repo>
cd python-langchain

# Instala las dependencias (crea el entorno virtual automáticamente)
uv sync
```

## 🔑 Configuración

Copia el archivo de ejemplo y completa tus credenciales:

```bash
cp env.example .env
```

Variables de entorno (`.env`):

| Variable | Descripción | Obligatoria |
|---|---|---|
| `OPENAI_API_KEY` | API Key de OpenAI | Sí |
| `OPENAI_MODEL` | Modelo de chat a usar (por defecto `gpt-4o-mini` si se deja vacío) | No |
| `MAX_TOKENS` | Límite de tokens de salida (referencial) | No |
| `DATABASE_URL` | Cadena de conexión PostgreSQL, ej. `postgresql://user:password@localhost:5432/langchain_memory` — solo necesaria si usas el backend de memoria en PostgreSQL | No |

> `.env` y la carpeta `data/` están en `.gitignore`: tus claves y los documentos/DBs locales nunca se suben al repo.

## ▶️ Cómo ejecutar cada demo

Todos los scripts se ejecutan con `uv run` desde la raíz del proyecto (usan imports absolutos `src.langchain_section...`).

### 1. LCEL — Fundamentos (Sección 7: Introducción a LangChain)

```bash
uv run python -m src.langchain_section.demos.demo_lcel
```

Incluye 5 funciones de ejemplo (`demo_simple_chain`, `demo_steps_inspection`, `demo_batch`, `demo_stream`, `demo_passthrough`). Por defecto el bloque `if __name__ == "__main__":` solo ejecuta `demo_passthrough()`; para probar las demás, edita el final de [`demo_lcel.py`](src/langchain_section/demos/demo_lcel.py) y descomenta la que quieras correr:

- **`demo_simple_chain`** — cadena básica `prompt | llm | parser`.
- **`demo_steps_inspection`** — inspecciona el resultado intermedio de cada eslabón de la cadena (prompt → LLM → parser).
- **`demo_batch`** — procesa varios inputs en paralelo con `.batch()`.
- **`demo_stream`** — respuesta en streaming token a token con `.stream()`.
- **`demo_passthrough`** — simula un retriever con `RunnableLambda` + `RunnablePassthrough` para no perder la pregunta original (base conceptual de RAG).

### 2. Memoria conversacional persistente (Sección 8: LangChain memoria persistente)

```bash
uv run python -m src.langchain_section.demos.demo_memory
```

Chatbot de consola con historial persistente. Al iniciar, elige el backend:

1. **SQLite** (por defecto, archivo local en `data/chat_history.db`)
2. **PostgreSQL** (requiere `DATABASE_URL` configurada; si falla la conexión, hace fallback automático a SQLite)

Comandos disponibles durante el chat: `historial`, `limpiar`, `sesiones`, `salir`.

### 3. RAG con documentos reales (Sección 9: LangChain + RAG)

Agrega tus archivos `.txt` o `.pdf` en `data/documents/` (ya existen algunos de ejemplo) y ejecuta:

```bash
uv run python -m src.langchain_section.demos.demo_rag
```

El script carga los documentos, los divide en chunks (`RecursiveCharacterTextSplitter`), genera embeddings y los indexa en un vector store **Chroma** local (`data/chromadb_rag_demo`). Luego abre un chat que responde **solo con base en el contenido indexado** y muestra las fuentes/páginas consultadas.

Comandos disponibles: `archivos`, `reindexar`, `chunks`, `salir`.

### 4. Agente RAG con LangGraph (Sección 10: LangGraph)

```bash
uv run python -m src.langchain_section.demos.demo_langgraph
```

Versión agéntica del RAG anterior: en vez de buscar siempre en los documentos, un grafo de estados (LangGraph) decide en cada turno si la pregunta requiere recuperación de contexto o puede responderse directamente.

```mermaid
graph LR
    START((START)) --> analyze[analyze]
    analyze -- needs_retrieval=true --> retrieve[retrieve]
    analyze -- needs_retrieval=false --> generate[generate]
    retrieve --> generate
    generate --> END((END))
```

- **`analyze`** — el LLM decide (`needs_retrieval: true/false`) si hace falta buscar en la base de conocimiento.
- **`retrieve`** — busca los chunks más relevantes en Chroma (`data/chromadb_knwoledge`).
- **`generate`** — genera la respuesta usando el contexto recuperado (si lo hay) y el historial reciente.

Además gestiona **sesiones persistentes** (SQLite o PostgreSQL, con el mismo fallback automático) permitiendo retomar conversaciones anteriores. Comandos disponibles: `sesion`, `historial`, `limpiar`, `salir`.

## 🐘 PostgreSQL opcional (para memoria persistente)

Si quieres probar el backend de memoria en PostgreSQL en local con Docker:

```bash
docker run --name langchain-postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:16
```

Y en tu `.env`:

```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/langchain_memory
```

La base de datos indicada en la URL se crea automáticamente si no existe (ver `PostgreSQLMemoryBackend._ensure_database_exists`).

## 📝 Notas técnicas

- La configuración global (modelos, tamaños de chunk, rutas de persistencia) vive en [`settings.py`](src/langchain_section/config/settings.py).
- Cada demo persiste su vector store en una ruta distinta dentro de `data/` para no mezclar índices entre ejercicios.
- Todo lo generado en tiempo de ejecución (`data/`, `.env`, `__pycache__/`) está excluido del control de versiones vía `.gitignore`.

## 📄 Licencia

Proyecto de uso educativo, desarrollado como parte del curso [Python IA Aplicada](https://cursos.devtalles.com/courses/python-ia-aplicada) de DevTalles.
