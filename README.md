## AI Personal Knowledge Assistant

A production-ready, local-first AI personal knowledge assistant built with:

- **Ollama + `llama3:8b`** as the LLM (HTTP API, no LangChain)
- **FAISS** as the vector database
- **`sentence-transformers`** for text embeddings
- **Streamlit** for the chat UI and document upload

The system implements a manual RAG pipeline and an LLM-based agent that decides when to retrieve from your knowledge base versus answering directly.

---

## Features

- **Document ingestion (RAG)**:
  - PDF (`.pdf`) via `pypdf`
  - Plain text (`.txt`)
  - Markdown (`.md`, `.markdown`)
- **Embeddings & vector store**:
  - `sentence-transformers` (`all-MiniLM-L6-v2` by default)
  - FAISS index with on-disk persistence
  - On-disk **embedding cache** to avoid recomputation
  - **Document-level deduplication** so the same file is not re-indexed
- **Agent capabilities**:
  - LLM-based router decides to:
    - answer directly from chat history, or
    - call tools for semantic search and summarization
  - Tools:
    - `semantic_search` (FAISS-backed semantic document search)
    - `summarize_context` (LLM-powered summarization of retrieved chunks)
  - **Max-iteration guardrail** on the agent loop
  - Per-step trace of:
    - which tool was used
    - inputs
    - retrieved chunks
- **Conversation memory**:
  - Chat history kept in Streamlit session state and fed into the agent.
- **Frontend (Streamlit)**:
  - Chat-style Q&A interface
  - Document upload and ingestion panel
  - Debug panel showing tools used, retrieved chunks, and guardrail info.
- **Clean error handling**:
  - Graceful handling when Ollama is not running
  - Robustness against corrupted caches/indexes

---

## Project Structure

- `ingestion.py` – Document ingestion and chunking
- `embeddings.py` – SentenceTransformer wrapper with disk cache
- `vector_store.py` – FAISS-based vector store and metadata
- `tools.py` – Semantic search and context summarization tools
- `agent.py` – LLM-based agent and agent loop
- `llm.py` – Minimal Ollama HTTP client
- `streamlit_app.py` – Streamlit UI and wiring
- `app.py` – Streamlit entrypoint (`streamlit run app.py`)
- `requirements.txt` – Python dependencies

---

## Prerequisites

- **Python**: 3.10+ recommended
- **Ollama** installed locally
  - See the Ollama installation docs for your OS (`https://ollama.com/download`).

---

## 1. Set up Ollama with `llama3:8b`

1. Install Ollama (once), following the official instructions.
2. Pull the `llama3:8b` model:

```bash
ollama pull llama3:8b
```

3. Ensure the Ollama server is running. On most systems, starting Ollama once will keep it running in the background. If needed, you can explicitly start it:

```bash
ollama serve
```

The app expects Ollama to be reachable at `http://localhost:11434` (default).

---

## 2. Install Python dependencies

From the project root (this folder):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
# source .venv/bin/activate  # macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- `streamlit`
- `faiss-cpu`
- `pypdf`
- `sentence-transformers`
- `numpy`
- `requests`

No LangChain is used in this project; the RAG pipeline and agent are implemented manually.

---

## 3. Run the app

From the project root, with your virtual environment activated and Ollama running:

```bash
streamlit run app.py
```

This will open the Streamlit UI in your browser (typically at `http://localhost:8501`).

---

## 4. Using the Assistant

1. **Upload documents**:
   - In the left **Knowledge Base** sidebar, upload any combination of:
     - `.pdf`
     - `.txt`
     - `.md` / `.markdown`
   - Click **Ingest documents**.
   - The app:
     - Stores files under `data/uploads/`
     - Extracts text
     - Splits into overlapping chunks
     - Embeds with `sentence-transformers`
     - Indexes vectors in FAISS
     - Caches embeddings on disk
     - Skips documents that were already indexed before

2. **Ask questions**:
   - Use the **Ask a question** section in the main panel.
   - The agent:
     - Maintains **chat history** as conversational memory.
     - Calls an internal LLM-based router to decide whether to:
       - Answer directly from history, or
       - Call tools to search your knowledge base and summarize context.

3. **Inspect tools and retrieved chunks**:
   - Expand the **“Debug: Tools & Retrieved Chunks”** section under each answer.
   - You will see:
     - Which tool(s) were used (`semantic_search`, `summarize_context`, or `none`)
     - Tool input parameters
     - Retrieved chunks (source file and text content)
     - Whether the max-iteration guardrail was reached.

---

## 5. Implementation Notes

- **Agent loop (`agent.py`)**:
  - Uses an internal helper to ask the LLM whether to:
    - `"retrieve"` – run semantic search over the vector store, optionally summarize, then answer using the retrieved context.
    - `"answer_direct"` – answer immediately using the chat history (no retrieval).
  - Imposes a configurable `max_iterations` guardrail to avoid unbounded loops.
  - Returns an `AgentResponse` that includes:
    - Final answer text
    - A list of `AgentStepTrace` objects (tools used, tool inputs, retrieved chunks)
    - Whether the max-iteration limit was reached.

- **LLM client (`llm.py`)**:
  - Talks directly to the Ollama HTTP API (`/api/chat`).
  - Provides:
    - `chat(messages)` for multi-turn chat
    - `complete(prompt, system_prompt)` convenience helper for instruction-style prompts.

- **Ingestion and caching**:
  - `ingestion.py` computes a stable `doc_id` per uploaded file using a SHA-256 hash of the file contents + filename.
  - The vector store tracks which `doc_id`s have already been indexed and skips them on re-ingestion.
  - `embeddings.py` keeps an on-disk cache (`data/embeddings_cache.pkl`) keyed by SHA-256 of each text chunk.

---

## 6. Troubleshooting

- **Ollama not running**:
  - If you see an error like “Ensure Ollama is running (`ollama serve`) and the `llama3:8b` model is installed”, double-check:
    - `ollama serve` is running
    - `ollama pull llama3:8b` has completed successfully.

- **Corrupted index or cache**:
  - If FAISS or cache files are corrupted, you can delete the `data/` directory and re-ingest documents:
    - `data/vector_store.faiss`
    - `data/vector_store_metadata.pkl`
    - `data/embeddings_cache.pkl`
    - `data/uploads/` (optional)

---

## 7. Extending the Assistant

Ideas:

- Add more tools (e.g., web search, calendar, code search).
- Add document management (list/delete indexed documents).
- Support additional file types (Word, HTML, etc.).
- Persist full chat histories to disk for later retrieval.

The codebase is modular and type-hinted, so you can evolve the agent and tools without touching the core ingestion and vector store logic.

