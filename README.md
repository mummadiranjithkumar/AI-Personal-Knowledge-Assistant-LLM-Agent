🧠 AI Personal Knowledge Assistant (Local LLM Agent with RAG)

A production-ready, local-first AI personal knowledge assistant that lets you upload documents and chat with them using a fully offline LLM.

Built with a custom RAG pipeline, FAISS vector search, and an LLM-driven tool-using agent — without LangChain.

🚀 Features

📄 Upload and chat with your personal documents
🔍 Semantic search over your knowledge base (FAISS)
🧠 LLM agent that decides when to retrieve vs answer directly
💬 Conversational memory (chat history aware)
⚡ Embedding cache to avoid recomputation
♻️ Document deduplication (no re-indexing same file)
🛠 Debug panel showing:

tools used

retrieved chunks

agent reasoning steps
💻 100% local execution (no paid APIs)

🧠 Tech Stack

Python

Streamlit – UI

Ollama (Llama 3 / Mistral) – Local LLM

FAISS – Vector database

sentence-transformers – Embeddings

PyPDF – PDF parsing

NumPy

🏗 Architecture

This project implements a manual RAG + LLM agent loop:

Documents are ingested and chunked

Chunks → embeddings

Stored in FAISS

User query →

Agent decides:

🔹 Answer directly

🔹 Use semantic search tool

Retrieved context → summarized → grounded response

📂 Project Structure
ingestion.py        → document loading & chunking
embeddings.py       → embedding model + disk cache
vector_store.py     → FAISS index & metadata
tools.py            → semantic search & summarization tools
agent.py            → LLM agent loop
llm.py              → Ollama HTTP client
streamlit_app.py    → UI logic
app.py              → entry point
requirements.txt    → dependencies
data/               → vector store & cache
⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/your-username/ai-personal-knowledge-assistant.git
cd ai-personal-knowledge-assistant
2️⃣ Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS / Linux
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Install & run Ollama

Download:

https://ollama.com

Pull the model:

ollama pull llama3:8b

Start Ollama:

ollama serve
5️⃣ Run the app
streamlit run app.py
💡 Usage
Upload Documents

Supported:

PDF

TXT

Markdown

Then click Ingest documents.

Ask Questions

The agent will:

use chat memory

decide whether retrieval is needed

return a grounded answer

Debug Mode

Expand:

🔎 Debug: Tools & Retrieved Chunks

to see:

tool calls

retrieved context

guardrail status

🧩 Key Highlights

✅ No LangChain — fully custom RAG pipeline
✅ Tool-using LLM agent
✅ Local-first & privacy-focused
✅ On-disk FAISS persistence
✅ Embedding cache for performance
✅ Modular & extensible codebase

🔮 Future Improvements

Web search tool

Document management UI

Additional file formats (DOCX, HTML)

Persistent chat history

Multi-user support

👨‍💻 Author

Ranjith Kumar Mummadi
AI / GenAI Engineer (Fresher – 2025)
