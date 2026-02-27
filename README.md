#  🧠  AI Personal Knowledge Assistant (Local LLM Agent with RAG)

An AI-powered personal knowledge system that allows you to upload documents and chat with them using natural language — completely offline using a local LLM.

This project implements a custom Retrieval-Augmented Generation (RAG) pipeline + LLM agent with FAISS and Ollama to deliver accurate, context-aware answers from your own data.

🚀 Features

Upload and process personal documents
Semantic search using FAISS vector store
LLM agent that decides when to retrieve vs answer directly
Conversation memory (chat history aware)
Embedding cache to avoid recomputation
Document deduplication (no duplicate indexing)
Debug view for:

tools used

retrieved chunks

agent steps

Fully local execution (no API key required)

🧠 Tech Stack

Python
Streamlit
Ollama (Llama 3 / Mistral)
FAISS
Sentence Transformers
PyPDF
NumPy
Requests

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/ai-personal-knowledge-assistant.git
cd ai-personal-knowledge-assistant
2️⃣ Create virtual environment
python -m venv .venv
.venv\Scripts\activate
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
💡 How It Works

Documents are uploaded and split into chunks
Chunks are converted into embeddings
Stored in FAISS vector database

User query →

Agent decides:

Answer directly from conversation memory
or

Retrieve relevant context using semantic search

Retrieved context → summarized → grounded answer

📂 Project Structure
ingestion.py        → document ingestion & chunking
embeddings.py       → embedding model with disk cache
vector_store.py     → FAISS vector database
tools.py            → semantic search & summarization tools
agent.py            → LLM agent loop
llm.py              → Ollama HTTP client
streamlit_app.py    → Streamlit UI
app.py              → entry point
requirements.txt    → dependencies
data/               → vector store & cache
📌 Requirements

Python 3.10+
Ollama running locally

🏷️ Project Type

Local-First LLM Agent for Personal Knowledge Management (RAG + Tool Use)

👨‍💻 Author

Ranjith Kumar Mummadi
