from __future__ import annotations

from typing import List

import streamlit as st

from agent import PersonalKnowledgeAgent, AgentResponse
from embeddings import EmbeddingModel
from ingestion import ingest_streamlit_files
from llm import OllamaClient, LLMMessage
from vector_store import VectorStore, DocumentChunk


def init_state() -> None:
    """
    Initialize long-lived objects in Streamlit session state.
    """
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = EmbeddingModel()

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(
            embedding_model=st.session_state.embedding_model
        )

    if "ollama_client" not in st.session_state:
        st.session_state.ollama_client = OllamaClient(model="llama3:8b")

    if "agent" not in st.session_state:
        st.session_state.agent = PersonalKnowledgeAgent(
            llm=st.session_state.ollama_client,
            vector_store=st.session_state.vector_store,
            max_iterations=4,
        )

    if "chat_history" not in st.session_state:
        # Stored as list[LLMMessage]
        st.session_state.chat_history: List[LLMMessage] = []


def render_sidebar() -> None:
    st.sidebar.header("Knowledge Base")
    st.sidebar.markdown(
        "Upload your personal documents (PDF, TXT, Markdown). "
        "They will be embedded with `sentence-transformers` and indexed in FAISS."
    )

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Ingest documents") and uploaded_files:
        with st.sidebar.spinner("Ingesting documents..."):
            results = ingest_streamlit_files(
                files=uploaded_files, vector_store=st.session_state.vector_store
            )
        st.sidebar.success("Ingestion complete.")
        for filename, status in results:
            st.sidebar.write(f"- **{filename}**: {status}")


def render_chat_history() -> None:
    if not st.session_state.chat_history:
        return

    st.subheader("Conversation")
    for msg in st.session_state.chat_history:
        role_label = "User" if msg.role == "user" else "Assistant"
        st.markdown(f"**{role_label}:** {msg.content}")


def render_debug_section(response: AgentResponse) -> None:
    with st.expander("🔍 Debug: Tools & Retrieved Chunks"):
        st.markdown("**Agent Tool Steps**")
        for i, step in enumerate(response.steps, start=1):
            st.write(f"Step {i}: tool = `{step.tool_used}`")
            if step.tool_input:
                st.json(step.tool_input)

            if step.retrieved_chunks:
                st.markdown("Retrieved chunks:")
                for c in step.retrieved_chunks:
                    _render_chunk(c)
                    st.markdown("---")

        st.markdown("**Guardrails**")
        st.write(f"Reached max iterations: `{response.reached_max_iterations}`")


def _render_chunk(chunk: DocumentChunk) -> None:
    st.markdown(f"- **Source**: `{chunk.source}`  \n")
    st.markdown(chunk.text.strip() or "_(empty chunk)_")


def main() -> None:
    st.set_page_config(page_title="AI Personal Knowledge Assistant", layout="wide")
    init_state()

    st.title("🧠 AI Personal Knowledge Assistant")
    st.markdown(
        "Ask questions over your personal documents using a local LLM "
        "(`llama3:8b` via Ollama), FAISS, and sentence-transformers."
    )

    render_sidebar()
    render_chat_history()

    st.markdown("---")
    st.subheader("Ask a question")

    with st.form(key="question_form", clear_on_submit=True):
        user_question = st.text_input("Your question", key="user_question_input")
        submitted = st.form_submit_button("Send")

    if submitted and user_question.strip():
        user_msg = LLMMessage(role="user", content=user_question.strip())
        st.session_state.chat_history.append(user_msg)

        try:
            agent: PersonalKnowledgeAgent = st.session_state.agent
            response: AgentResponse = agent.run(
                question=user_question.strip(),
                chat_history=st.session_state.chat_history,
            )
            assistant_msg = LLMMessage(role="assistant", content=response.answer)
            st.session_state.chat_history.append(assistant_msg)

            st.subheader("Answer")
            st.write(response.answer)

            render_debug_section(response)

        except Exception as e:
            st.error(
                "There was an error while generating the answer. "
                "Ensure Ollama is running (`ollama serve`) and the `llama3:8b` model is installed."
            )
            st.exception(e)


if __name__ == "__main__":
    main()

