import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_ollama import ChatOllama
import os 

from rag_chain_setup import get_retriever, create_rag_chain

# --- Constants ---
PERSIST_DIR = "./bge_db"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


st.title("⚖️ RAG Q&A for Indian Law (BNS ↔ IPC)")

# --- LLM Configuration ---
# Using a local model via Ollama
OLLAMA_MODEL = "mistral" # Make sure you have pulled this model with `ollama run mistral`
try:
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.1,
    )
   
except Exception as e:
    st.error(f"Failed to connect to Ollama: {e}")
    st.info(f"Please ensure Ollama is running and you have pulled the '{OLLAMA_MODEL}' model (e.g., `ollama run {OLLAMA_MODEL}`).")
    st.stop()


# We will use a single, fixed session ID for each user's browser session.
# This removes the need for the user to manually enter a session ID.
session_id = "user_session"

if 'store' not in st.session_state:
    st.session_state.store={}

# --- Load retriever and create RAG chain ---
try:
    retriever = get_retriever(embedding_model=EMBEDDING_MODEL, persist_directory=PERSIST_DIR)
    st.sidebar.success("Vector store loaded successfully.")
    rag_chain = create_rag_chain(llm, retriever)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Display chat messages from history on app rerun
history = get_session_history(session_id)
for message in history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("Ask a question about the PDF"):
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Invoke the chain directly, providing the current question and the chat history
            response = rag_chain.invoke(
                {"input": prompt, "chat_history": history.messages}
            )
            
            # Manually update the chat history with the new user query and AI response
            history.add_user_message(prompt)
            history.add_ai_message(response["answer"])
    
            st.markdown(response["answer"])  # Display the main answer
    
            # Display the source documents in an expander
            with st.expander("View Sources"):
                for doc in response["context"]:
                    # Safely access metadata
                    source = doc.metadata.get('source', 'Unknown source')
                    page = doc.metadata.get('page', 'N/A')
                    st.info(f"Source: {os.path.basename(source)} (Page: {page})")
                    st.markdown(f"> {doc.page_content}")
