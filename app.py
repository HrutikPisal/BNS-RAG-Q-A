import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
import os 
from dotenv import load_dotenv

from rag_chain_setup import get_retriever, create_rag_chain

load_dotenv()
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')

# --- Constants ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "./chroma_db"

st.title("🧑‍⚖️ RAG Q&A for Indian Law (BNS ↔ IPC)")

llm=ChatOllama(model="gemma:2b", temperature=0.1)

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

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Display chat messages from history on app rerun
if session_id in st.session_state.store:
    for message in st.session_state.store[session_id].messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

if prompt := st.chat_input("Ask a question about the PDF"):
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        response = conversational_rag_chain.invoke(
            {"input": prompt}, config={"configurable": {"session_id": session_id}}
        )
        st.markdown(response["answer"])
