import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace

from rag_chain_setup import get_retriever, create_rag_chain

# --- Constants ---
PERSIST_DIR = "./bge_db"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Example model
HF_TOKEN = os.getenv("HF_TOKEN")


# --- Load Environment Variables ---
load_dotenv()
if not os.getenv("HF_TOKEN"):
    st.error("HF_TOKEN not found. Please set it in your .env file.")
    st.info("You can get a token from https://huggingface.co/settings/tokens")
    st.stop()

st.title("⚖️ Indian Law Chatbot")

# --- LLM Configuration ---
# Using a cloud model via Hugging Face Inference API
try:
    # Step 1: Create the HuggingFace endpoint LLM
    endpoint_llm = HuggingFaceEndpoint(
        repo_id=HF_MODEL,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=1024,
    )
    # Step 2: Wrap it as a Chat model
    llm = ChatHuggingFace(llm=endpoint_llm)
   
except Exception as e:
    st.error(f"Failed to initialize Hugging Face model: {e}")
    st.info(f"Please ensure you have accepted the license for {HF_MODEL} on the Hugging Face Hub if required.")
    st.stop()


# We will use a single, fixed session ID for each user's browser session.
# This removes the need for the user to manually enter a session ID.
session_id = "user_session"

if 'store' not in st.session_state:
    st.session_state.store={}

# --- Load retriever and create RAG chain ---
try:
    retriever = get_retriever(embedding_model=EMBEDDING_MODEL, persist_directory=PERSIST_DIR)
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

if prompt := st.chat_input("Ask a question about the Indian Law Reforms"):
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
