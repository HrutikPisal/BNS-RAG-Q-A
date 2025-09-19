import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Prompts ---
CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

QA_SYSTEM_PROMPT = """You are a legal-assistant RAG agent whose job is to answer statutory/legal questions by using ONLY the retrieved legal sources provided. Always:
1) Give a concise direct answer (1–3 sentences) first.
2) Then provide a short "Change summary (IPC → BNS)" if the question relates to a change in law.
3) For each statutory claim, **quote the exact section text (max 40 words)** and label which Act and section the quote is from (e.g., "BNS §103" or "IPC §302").
4) Where a mapping exists, show the mapping explicitly: "IPC §302 → BNS §103" (use the authoritative correspondence table if present).
5) Always include which source(s) support each statement — display source titles and page/section metadata (from the retrieved source objects).
6) If the retrieved sources do not clearly answer the request say: "I cannot find a direct answer in the provided documents" and list which docs were searched.
7) Do not hallucinate — if unsure, state uncertainty and point to the specific retrieved source text."""

def get_retriever(embedding_model: str, persist_directory: str):
    """Loads a retriever from a persisted Chroma vector store."""
    if not os.path.isdir(persist_directory):
        raise FileNotFoundError(f"Vector store not found at '{persist_directory}'. Please run `build_vectorstore.py` first.")
    
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore.as_retriever()

def create_rag_chain(llm, retriever):
    """Creates the full RAG chain."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{context}\n\nQUESTION: {input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain