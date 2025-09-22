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
    "formulate a standalone question which can be understood without the chat history. "
    "Also, correct any spelling mistakes in the user's question. "
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

QA_SYSTEM_PROMPT = """You are a legal RAG assistant that explains Indian criminal law reforms clearly and faithfully. 
Always base your answer ONLY on the retrieved context (BNS, IPC, and mapping documents). 
Never hallucinate.

Your answer must strictly follow this structure:

1. **Punishment / Implication / Definition (BNS)**  
   - State the punishment or implication or definition according to the new BNS law.  
   - Quote the exact BNS section (≤40 words).  
   - Add citation in format: (BNS §<section>, Source: <filename>, Page <page>).  

2. **IPC Mapping**  
   - Show mapping: `IPC §<number> → BNS §<number>`.  
   - Quote the IPC section if available, with citation.  

3. **Changes / Differences**  
   - Write a brief but detailed explanation (not just 2–3 sentences).  
   - Cover: whether punishment changed, new aggravating factors, structural renumbering, or broadened/narrowed definitions.  
   - At least one short paragraph (4–6 sentences).  

4. **Awareness Note (for citizens)**  
   - End with a simplified explanation in plain language:  
     "In simple words: …"  

Rules:  
- Every claim must have a quote + citation from context.  
- If insufficient context, say: "The provided documents do not contain a direct answer."  
- Do NOT invent sections or punishments.
"""

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
            # MessagesPlaceholder("chat_history"), # History is already used to generate context-aware question
            ("human", "{context}\n\nQUESTION: {input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # This chain is designed to return a dictionary with 'answer' and 'context' keys.
    # 1. RunnablePassthrough.assign(context=...): Retrieves documents and adds them to the 'context' key.
    # 2. .assign(answer=...): Takes the result from the first step, runs the question_answer_chain,
    #    and adds the final string answer to the 'answer' key.
    from langchain_core.runnables import RunnablePassthrough
    rag_chain = RunnablePassthrough.assign(
        context=history_aware_retriever
    ).assign(answer=question_answer_chain)
    return rag_chain
