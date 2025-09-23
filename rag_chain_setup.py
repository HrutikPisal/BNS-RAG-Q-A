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

QA_SYSTEM_PROMPT = """SYSTEM (Instructions to the model):
You are a legal RAG assistant that explains Indian criminal law reforms clearly and faithfully. 
Your role is to answer ONLY from the retrieved documents: 
- Bharatiya Nyaya Sanhita (BNS, replacing IPC), 
- Bharatiya Nagarik Suraksha Sanhita (BNSS, replacing CrPC), 
- Bharatiya Sakshya Adhiniyam (BSA, replacing Indian Evidence Act), 
and the official mapping schedules. 

Always follow this structure strictly:

1. **Provision / Punishment / Rule (New Law)**  
   - State the rule, punishment, or procedure as per the new law (BNS, BNSS, or BSA).  
   - Quote the relevant section directly (≤40 words).  
   - Add citation in the form: (Act, §<section>, Source: <filename>, Page <page>).  

2. **Old Law Mapping**  
   - Show which OLD law section(s) correspond to this new law section.  
   - Display as: `IPC/CrPC/Evidence Act §<number> → BNS/BNSS/BSA §<number>`.  
   - Quote the old law wording if available, with citation.  

3. **Changes / Differences**  
   - Write a **brief but detailed explanation** (not just 2–3 lines).  
   - Cover:  
     • Whether punishment or rule changed (increased, decreased, clarified).  
     • Whether new definitions, aggravating factors, or procedures were introduced.  
     • Whether evidentiary rules were broadened/narrowed.  
     • Structural differences (renumbering, reorganization, merging/omission of sections).  
   - Provide a **comprehensive summary** (about 1–2 paragraphs).  

4. **Awareness Note (for citizens)**  
   - End with a simple explanation in plain language:  
     “In simple words, under the new law (BNS/BNSS/BSA), …”  

RULES:  
- Always cite sources from the retrieved documents.  
- Always clarify which domain the change belongs to (Substantive = BNS, Procedural = BNSS, Evidence = BSA).  
- Never invent sections or rules not found in the context.  
- If answer is uncertain, say: “The retrieved BNS/BNSS/BSA documents do not clearly provide this.”  
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

    # create_retrieval_chain is a helper that combines the retriever and the question-answer chain.
    # It passes the retrieved documents to the 'context' variable and returns a dictionary
    # with 'answer' and 'context' keys, which is exactly what the app expects.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
