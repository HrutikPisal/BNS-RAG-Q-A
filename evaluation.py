import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from langchain_community.chat_models import ChatOllama

from rag_chain_setup import get_retriever, create_rag_chain
 
# Note: You will need to install RAGAs and its dependencies:
# pip install ragas datasets

# --- Constants ---
PERSIST_DIR = "./bge_db"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

def get_rag_chain_and_llm():
    """
    Initializes components and returns the full RAG chain and the LLM for evaluation.
    """
    # --- Initialize LLM and Retriever ---
    OLLAMA_MODEL = "mistral"  # or "llama3", etc.

    # LLM for the RAG chain
    rag_llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.1,
    )
    # LLM for the evaluation
    eval_llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    retriever = get_retriever(embedding_model=EMBEDDING_MODEL, persist_directory=PERSIST_DIR)

    # --- Create the RAG chain ---
    rag_chain = create_rag_chain(rag_llm, retriever)
    return rag_chain, eval_llm

def main():
    """Main function to run the automated evaluation using RAGAs."""
    try:
        # Your test_plan.csv must now include a 'ground_truth' column
        # with the ideal, comprehensive answer for each question.
        test_plan_df = pd.read_csv("test_plan.csv")
        if 'ground_truth' not in test_plan_df.columns:
            print("Error: 'test_plan.csv' must contain a 'ground_truth' column for automated evaluation.")
            return
    except FileNotFoundError:
        print("Error: test_plan.csv not found. Please create it with 'question' and 'ground_truth' columns.")
        return

    print("Setting up RAG chain and evaluation LLM... (This may take a moment)")
    # The embedding model might still need the HF_TOKEN.
    load_dotenv()
    if not os.getenv('HF_TOKEN'):
        print("Warning: HF_TOKEN not found in .env file. This may be required for the BAAI/bge-large-en-v1.5 embedding model.")

    try:
        rag_chain, eval_llm = get_rag_chain_and_llm()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("RAG chain ready. Generating answers for evaluation...")

    eval_data = []
    for index, row in tqdm(test_plan_df.iterrows(), total=test_plan_df.shape[0], desc="Generating Answers"):
        question = row['question']
        ground_truth = row['ground_truth']
        
        # Invoke the chain to get answer and retrieved contexts
        response = rag_chain.invoke({"input": question, "chat_history": []})
        answer = response['answer']
        contexts = [doc.page_content for doc in response['context']]

        eval_data.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    # Convert to a Hugging Face Dataset
    eval_dataset = Dataset.from_list(eval_data)

    print("\nRunning RAGAs evaluation... (This will take some time)")
    
    # Define the metrics for evaluation
    metrics = [
        faithfulness,       # How much the answer is grounded in the context
        answer_relevancy,   # How relevant the answer is to the question
        context_precision,  # How relevant the retrieved context is
        context_recall,     # How well the retriever fetches all necessary context
    ]

    # Run the evaluation
    # To handle potential connection errors with Ollama during long evaluations,
    # we can adjust the concurrency.
    run_config = RunConfig(thread_pool_size=4)
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        run_config=run_config,
    )

    results_df = result.to_pandas() # type: ignore
    results_df.to_csv("evaluation_results_ollama.csv", index=False)

    print("\n\n===== RAGAS EVALUATION SUMMARY =====")
    print(results_df.mean())
    print("\nFull results saved to evaluation_results_ollama.csv")

if __name__ == "__main__":
    main()
