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

from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace

from rag_chain_setup import get_retriever, create_rag_chain
 
# Note: You will need to install RAGAs and its dependencies:
# pip install ragas datasets langchain-huggingface

# --- Constants ---
PERSIST_DIR = "./bge_db"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Same model as in app.py

def get_rag_chain_and_llm():
    """
    Initializes components and returns the full RAG chain and the LLM for evaluation.
    """
    # --- Initialize LLM and Retriever ---
    # LLM for the RAG chain (matches app.py)
    rag_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
        repo_id=HF_MODEL,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=1024,
    ))

    # LLM for the evaluation (low temperature for consistency)
    eval_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
        repo_id=HF_MODEL,
        task="text-generation",
        temperature=0.01, # Using a very low temp for deterministic evaluation
        max_new_tokens=1024,
    ))

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
    if not os.getenv("HF_TOKEN"):
        print("Error: HF_TOKEN not found in .env file. This is required for the Hugging Face model.")
        print("You can get a token from https://huggingface.co/settings/tokens")
        return

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
    # or rate limiting with HF API, we can adjust the concurrency.
    run_config = RunConfig(thread_pool_size=2) # Lowered to 2 to avoid HF rate limits
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        run_config=run_config,
    )

    results_df = result.to_pandas() # type: ignore
    results_df.to_csv("evaluation_results_hf.csv", index=False)

    print("\n\n===== RAGAS EVALUATION SUMMARY =====")
    print(results_df.mean())
    print("\nFull results saved to evaluation_results_hf.csv")

if __name__ == "__main__":
    main()
