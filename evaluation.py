import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_ollama import ChatOllama

from rag_chain_setup import get_retriever, create_rag_chain
 
# Note: You may need to install pandas and tqdm:
# pip install pandas tqdm

# --- Constants ---
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_rag_chain():
    """
    Initializes components and returns the full RAG chain for evaluation.
    """
    load_dotenv()
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

    # --- Initialize LLM and Retriever ---
    llm = ChatOllama(model="gemma:2b", temperature=0.1)
    retriever = get_retriever(embedding_model=EMBEDDING_MODEL, persist_directory=PERSIST_DIR)

    # --- Create the RAG chain ---
    rag_chain = create_rag_chain(llm, retriever)
    return rag_chain

def main():
    """Main function to run the evaluation."""
    try:
        test_plan_df = pd.read_csv("test_plan.csv")
    except FileNotFoundError:
        print("Error: test_plan.csv not found. Please create it in the root directory.")
        return

    print("Setting up RAG chain... (This may take a moment)")
    try:
        rag_chain = get_rag_chain()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("RAG chain ready.")

    results = []
    for index, row in tqdm(test_plan_df.iterrows(), total=test_plan_df.shape[0], desc="Evaluating Questions"):
        question = row['question']
        
        # For evaluation, we pass an empty chat history.
        response = rag_chain.invoke({"input": question, "chat_history": []})
        answer = response['answer']

        print("\n" + "="*80)
        print(f"QUESTION {index+1}: {question}")
        print("-" * 80)
        print(f"EXPECTED BNS: {row['expected_bns_section']} | EXPECTED IPC: {row['expected_ipc_section']}")
        print(f"EXPECTED SUMMARY: {row['expected_change_summary']}")
        print("-" * 80)
        print(f"GENERATED ANSWER:\n{answer}")
        print("="*80)

        correctness = input("Correctness (cited right sections)? (p/f): ").lower().strip()
        completeness = input("Completeness (covered punishment + changes)? (p/f): ").lower().strip()
        faithfulness = input("Faithfulness (quoted real text, no hallucination)? (p/f): ").lower().strip()

        results.append({
            "question": question, "generated_answer": answer,
            "correctness": 1 if correctness == 'p' else 0,
            "completeness": 1 if completeness == 'p' else 0,
            "faithfulness": 1 if faithfulness == 'p' else 0,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)

    total = len(results_df)
    print("\n\n===== EVALUATION SUMMARY =====")
    print(f"Correctness Score: {results_df['correctness'].sum() / total:.2%}")
    print(f"Completeness Score: {results_df['completeness'].sum() / total:.2%}")
    print(f"Faithfulness Score: {results_df['faithfulness'].sum() / total:.2%}")
    print("\nResults saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
