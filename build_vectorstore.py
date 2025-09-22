import os
import shutil
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# --- Constants ---
DATA_DIR = "./data/"
#PERSIST_DIR = "./chroma_db"
PERSIST_DIR = "./bge_db"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

def main():
    """
    Builds and persists a Chroma vector store from PDF documents in the data directory.
    """
    load_dotenv()
    os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

    print("Loading documents...")
    pdf_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{DATA_DIR}'. Aborting.")
        return

    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_path)
            all_docs.extend(loader.load())
            print(f"Loaded {len(loader.load())} pages from {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"Failed to load {pdf_path}: {e}")
    
    if not all_docs:
        print("No documents were loaded successfully. Aborting.")
        return

    print("\nSplitting documents into chunks...")
    # For legal documents, a generic chunk size can be suboptimal. A larger chunk size
    # helps keep entire sections (e.g., a crime's definition and its punishment) together
    # in a single chunk, providing more complete context to the LLM.
    # A larger overlap ensures better context continuity if a section must be split.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_docs)

    print("Creating embeddings and building vector store... (This may take a while)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIR
    )

    print(f"\nVector store built successfully with {len(splits)} chunks.")
    print(f"Data persisted to '{PERSIST_DIR}'.")

if __name__ == "__main__":
    main()
