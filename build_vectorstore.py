import os
import shutil
from dotenv import load_dotenv
import re

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


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

    print("\nGrouping documents and splitting into chunks...")

    # Group documents by source file to process each PDF independently
    docs_by_source = {}
    for doc in all_docs:
        source = doc.metadata.get('source')
        if not source:
            continue
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)

    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200
    )

    for source, doc_list in docs_by_source.items():
        # Sort pages to ensure correct order before merging
        doc_list.sort(key=lambda d: d.metadata.get('page', 0))

        # Inject page markers into the text. This allows us to split the entire
        # document's text at once, overcoming the issue of splitting across page
        # boundaries, while still being able to trace a chunk back to its source page.
        text_with_page_markers = []
        for doc in doc_list:
            page_num = doc.metadata.get('page', 0) + 1 # PyMuPDF is 0-indexed
            text_with_page_markers.append(f"[PAGE_MARKER:{page_num}]\n{doc.page_content}")
        
        full_text = "\n\n".join(text_with_page_markers)
        
        # Split the full text
        chunks = text_splitter.split_text(full_text)
        
        for chunk in chunks:
            # For each chunk, find the last page marker to assign a page number.
            # This is a reasonable approximation for where the chunk is located.
            matches = re.findall(r'\[PAGE_MARKER:(\d+)\]', chunk)
            page_number = int(matches[-1]) if matches else 1 # Fallback to page 1

            # Clean the markers from the text before creating the Document
            cleaned_chunk = re.sub(r'\[PAGE_MARKER:\d+\]\n?', '', chunk).strip()

            if cleaned_chunk: # Avoid creating empty documents
                all_splits.append(Document(
                    page_content=cleaned_chunk,
                    metadata={'source': source, 'page': page_number}
                ))

    print("Creating embeddings and building vector store... (This may take a while)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if os.path.exists(PERSIST_DIR):
        print(f"Removing existing vector store at '{PERSIST_DIR}'...")
        shutil.rmtree(PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=all_splits, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIR
    )
    print(f"\nVector store built successfully with {len(all_splits)} chunks.")
    print(f"Data persisted to '{PERSIST_DIR}'.")

if __name__ == "__main__":
    main()