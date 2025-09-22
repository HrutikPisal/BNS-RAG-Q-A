# 🧑‍⚖️ RAG Q&A for Indian Law (BNS ↔ IPC Awareness)

This project is a Retrieval-Augmented Generation (RAG) application that helps users ask legal questions about the new Bharatiya Nyaya Sanhita (BNS) and compare its provisions with the old Indian Penal Code (IPC).

## Overview

The application provides a conversational interface to a knowledge base built from the official PDF documents of the BNS, IPC, and their official mapping tables. It is designed for legal professionals, students, and citizens to quickly understand the changes introduced in India's new criminal laws.

### Features

-   **Interactive Q&A:** Ask questions in natural language (e.g., "What is the punishment for murder?").
-   **IPC vs. BNS Comparison:** Get answers that highlight the changes from the IPC to the BNS.
-   **Sourced Answers:** Responses include direct quotes from the legal texts and cite the relevant sections.
-   **Persistent Knowledge Base:** Uses a Chroma vector database to store document embeddings, avoiding the need to re-process files on every run.
-   **Local LLM Support:** Powered by Ollama, allowing you to run models like Mistral or Llama 3 locally.
-   **Built-in Evaluation:** Includes a script to evaluate the RAG pipeline's performance on a set of benchmark questions.

## Tech Stack

-   **Frameworks:** LangChain, Streamlit
-   **Vector DB:** ChromaDB
-   **Embedding Model:** HuggingFace (`BAAI/bge-large-en-v1.5`)
-   **LLM Backend:** Ollama
-   **Document Loading:** PyMuPDF

## Setup and Usage

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/HrutikPisal/BNS-RAG-Q-A.git
cd BNS-RAG-Q-A

```

### 2. Create a Virtual Environment and Install Dependencies

It's recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory and add your Hugging Face token. This token requires 'read' permissions for downloading the embedding model and 'write' permissions to use the Inference API for the LLM.

**Note:** You must also visit the Hugging Face repository for your chosen model (e.g., `mistralai/Mistral-7B-Instruct-v0.2` or `meta-llama/Meta-Llama-3.1-8B-Instruct`) and accept its license agreement to use it via the API.

```
HF_TOKEN="your-hugging-face-token"
```

### 4. Add Legal Documents

Place the PDF files for the Bharatiya Nyaya Sanhita (BNS), Indian Penal Code (IPC), and any official mapping documents into the `./data/` directory.

### 5. Build the Vector Store

Run the following script once to process the PDFs and create the persistent vector database in the `./bge_db/` directory.

```bash
python build_vectorstore.py
```
This script only needs to be run again if you add or change the PDF files in the `data` directory.

### 6. Run the Streamlit Application

Start the interactive web application with the following command:

```bash
streamlit run app.py
```
This will open the application in your web browser.

### 7. (Optional) Run the Evaluation

To test the performance of the RAG pipeline, you can run the evaluation script. This uses the RAGAs framework to automatically score the answers for a predefined set of questions in `test_plan.csv`.

```bash
python evaluation.py
```
The results will be saved to `evaluation_results.csv`.

## Project Structure

```
.
├── data/                  # Folder to store source PDF documents.
├── chroma_db/             # Persisted Chroma vector database (ignored by git).
├── app.py                 # Main Streamlit application file.
├── rag_chain_setup.py     # Utility for creating the RAG chain.
├── build_vectorstore.py   # Script to create and persist the vector store.
├── evaluation.py          # Script to run the evaluation pipeline.
├── test_plan.csv          # CSV with benchmark questions for evaluation.
├── requirements.txt       # Python dependencies.
├── .env                   # Environment variables (e.g., HF_TOKEN).
└── README.md              # This file.
```
