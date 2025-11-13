🧑‍⚖️ BNS Law RAG Q&A
Conversational Indian Law Assistant with IPC → BNS Comparison (2025 Reforms)

This project is a Retrieval-Augmented Generation (RAG) application designed to help citizens, students, and legal professionals understand India’s new criminal law reforms and compare them with the old laws.

It provides accurate, citation-based answers to legal queries using official PDFs of:

Bharatiya Nyaya Sanhita (BNS), 2023

Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023

Bharatiya Sakshya Adhiniyam (BSA), 2023

IPC / CrPC / IEA (old laws for comparison)

✨ Features
🔍 1. Interactive Legal Q&A

Ask natural language questions like:

“What is the punishment for theft under BNS?”

⚖️ 2. IPC → BNS Comparison

Automatically highlights changes:

Sections

Terminology

Punishment duration

Compoundable / cognizable changes

📚 3. Retrieval with Citations

Every answer references the exact section from the connected PDFs.

💾 4. Persistent Chroma Vector DB

Embeddings are stored in a local Chroma database, so PDFs don’t need re-processing.

🤖 5. Hugging Face LLM + Embeddings

Embeddings: BAAI/bge-large-en-v1.5 (via Hugging Face API or local fallback)

LLM: Any HF inference model (Mistral, Llama 3, Gemma, Mixtral)

🧪 6. Built-in RAG Evaluation

Run automated scoring (via RAGAS) using test questions in test_plan.csv.

🏗️ Tech Stack
Component	Library / Service
Framework	Streamlit
Vector DB	ChromaDB
Embedding Model	BAAI/bge-large-en-v1.5
LLM	HuggingFace Inference API
Document Parsing	PyMuPDF
RAG Engine	LangChain
Deployment	Hugging Face Spaces

The project includes a custom HF embedding wrapper, updated to use:

InferenceClient.feature_extraction()

Correct router URLs

Robust fallback logic

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/HrutikPisal/BNS-RAG-Q-A.git
cd BNS-RAG-Q-A

2️⃣ Create a Virtual Environment & Install Dependencies
python -m venv venv


Activate it:

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate


Install packages:

pip install -r requirements.txt

3️⃣ Environment Variables

Create a .env file:

HF_TOKEN="your-huggingface-api-token"

Token Requirements:

read → download embedding model

write → use Inference API
Accept the license of the LLM you want to use (Mistral/Llama/etc.).

4️⃣ Add Legal PDFs

Place your official PDFs inside:

./data/
    ├── BNS.pdf
    ├── BNSS.pdf
    ├── BSA.pdf
    ├── IPC.pdf
    ├── CrPC.pdf
    ├── IEA.pdf
    └── mappings.pdf   (optional)

5️⃣ Build the Vector Store

Run once to generate embeddings and store them persistently in ./bge_db/:

python build_vectorstore.py


Re-run only when you add/update PDFs.

6️⃣ Run the Streamlit App
streamlit run app.py


This opens a full interactive legal chatbot interface in your browser.

7️⃣ (Optional) Run Automated Evaluation
python evaluation.py


Outputs:
evaluation_results.csv

Uses RAGAS to measure:

Faithfulness

Context precision

Context recall

Ground-truth similarity

🔧 Project Structure
.
├── app.py                   # Streamlit UI for the chatbot
├── rag_chain_setup.py       # RAG pipeline + HF embedding wrapper (updated)
├── build_vectorstore.py     # Creates Chroma DB from PDFs
├── evaluation.py            # RAG evaluation via RAGAS
├── data/                    # Source legal PDFs
├── bge_db/                  # Persistent Chroma vector DB
├── test_plan.csv            # Benchmark Q&A dataset
├── requirements.txt
├── .env
└── README.md

🧩 Updated HuggingFace Embedding System

The project includes a patched HF embedding client to fix common problems like:

InferenceClient object has no attribute 'embeddings'

Router 404 / 410 errors

Legacy API shutdowns

The updated wrapper uses:

client.feature_extraction(...)


and correct router URL:

https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction

📌 Recommended Models
Embeddings

BAAI/bge-large-en-v1.5

BAAI/bge-base-en-v1.5

LLMs for RAG

mistralai/Mistral-7B-Instruct-v0.2

meta-llama/Meta-Llama-3.1-8B-Instruct

google/gemma-2-9b-it

📤 Deploy on Hugging Face Spaces

Compatible with:

CPU (slow but works)

T4 GPU

A10G (recommended)

Add these variables to HF Space settings:

HF_TOKEN

❤️ Contribution

Pull requests & improvements are welcome!
Feel free to open issues for:

Indian law mistakes

Citation errors

Performance bugs

Better mapping datasets

🛡️ Disclaimer

This tool is for public awareness and education only.
It is not a substitute for legal advice.
Always consult a qualified legal professional for real cases.

