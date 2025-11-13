🧑‍⚖️ BNS RAG Q&A — Indian Law Awareness Assistant
Retrieval-Augmented Generation (RAG) System for BNS ↔ IPC Legal Understanding
<p align="center"> <img src="A_combination_logo_and_flowchart_digital_illustrat.png" width="400"/> </p>
📘 Overview

BNS RAG Q&A is an AI-powered legal assistant built using RAG (Retrieval-Augmented Generation) to explain India’s new criminal law reforms (BNS/BNSS/BSA 2023) and compare them with the older IPC/CrPC/IEA laws.

Just ask questions like:

“What is the punishment for theft under BNS?”

“How does BNS Section 302 differ from IPC 302?”

“Explain the changes in grievous hurt definition.”

The system retrieves the exact legal text from your uploaded PDFs and uses an LLM to generate clear, cited answers.

✨ Features
🔍 Natural Language Q&A

Ask questions conversationally; the system retrieves relevant sections and explains them.

⚖️ IPC → BNS Comparison

Automatically highlights:

Section number changes

New wording

Punishment differences

Cognizable / compoundable changes

📚 True RAG with Citations

All answers include source references from the law PDFs.

📦 Persistent Vector Store

Indexes your PDFs into a local Chroma database so embeddings don’t recreate on every run.

🤖 HuggingFace LLM Support

Works with:

Mistral

LLaMA 3 / 3.1

Gemma

Mixtral

Other HF Inference API models

🧪 RAG Evaluation Included

Evaluate your system using RAGAS with test_plan.csv.

🏗️ Architecture Overview
<p align="center"> <img src="A_combination_logo_and_flowchart_digital_illustrat.png" width="600"/> </p>
💻 Tech Stack
Component	Technology
UI	Streamlit
Embeddings	BAAI/bge-large-en-v1.5
LLM	HuggingFace Inference API
Vector Store	ChromaDB
RAG Framework	LangChain
PDF Loader	PyMuPDF
🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/HrutikPisal/BNS-RAG-Q-A.git
cd BNS-RAG-Q-A

2️⃣ Create a Virtual Environment & Install Requirements
python -m venv venv


Activate:

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate


Install deps:

pip install -r requirements.txt

3️⃣ Set Up Environment Variables

Create a .env file:

HF_TOKEN="your_huggingface_api_token"

Token Requirements:

read → for model access

write → to call Inference API

Accept license for your chosen LLM (Mistral/Llama/Gemma)

4️⃣ Add Legal Documents

Place your official PDFs inside /data:

data/
│── BNS.pdf
│── BNSS.pdf
│── BSA.pdf
│── IPC.pdf
│── CrPC.pdf
│── IEA.pdf
└── mapping_docs.pdf   (optional)

5️⃣ Build the Vector Store

Run once to generate embeddings:

python build_vectorstore.py


Re-run only when you add/change PDFs.

6️⃣ Run the App
streamlit run app.py


The chatbot will open in your browser.

7️⃣ (Optional) Run RAG Evaluation
python evaluation.py


Outputs:

evaluation_results.csv

📂 Project Structure
.
├── app.py                   # Streamlit app
├── rag_chain_setup.py       # RAG chain + HF embedding wrapper (updated)
├── build_vectorstore.py     # PDF → Chroma DB pipeline
├── evaluation.py            # RAG evaluation using RAGAS
├── data/                    # Legal PDFs
├── bge_db/                  # Persistent Chroma DB
├── test_plan.csv            # Benchmark questions
├── requirements.txt
├── .env
└── README.md

🔧 HuggingFace Embedding Fix Included

The updated version of rag_chain_setup.py uses:

✔ InferenceClient.feature_extraction()
✔ Correct router URL:

https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction


✔ Fallback to:

Router API

Legacy API

Local BGE embeddings

✔ Robust error handling for 404/410/timeout

📜 Disclaimer

This tool is for legal awareness and education only.
It is NOT a substitute for professional legal advice.
Consult a qualified lawyer for real-world cases.

❤️ Contributions Welcome

Improve mapping datasets

Correct legal interpretations

Improve RAG performance

Add more acts (POSH, IT Act, Motor Vehicles Act, etc.)
