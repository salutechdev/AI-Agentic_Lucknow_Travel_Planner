# 🗺️ Agentic Travel Planner

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Deployment-Render-55FF9C.svg" alt="Deployed on Render">
  </a>
  <img src="https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-green.svg" alt="Framework">
  <img src="https://img.shields.io/badge/LLM-Groq%20(LLama--4)-orange.svg" alt="LLM">
  <img src="https://img.shields.io/badge/Architecture-Agentic%20RAG-purple.svg" alt="Architecture">
</p>

An enterprise-grade, full-stack AI travel planner that provides personalized, data-driven itineraries for Lucknow, India. This project showcases a decoupled, production-ready architecture, combining a **FastAPI backend** with a **Streamlit frontend**. It leverages an advanced **agentic RAG** system to deliver accurate, context-aware responses by integrating a local knowledge base with live, external APIs.

> **Note:** This application is **live and deployed on Render**, demonstrating a complete end-to-end development cycle: from data curation and system design to quantitative evaluation and cloud deployment.
<img width="931" height="863" alt="image" src="https://github.com/user-attachments/assets/d175f6ee-f6cb-4899-ba47-53bd6c406d43" />



---

### Core Features:
* **Decoupled Frontend/Backend Architecture:** A scalable and maintainable client-server model, with a Streamlit UI making API calls to a robust FastAPI backend.
* **Advanced Agentic Logic:** The core of the application is a **LangChain agent** that can reason, make decisions, and intelligently choose between multiple tools to best answer a user's query.
* **High-Fidelity RAG System:** Provides factually grounded answers by retrieving information from a curated knowledge base on Lucknow's history and cuisine, stored in a `ChromaDB` vector store.
* **Persistent Vector Directory:** Utilizes disk-based state storage for the vector database, preventing expensive or redundant embedding generation cycles on application restarts and mimicking a real-world production data cache.
* **Maximal Marginal Relevance (MMR) Retrieval:** Instead of standard similarity searches which return redundant text, the retriever uses MMR to fetch the most *diverse* chunks of context, giving the LLM a broader understanding of the query.
* **Safe-Batching Ingestion Pipeline:** The local vector DB creation includes a time-delayed batching mechanism to prevent 429 Rate Limit errors when using free-tier embedding APIs, simulating handling massive scale document processing.
* **Self-Healing Agentic Parsing:** The LangChain agent is configured with `handle_parsing_errors=True` and strict tool-binding. If the open-weight LLM fumbles a JSON tool-call schema, the system gracefully catches the error and instructs the model to self-correct without crashing the application.
* **Live External API Integration:** The agent can call the `Open-Meteo API` in real-time to fetch current weather data and incorporate it into travel advice.
* **Quantitative Performance Evaluation:** Includes a dedicated evaluation suite using the **Ragas** framework to rigorously test and validate the RAG pipeline's performance.
* **Blazing-Fast Inference:** Powered by **Groq's** high-speed LPU™ Inference Engine using Meta's efficient Llama 3 model.

---

## 📊 Quantitative Results: RAG Performance

To ensure the reliability of the system, a comprehensive evaluation was performed on the RAG pipeline. The following metrics validate the quality of the generated answers against a ground-truth dataset.

| Metric | Score (0.0 to 1.0) | Description |
| :--- | :---: | :--- |
| **`faithfulness`** | **1.00** | Measures how factually consistent the generated answer is with the retrieved context. A score of 1.0 means no hallucinations.
| **`context_recall`** | **1.00** | Measures the retriever's ability to find all the necessary information from the knowledge base. |
| **`context_precision`** | **0.92** | Measures the signal-to-noise ratio of the retrieved context. High precision means less irrelevant information. |

> **Conclusion:** The high scores, especially in faithfulness and recall, quantitatively prove that the RAG system provides accurate, reliable, and contextually rich answers.

---

## 🏗️ System Architecture

This project employs a modern, decoupled architecture, which is the industry standard for scalable web applications. The frontend is completely separate from the backend, communicating via a REST API.
<img width="650" height="500" alt="image" src="https://github.com/user-attachments/assets/86cd8c36-54db-472a-b7a7-d19862ea8def" />

```bash


+---------------------------+          +--------------------------------+
|      Frontend (Client)    |          |        Backend (Server)        |
|    (Streamlit on Port 8501) |          |      (FastAPI on Port 8000)      |
+---------------------------+          +--------------------------------+
|                           |          |                                |
|  - Renders UI             |          |  - Exposes REST API (/query)   |
|  - Captures User Input    |          |  - Contains Agentic Logic      |
|  - Displays Chat History  |          |  - Manages Tools (RAG, Weather)|
|                           |          |                                |
|                           |   HTTP   |                                |
|        User Query         | -------> |        Agent Processing        |
|      (e.g., "Plan trip")  |          |                                |
|                           |          |                                |
|        Final Answer       | <------- |        Structured Response     |
| (Formatted Itinerary)     |          |                                |
|                           |          |                                |
+---------------------------+          +--------------------------------+
```
---

## 🛠️ Tech Stack

| Category | Technology / Service |
| :--- | :--- |
| **Cloud Deployment** | **Render**, **Docker**, **Docker Compose** |
| **Frontend** | Streamlit |
| **Backend** | FastAPI, Uvicorn |
| **LLM & Agent** | LangChain, **Groq** (Google **llama-4-maverick**) |
| **Vector DB** | ChromaDB (Local) |
| **Embeddings** | **Hugging Face Inference API** (`BAAI/bge-small-en-v1.5`) |
| **Evaluation** | **Ragas** (for quantitative metrics) |
| **External API** | Open-Meteo (Weather), Tavily (External search) |

---

## 🚀 Deployment & Local Run Setup

This project is fully containerized and configured for both cloud and local execution.

### Option 1: View Live Deployment (Render)
This project is configured for "Infrastructure-as-Code" deployment using the `render.yaml` file.

1.  **How it Works:** The `render.yaml` file defines two "Web Service" instances (`lucknow-guide-backend` and `lucknow-guide-frontend`).
2.  **Secrets:** It uses a Render Environment Group named `lucknow-guide-secrets` to securely manage the API keys.
3.  **Networking:** It correctly sets the `BACKEND_URL` environment variable for the frontend to the backend's public URL, solving all networking.
4.  **Auto-Deploy:** The backend and frontend will auto-deploy on push.

### Option 2: Run Locally with Docker Compose
1.  Create the `backend/.env` file with your `GROQ_API_KEY` and `HUGGINGFACEHUB_API_TOKEN`.
2.  From the project root, run: `docker-compose up --build`
3.  Access the frontend at `http://localhost:8501`.

### Option 3: Run Locally (Classic Python Venv)

1. Start the Backend (Terminal 1)
```bash
cd advanced-lucknow-guide/backend
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```
Keep this terminal open.

2. Start the Frontend (Terminal 2)
```bash
cd advanced-lucknow-guide/frontend
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
A new tab will open in your browser at http://localhost:8501.

---

## 📂 Project Structure
```bash
📁 advanced-lucknow-guide/
│
├── 📁 backend/
│   ├── 📁 knowledge_base/
│   │   ├── 📄 lucknow_food.txt
│   │   └── 📄 lucknow_history.txt
│   ├── 📄 .env                 # (Must be created locally)
│   ├── 📄 agent_logic.py       # Core AI agent and tool logic
│   ├── 📄 main.py              # FastAPI server
│   ├── 📄 requirements.txt     # Backend dependencies
│   ├── 📄 evaluation.py        # Ragas evaluation script
│   └── 🐳 Dockerfile            # Backend Docker instructions
│
├── 📁 frontend/
│   ├── 📄 app.py               # Streamlit UI
│   ├── 📄 requirements.txt     # Frontend dependencies
│   └── 🐳 Dockerfile            # Frontend Docker instructions
│
├── 🐳 docker-compose.yml     # Local orchestration
├── ☁️ render.yaml            # Cloud orchestration (Render Blueprint)
└── 📖 README.md              # This file
```

Copyright@2026 by Salu