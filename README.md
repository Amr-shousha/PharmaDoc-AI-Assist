# PharmaDoc-AI-Assist
A hardware-isolated, local RAG engine for pharmaceutical regulatory compliance powered by LlamaIndex, Mistral-7B (GGUF), and Gradio

# 📑 PharmaDoc AI Assist: Intelligent Local RAG Engine

Welcome to **PharmaDoc AI Assist**, a fully integrated, production-ready local Retrieval-Augmented Generation (RAG) workspace. This application is specifically designed to handle sensitive pharmaceutical and regulatory compliance documentation using hardware-isolated, localized semantic extraction. 

By leveraging local LLM execution, it guarantees that no clinical data or proprietary documentation ever leaves your secure environment.

---

## 🚀 Key Features

* **100% Local Ingestion & Generation:** Fully isolated pipeline using high-performance quantized local models, eliminating third-party API dependencies or privacy leaks.
* **Hybrid Structural Metadata Filtering:** Implements rigid `LlamaIndex` structural node-filtering (`doc_type` isolation) to enforce high-precision contextual matching.
* **Advanced Document Chunking:** Orchestrated via LangChain's `RecursiveCharacterTextSplitter` for mathematically sound paragraph overlaps.
* **Production-Grade Workspace UI:** A multi-mode responsive operational interface built entirely using Gradio's advanced layout themes.

---

## 🛠️ Tech Stack & Architecture

* **Orchestration Framework:** `LlamaIndex` (Core & Retrievers)
* **Text Chunking:** `langchain-text-splitters`
* **Local LLM Engine:** `Mistral-7B-Instruct-v0.2` (Quantized via `LlamaCPP` GGUF)
* **Embedding Model:** `BAAI/bge-small-en-v1.5` via HuggingFace
* **UI Interface Workspace:** `Gradio` (Soft Theme Configuration)

---

## 📦 Ingestion, Installation & Usage

Follow these steps to spin up the local production workspace environment on your machine:

### 1. Clone & Install Dependencies
First, install all necessary core packages, vector store dependencies, and UI modules:

```bash
pip install -r requirements.txt
```
2. Fetch the Local Model Weights
Since large model weights (.gguf files) should not be pushed to GitHub, you need to grab the model manually.

Download the Mistral-7B-Instruct-v0.2-GGUF (Q4_K_M) file and place it directly inside the project root directory (the same folder where app.py lives).

3. Launch the Application Server
Run the unified Python execution script to initialize the local embeddings, test CUDA GPU connectivity, and mount the Gradio interface server:
```
python app.py
```
