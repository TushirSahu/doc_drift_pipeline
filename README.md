# 🕵️‍♂️ Documentation Drift Detection Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black)](https://ollama.com/)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-red)](https://qdrant.tech/)
[![Ragas](https://img.shields.io/badge/Evaluation-RAGAS-green)](https://docs.ragas.io/)

An **LLMOps system** designed to detect documentation drift by generating synthetic QA pairs and evaluating RAG (Retrieval-Augmented Generation) system performance using multiple quality metrics.

## 🎯 What It Does

Automatically validates technical documentation to ensure it stays up-to-date by:
- 🤖 **Data Synthesis**: Generating synthetic QA pairs directly from documentation using Ollama.
- 🔍 **Vector Retrieval**: Retrieving relevant context via semantic similarity search in Qdrant.
- 📊 **Metric Evaluation**: Scoring answers strictly on *Faithfulness*, *Relevancy*, and *Precision*.
- 🚨 **Drift Alerting**: Automatically flagging documentation when it becomes outdated or misaligned.

## 🛠️ Tech Stack

**Vector DB**: Qdrant | **LLM**: Ollama (`llama3`) | **Embeddings**: Local embedder | **Evaluation**: RAGAS | **Framework**: LangChain

## 🚀 Quick Start

```bash
# 1. Clone & Setup
git clone https://github.com/TushirSahu/doc_drift_pipeline.git
cd doc_drift_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure Environment
echo "QDRANT_URL=http://localhost:6333" > .env
echo "QDRANT_API_KEY=your_key" >> .env

# 3. Run Pipeline (Ensure Ollama is running in background)
ollama run llama3  # Run in a separate terminal
python pipeline.py
```

## 📈 Output Example

```text
==================================================
EVALUATION RESULTS
==================================================
Faithfulness Score      : 87.50%
Answer Relevancy        : 92.30%
Context Precision       : 88.75%
```

## 🧩 Key Components

- `database.py` — Qdrant vector store management & semantic search
- `generator.py` — Synthetic QA pair generation via Ollama
- `evaluator.py` — RAG evaluation using RAGAS metrics
- `pipeline.py` — Main execution and orchestration logic
