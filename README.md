# Documentation Drift Detection Pipeline

An LLMOps system that detects documentation drift by generating synthetic QA pairs and evaluating RAG system performance using multiple quality metrics.

## What It Does

Automatically validates technical documentation by:
- Generating synthetic QA pairs from documentation using Ollama
- Retrieving context via vector similarity search (Qdrant)
- Evaluating answers on Faithfulness, Relevancy, and Precision
- Alerting when documentation becomes outdated

## Tech Stack

**Vector DB**: Qdrant | **LLM**: Ollama (llama3) | **Embeddings**: Local embedder | **Evaluation**: RAGAS | **Framework**: LangChain

## Quick Start

```bash
# Setup
git clone https://github.com/TushirSahu/doc_drift_pipeline.git
cd doc_drift_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure
echo "QDRANT_URL=http://localhost:6333" > .env
echo "QDRANT_API_KEY=your_key" >> .env

# Run
ollama run llama3  # in another terminal
python pipeline.py
```

## Output Example

```
EVALUATION RESULTS
Faithfulness Score      : 87.50%
Answer Relevancy        : 92.30%
Context Precision       : 88.75%
```

## Key Components

- **database.py** — Qdrant vector store management & semantic search
- **generator.py** — Synthetic QA pair generation via Ollama
- **evaluator.py** — RAG evaluation using RAGAS metrics
- **pipeline.py** — Main orchestration logic
