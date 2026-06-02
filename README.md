# Documentation Drift Detection Pipeline

An LLMOps system designed to detect semantic drift in technical documentation. It works by generating synthetic QA pairs from your documentation and evaluating Retrieval-Augmented Generation (RAG) performance using standard quality metrics.

## Overview

This pipeline automatically validates technical documentation to ensure it stays up-to-date. Key capabilities include:
- **Data Synthesis**: Generates synthetic QA pairs directly from text using Ollama.
- **Vector Retrieval**: Performs semantic similarity searches to retrieve relevant context using Qdrant.
- **Metric Evaluation**: Uses RAGAS to strictly score answers on *Faithfulness*, *Relevancy*, and *Precision*.
- **Drift Alerting**: Flags documentation when it becomes outdated or misaligned with expected answers.

## Tech Stack

- **Vector Database**: Qdrant
- **LLM**: Ollama (`llama3`)
- **Embeddings**: Local dense embeddings (e.g., `nomic-embed-text`)
- **Evaluation**: RAGAS
- **Framework**: LangChain

## Quick Start

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/TushirSahu/doc_drift_pipeline.git
cd doc_drift_pipeline

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with your Qdrant configuration:

```bash
echo "QDRANT_URL=http://localhost:6333" > .env
echo "QDRANT_API_KEY=your_key" >> .env
```

### 3. Execution

Ensure your Ollama service is running in the background with your target model:

```bash
ollama run llama3
```

Run the main pipeline:

```bash
python pipeline.py
```

## Output Example

```text
==================================================
EVALUATION RESULTS
==================================================
Faithfulness Score      : 87.50%
Answer Relevancy        : 92.30%
Context Precision       : 88.75%
```

## Repository Structure

- `src/database.py`: Qdrant vector store management and semantic search
- `src/generator.py`: Synthetic QA pair generation via Ollama
- `src/evaluator.py`: RAG evaluation using RAGAS metrics
- `pipeline.py`: Main execution and orchestration logic
