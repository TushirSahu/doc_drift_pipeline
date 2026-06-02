# 🚀 DocDrift: Automated Documentation Drift Detection

> An LLMOps pipeline designed to detect semantic drift in technical documentation before it breaks your production RAG systems.

DocDrift automates the validation of technical documentation. It ingests new documentation, generates synthetic ground-truth data, and runs a localized Retrieval-Augmented Generation (RAG) evaluation to ensure your AI systems can still accurately interpret your updated docs.

## ✨ Core Capabilities

* **Synthetic Data Generation:** Automatically generates high-quality QA pairs directly from raw markdown using local LLMs.
* **Semantic Vector Retrieval:** Embeds and retrieves context using modern dense vector search via Qdrant Cloud.
* **Continuous LLM Evaluation:** Strictly scores RAG performance using the **Ragas** framework to monitor *Faithfulness*, *Answer Relevancy*, and *Context Precision*.
* **CI/CD Integration Ready:** Designed to run in automated pipelines (like GitHub Actions) to block Pull Requests if documentation updates cause RAG performance to drop below acceptable thresholds.

## 🛠️ Tech Stack

* **Orchestration:** LangChain
* **Evaluation Framework:** Ragas
* **Vector Database:** Qdrant Cloud
* **Local LLM Engine:** Ollama (`llama3.2:1b`)
* **Embedding Model:** Nomic AI (`nomic-embed-text`)

---

## 🚀 Quick Start

### 1. Environment Setup

Clone the repository and spin up a clean, isolated environment to prevent dependency conflicts:

```bash
git clone [https://github.com/TushirSahu/doc_drift_pipeline.git](https://github.com/TushirSahu/doc_drift_pipeline.git)
cd doc_drift_pipeline

conda create -n doc-drift python=3.11 -y
conda activate doc-drift

pip install -r requirements.txt