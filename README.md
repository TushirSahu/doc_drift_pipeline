# Documentation Drift Detection Pipeline

A comprehensive LLMOps pipeline for detecting documentation drift and evaluating Retrieval-Augmented Generation (RAG) system performance through automated synthetic QA generation and semantic evaluation metrics.

## Overview

This project addresses a critical challenge in AI systems: **documentation drift**—when software documentation becomes outdated or misaligned with actual implementation. The pipeline automatically generates synthetic question-answer pairs from documentation, retrieves relevant context using vector similarity search, and evaluates RAG system faithfulness through multiple quality metrics.

## Key Features

- **Automated QA Synthesis**: Generates high-quality synthetic question-answer pairs directly from documentation using Ollama LLM
- **Vector-Based Retrieval**: Leverages Qdrant vector database with local embeddings for semantic document retrieval
- **Multi-Metric Evaluation**: Assesses RAG performance using:
  - Faithfulness Score (does the answer align with source?)
  - Answer Relevancy (is the answer relevant to the question?)
  - Context Precision (how precise is the retrieved context?)
- **Drift Detection**: Automatically flags documentation when faithfulness falls below configurable thresholds
- **Production-Ready**: Includes error handling, structured logging, and integration with Ollama for offline LLM inference

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Vector Database** | Qdrant (cloud or local) |
| **Embeddings** | Local embedder (customizable) |
| **LLM Inference** | Ollama (llama3, nomic-embed-text) |
| **RAG Evaluation** | RAGAS framework |
| **Orchestration** | LangChain Community |
| **Data Processing** | Pydantic, Pandas, Hugging Face Datasets |

## Project Structure

```
doc_drift_pipeline/
├── pipeline.py                 # Main execution pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/                       # Sample documentation files
│   └── auth_service_v2.md      # Example technical documentation
└── src/
    ├── database.py             # CloudVectorStoreManager - Qdrant integration
    ├── embedder.py             # LocalEmbedder - Vector embeddings
    ├── generator.py            # SyntheticDataGenerator - QA pair creation
    └── evaluator.py            # RAGEvaluator - Performance metrics calculation
```

## Installation

### Prerequisites
- Python 3.10+
- Ollama installed and running locally
- Qdrant instance (cloud or local)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/TushirSahu/doc_drift_pipeline.git
   cd doc_drift_pipeline
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   QDRANT_URL=http://localhost:6333  # or your Qdrant cloud URL
   QDRANT_API_KEY=your_api_key_here
   ```

5. **Start Ollama**
   ```bash
   ollama run llama3  # Downloads and runs the model
   ```

## Usage

### Run the Pipeline

```bash
python pipeline.py
```

**Output:**
- Ingests a sample auth service documentation into the vector store
- Generates 5 synthetic QA pairs
- Evaluates answers using RAGAS metrics
- Displays faithfulness, relevancy, and precision scores
- Alerts if documentation drift is detected (faithfulness < 80%)

### Example Output
```
==================================================
EVALUATION RESULTS
==================================================
Faithfulness Score      : 87.50%
Answer Relevancy        : 92.30%
Context Precision       : 88.75%

Pipeline execution completed.
```

## Core Components

### CloudVectorStoreManager (`src/database.py`)
- Manages document ingestion into Qdrant
- Chunks text with configurable overlap
- Performs semantic similarity search
- Handles collection creation and management

### SyntheticDataGenerator (`src/generator.py`)
- Generates QA pairs using Ollama with JSON schema formatting
- Produces structured question-answer pairs from any documentation
- Configurable number of questions per document

### RAGEvaluator (`src/evaluator.py`)
- Retrieves relevant context using vector similarity
- Generates RAG answers with grounding in retrieved context
- Computes RAGAS evaluation metrics
- Identifies documentation gaps and inconsistencies

## Configuration

Customize pipeline behavior by editing `pipeline.py`:

```python
qa_pairs = generator.generate_qa_pairs(text, num_questions=5)  # Adjust QA pairs
results = evaluator.run_evaluation(qa_pairs)  # Evaluation runs automatically

if results.get("faithfulness_score", 0) < 0.8:  # Adjust threshold
    logger.warning("Documentation drift detected")
```

## Potential Improvements

- [ ] Support for multiple document formats (PDF, HTML, JSON)
- [ ] Batch processing for large documentation sets
- [ ] CI/CD integration for continuous drift detection
- [ ] Web dashboard for visualization of evaluation metrics
- [ ] Feedback loop to retrain embedders on domain-specific documents
- [ ] Multi-language support

## Use Cases

- **Documentation Quality Assurance**: Automated testing of technical documentation accuracy
- **API Documentation Validation**: Ensure API docs match actual endpoints and responses
- **Knowledge Base Maintenance**: Identify outdated information in support documentation
- **Compliance Verification**: Validate that compliance documentation remains aligned with implementation
- **Internal Knowledge**: Keep internal wikis and runbooks synchronized with actual systems

## Results & Metrics

The pipeline provides actionable insights:
- **Faithfulness Score**: Measures how well answers ground to source material
- **Answer Relevancy**: Evaluates answer quality relative to the question
- **Context Precision**: Assesses retrieval system quality

These metrics enable continuous monitoring of documentation accuracy without manual review.

## Author

**Tushir Sahu**

## License

MIT License - See LICENSE file for details
