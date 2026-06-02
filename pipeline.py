import os
import glob
import logging
from src.database import CloudVectorStoreManager
from src.generator import SyntheticDataGenerator
from src.evaluator import RAGEvaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting LLMOps Continuous Evaluation Pipeline...")
    #step 1: Create sample documentation file
    # os.makedirs("data", exist_ok=True)
    # sample_file = "data/auth_service_v2.md"

    # doc_content = """
    # # Auth Service v2.0
    # The new Auth Service v2.0 uses OAuth2 and JWT tokens instead of session cookies. 
    # Tokens expire after 15 minutes by default. 
    # To refresh a token, you must hit the `/api/v2/auth/refresh` endpoint.
    # Admin users have an extended maximum session time of 12 hours.
    # """  

    # with open(sample_file, "w") as f:
    #     f.write(doc_content)    
    # logger.info(f"Document written to {sample_file}")

    # logger.info("Ingesting document into vector store...")
    # db_manager = CloudVectorStoreManager()
    # with open(sample_file, "r") as f:
    #     text = f.read()
    
    # chunks_added = db_manager.add_documents(
    #     doc_id="doc_auth_v2", text=text, 
    #     metadata={"source": sample_file, "version": "v2.0"})

    # #Step 2: Ingest document into vector store
    # logger.info(f"Document ingested with {chunks_added} chunks")


    logger.info("Scanning for documentation files...")
    markdown_files = glob.glob("data/*.md")
    
    if not markdown_files:
        logger.error("No markdown files found in /data directory. Exiting.")
        return

    db = CloudVectorStoreManager()
    full_text = ""

    for file_path in markdown_files:
        with open(file_path, "r") as f:
            text = f.read()
            full_text += text + "\n"
        
        db.add_documents(
            doc_id=file_path.replace("/", "_"), 
            text=text, 
            metadata={"source": file_path}
        )
    logger.info("Successfully ingested dynamic files into Qdrant Cloud.")

    
    logger.info("Generating synthetic QA pairs...")
    generator = SyntheticDataGenerator()

    qa_pairs = generator.generate_qa_pairs(text, num_questions=5)
    logger.info(f"Generated {len(qa_pairs)} QA pairs")

    logger.info("Running RAG evaluation...")
    evaluator = RAGEvaluator()
    results = evaluator.run_evaluation(questions=[pair['question'] for pair in qa_pairs],
                                     contexts=[pair['answer'] for pair in qa_pairs],
                                     answers=[pair['answer'] for pair in qa_pairs])

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    df = results.to_pandas()
    metrics_to_display = ["faithfulness", "answer_relevancy", "context_precision"]
    for metric in metrics_to_display:
        if metric in df.columns:
            print(f"{metric.replace('_',' ').title():<25}: {df[metric].mean()*100:.2f}%")
        else:
            logger.warning(f"Metric '{metric}' not found in results.")
            
    # for metric, score in results.scores.items():
    #     print(f"{metric.replace('_',' ').title():<25}: {score*100:.2f}%")

    # print("\nPipeline execution completed.")

    # if results.get("faithfulness_score", 0) < 0.8:
    #     logger.warning("Faithfulness score below threshold! \
    #         Potential documentation drift detected.")
    # else:
    #     logger.info("Documentation appears to be faithful to the source.")

    if df["faithfulness"].mean() < 0.8:
        logger.warning("Faithfulness score below threshold! \
            Potential documentation drift detected.")
    else:
        logger.info("Documentation appears to be faithful to the source.")

if __name__ == "__main__":
    main()