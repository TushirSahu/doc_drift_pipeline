import os 
import json
import logging
from typing import List
from pydantic import BaseModel
from ollama import chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAPair(BaseModel):
    question: str
    answer: str

class QADataset(BaseModel):
    qa_pairs: List[QAPair]

class SyntheticDataGenerator:
    def __init__(self,model_name:str = "llama3"):
        self.model_name = model_name

    def generate_qa_pairs(self,text:str,num_questions:int = 5) -> List[dict]:
        prompt = f"""
        You are an expert technical writer and AI evaluator. 
        Read the following documentation text and generate exactly {num_questions} 
        question-and-answer pairs based strictly on the provided text.
        
        Text:
        {text}
        """
        try:
            logger.info(f"Generating {num_questions} QA pairs using model '{self.model_name}'")

            response = chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format = QADataset.model_json_schema(),
                options = {"temperature":0.0}
            )
            dataset = QADataset.model_validate_json(response.message.content)
            return [pair.model_dump() for pair in dataset.qa_pairs]
        except Exception as e:
            logger.error(f"Error occurred while generating QA pairs: {e}")
            return []

if __name__ == "__main__":
    sample_text = "Qdrant Cloud provides a free tier cluster with 1GB of RAM. It uses cosine similarity by default for vector distances."
    generator = SyntheticDataGenerator()
    
    print("Testing generator...")
    pairs = generator.generate_qa_pairs(sample_text, num_questions=2)
    print(json.dumps(pairs, indent=2))