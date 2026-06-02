import logging
import ollama
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbedder:
    def __init__(self,model_name:str = "nomic-embed-text"):
        self.model_name = model_name
        self.dimensions = 768
    
    def get_embeddings(self,text:str) -> List[float]:
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            return response['embedding']
        except Exception as e:
            logger.error(f"Error occurred while generating embedding for text: {text}")
            raise e
    
    def get_embeddings_batch(self,texts: List[str]) -> List[List[float]]:
        return [self.get_embeddings(text) for text in texts]
