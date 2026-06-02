import sys
import types
import logging
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat
# ---------------------------------------------
import pandas as pd
from datasets import Dataset
from ollama import chat
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from src.database import CloudVectorStoreManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, model_name:str = "llama3.2:1b", embed_model:str = "nomic-embed-text"):
        self.model_name = model_name
        self.db_manager = CloudVectorStoreManager()
        self.eval_llm = ChatOllama(model=model_name)
        self.eval_embeddings = OllamaEmbeddings(model=embed_model)

    def generate_rag_answers(self, question: str, contexts: list) -> str:
        context_str = "\n".join(contexts)
        prompt = f"""
        Answer the question using ONLY the provided context. If you don't know, say 'I don't know'.
        
        Context: {context_str}
        
        Question: {question}
        """

        response = chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],)
        
        return response.message.content

    def run_evaluation(self, questions: list, contexts: list, answers: list) -> dict:
        logger.info("Running RAG evaluation...")

        data_for_eval = {
            "question": [],
            "answer": [],
            "context": [],
            "ground_truth": []
        }

        for pair in zip(questions, contexts, answers):
            question = pair[0]
            ground_truth = pair[2]

            retrieved_contexts = self.db_manager.query_similarity(question, limit=2)

            generated_answer = self.generate_rag_answers(question, retrieved_contexts)

            data_for_eval["question"].append(question)
            data_for_eval["answer"].append(generated_answer)
            data_for_eval["context"].append(retrieved_contexts)
            data_for_eval["ground_truth"].append(ground_truth)

        
        hf_dataset= Dataset.from_dict(data_for_eval)

        logger.info("Calculating metrics...")
        result = evaluate(
            dataset=hf_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=self.eval_llm,
            embeddings=self.eval_embeddings
        )
        return result