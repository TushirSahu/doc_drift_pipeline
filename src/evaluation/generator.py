import logging
from typing import List

from ollama import chat
from pydantic import BaseModel

from src.core.settings import cfg

logger = logging.getLogger(__name__)


class QAPair(BaseModel):
    question: str
    answer: str


class QADataset(BaseModel):
    qa_pairs: List[QAPair]


class SyntheticDataGenerator:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or cfg("models", "llm", default="llama3.2:3b")

    def generate_qa_pairs(self, text: str, num_questions: int | None = None) -> List[dict]:
        num_questions = num_questions or cfg("evaluation", "num_questions", default=5)
        prompt = f"""You are an expert technical writer and AI evaluator.
Read the following documentation text and generate exactly {num_questions}
question-and-answer pairs based strictly on the provided text.

Text:
{text}"""

        try:
            logger.info("Generating %d QA pairs", num_questions)
            response = chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format=QADataset.model_json_schema(),
                options={"temperature": 0.0},
            )
            dataset = QADataset.model_validate_json(response.message.content)
            return [pair.model_dump() for pair in dataset.qa_pairs]
        except Exception as e:
            logger.error("QA generation failed: %s", e)
            return []
