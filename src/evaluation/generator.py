import json
import logging
import re
from typing import List

from pydantic import BaseModel

from src.core import llm

logger = logging.getLogger(__name__)


class QAPair(BaseModel):
    question: str
    answer: str


class QADataset(BaseModel):
    qa_pairs: List[QAPair]


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model reply (handles code fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 else text


class SyntheticDataGenerator:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name  # None → llm.chat resolves per provider

    def generate_qa_pairs(self, text: str, num_questions: int | None = None) -> List[dict]:
        from src.core.settings import cfg

        num_questions = num_questions or cfg("evaluation", "num_questions", default=5)
        prompt = f"""You are an expert technical writer and AI evaluator.
Read the documentation below and generate exactly {num_questions} question-and-answer
pairs based strictly on the text. Reply with ONLY a JSON object of this shape:
{{"qa_pairs": [{{"question": "...", "answer": "..."}}]}}

Documentation:
{text}"""

        try:
            logger.info("Generating %d QA pairs", num_questions)
            reply = llm.chat([{"role": "user", "content": prompt}], model=self.model_name)
            data = json.loads(_extract_json(reply))
            dataset = QADataset.model_validate(data)
            return [pair.model_dump() for pair in dataset.qa_pairs]
        except Exception as e:  # noqa: BLE001 — generation is best-effort
            logger.error("QA generation failed: %s", e)
            return []
