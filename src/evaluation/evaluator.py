import sys
import types

dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

import logging
from typing import Dict

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ollama import chat
from ragas import RunConfig, evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.core.settings import cfg
from src.evaluation.export import export_csv, export_json
from src.ingestion.vectorstore import CloudVectorStoreManager

logger = logging.getLogger(__name__)

# Map config metric names → Ragas metric objects
METRIC_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
    "answer_correctness": answer_correctness,
}


class RAGEvaluator:
    def __init__(self, model_name: str | None = None, embed_model: str | None = None):
        self.model_name = model_name or cfg("models", "llm", default="llama3.2:3b")
        self.embed_model = embed_model or cfg("models", "embed", default="nomic-embed-text")
        self.db_manager = CloudVectorStoreManager()
        self.eval_llm = ChatOllama(model=self.model_name, temperature=0.0)
        self.eval_embeddings = OllamaEmbeddings(model=self.embed_model)

    def _active_metrics(self) -> list:
        names = cfg("evaluation", "metrics", default=list(METRIC_MAP.keys()))
        return [METRIC_MAP[n] for n in names if n in METRIC_MAP]

    def generate_rag_answers(self, question: str, contexts: list) -> str:
        context_str = "\n".join(contexts)
        prompt = f"""Answer the question using ONLY the provided context. If you don't know, say 'I don't know'.

Context: {context_str}

Question: {question}"""

        response = chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content

    def run_evaluation(
        self,
        questions: list,
        contexts: list,
        answers: list,
        top_k: int | None = None,
        use_mmr: bool | None = None,
        use_hybrid: bool | None = None,
        rerank: bool | None = None,
        export: bool = True,
    ):
        logger.info("Running RAG evaluation...")
        top_k = top_k or cfg("retrieval", "top_k", default=2)

        data_for_eval = {
            "user_input": [],
            "response": [],
            "retrieved_contexts": [],
            "reference": [],
        }

        for question, _ctx, ground_truth in zip(questions, contexts, answers):
            retrieved = self.db_manager.query_similarity(
                question,
                limit=top_k,
                use_mmr=use_mmr,
                use_hybrid=use_hybrid,
                rerank=rerank,
            )
            generated = self.generate_rag_answers(question, retrieved)
            data_for_eval["user_input"].append(question)
            data_for_eval["response"].append(generated)
            data_for_eval["retrieved_contexts"].append(retrieved)
            data_for_eval["reference"].append(ground_truth)

        hf_dataset = Dataset.from_dict(data_for_eval)
        runner_config = RunConfig(
            timeout=cfg("evaluation", "timeout", default=600),
            max_workers=cfg("evaluation", "max_workers", default=1),
        )

        result = evaluate(
            dataset=hf_dataset,
            metrics=self._active_metrics(),
            llm=self.eval_llm,
            embeddings=self.eval_embeddings,
            run_config=runner_config,
        )

        if export:
            df = result.to_pandas()
            export_csv(df, "latest_eval.csv")
            scores = {c: float(df[c].mean()) for c in df.columns if c in METRIC_MAP}
            export_json({"scores": scores}, "latest_eval.json")

        return result

    def compare_retrievers(
        self, questions: list, contexts: list, answers: list
    ) -> Dict[str, Dict[str, float]]:
        """
        Run the same QA set through 3 retrieval configs side-by-side.
        """
        configs = [
            ("top_k_2", {"top_k": 2}),
            ("top_k_5", {"top_k": 5}),
            ("mmr_k_5", {"top_k": 5, "use_mmr": True}),
        ]
        comparison: Dict[str, Dict[str, float]] = {}
        for name, kwargs in configs:
            logger.info("Retriever config: %s", name)
            result = self.run_evaluation(
                questions, contexts, answers, export=False, **kwargs
            )
            df = result.to_pandas()
            comparison[name] = {
                c: float(df[c].mean()) for c in df.columns if c in METRIC_MAP
            }

        export_json({"comparison": comparison}, "retriever_comparison.json")
        return comparison

    def compare_naive_vs_agentic(
        self, questions: list, contexts: list, answers: list
    ) -> Dict[str, Dict[str, float]]:
        """

        agentic    — LLM decides when to search, can search multiple times
        """
        from src.agentic.controller import AgenticController

        comparison: Dict[str, Dict[str, float]] = {}

        # --- Naive RAG (baseline) ---
        logger.info("Evaluating naive RAG...")
        naive_result = self.run_evaluation(
            questions, contexts, answers, export=False
        )
        naive_df = naive_result.to_pandas()
        comparison["naive_rag"] = {
            c: float(naive_df[c].mean()) for c in naive_df.columns if c in METRIC_MAP
        }

        # --- Agentic RAG ---
        logger.info("Evaluating agentic RAG...")
        controller = AgenticController()
        agent_data = {
            "user_input": [],
            "response": [],
            "retrieved_contexts": [],
            "reference": [],
        }
        for question, _ctx, ground_truth in zip(questions, contexts, answers):
            result = controller.run(question)
            agent_data["user_input"].append(question)
            agent_data["response"].append(result["answer"])
            # Flatten retrieved search results into context list for Ragas
            flat_contexts = result.get("retrieved_contexts", [])
            agent_data["retrieved_contexts"].append(
                flat_contexts if flat_contexts else ["No context retrieved"]
            )
            agent_data["reference"].append(ground_truth)

        hf_dataset = Dataset.from_dict(agent_data)
        runner_config = RunConfig(
            timeout=cfg("evaluation", "timeout", default=600),
            max_workers=cfg("evaluation", "max_workers", default=1),
        )
        agentic_result = evaluate(
            dataset=hf_dataset,
            metrics=self._active_metrics(),
            llm=self.eval_llm,
            embeddings=self.eval_embeddings,
            run_config=runner_config,
        )
        agent_df = agentic_result.to_pandas()
        comparison["agentic_rag"] = {
            c: float(agent_df[c].mean()) for c in agent_df.columns if c in METRIC_MAP
        }

        export_json({"comparison": comparison}, "naive_vs_agentic.json")
        return comparison
