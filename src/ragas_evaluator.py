import logging
import pandas as pd
from typing import Optional
from src.utils import load_config

logger = logging.getLogger(__name__)
config = load_config()


# ── LangChain LLM wrapper (reuses loaded Qwen3) ──────────────────────────────
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun


class _LocalQwenLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "local_qwen"

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        from scripts.generate_qa_from_chunks import _generate_with_backend
        return _generate_with_backend(prompt, backend="local", api_client=None)


# ── LangChain Embeddings wrapper (reuses BGE-M3) ─────────────────────────────
from langchain_core.embeddings import Embeddings


class _BGEEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from src.vector_store import VectorStore
        vs = VectorStore()
        results = []
        for i in range(0, len(texts), 32):
            results.extend(vs._embed_text_batch(texts[i: i + 32]))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _build_llm(mode: str):
    if mode == "api":
        from langchain_openai import ChatOpenAI
        api_cfg = config.get("ragas", {}).get("api", {})
        return ChatOpenAI(
            model=api_cfg.get("model", "gpt-4o-mini"),
            base_url=api_cfg.get("base_url") or None,
        )
    return _LocalQwenLLM()


def run_ragas(
    dataset: list[dict],
    mode: str = "local",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run RAGAS evaluation on dataset.

    Each item requires: {"question": str, "answer": str, "contexts": list[str]}
    Metrics: faithfulness, answer_relevancy, context_precision (no ground_truth needed)
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm = LangchainLLMWrapper(_build_llm(mode))
    emb = LangchainEmbeddingsWrapper(_BGEEmbeddings())

    hf_dataset = Dataset.from_list(dataset)
    result = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=emb,
    )
    df = result.to_pandas()
    logger.info(
        "RAGAS scores — faithfulness: %.4f | answer_relevancy: %.4f | context_precision: %.4f",
        df["faithfulness"].mean(),
        df["answer_relevancy"].mean(),
        df["context_precision"].mean(),
    )
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info("RAGAS results saved to %s", output_path)
    return df
