import logging
import os
import pandas as pd
from typing import Optional
from src.utils import load_config
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
config = load_config()


# ── LangChain LLM wrapper (reuses loaded local Qwen3) ────────────────────────
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
        api_key = os.getenv("MODELSCOPE_API_KEY") or api_cfg.get("api_key", "")
        return ChatOpenAI(
            model=api_cfg.get("model", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
            base_url=api_cfg.get("base_url", "https://api-inference.modelscope.cn/v1"),
            api_key=api_key,
        )
    return _LocalQwenLLM()


def run_ragas(
    dataset: list[dict],
    mode: str = "api",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run RAGAS evaluation on dataset.

    Each item requires: {"question": str, "answer": str, "contexts": list[str]}
    Metrics: faithfulness, answer_relevancy (no ground_truth needed; 2 calls per item)
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig

    llm = LangchainLLMWrapper(_build_llm(mode))
    emb = LangchainEmbeddingsWrapper(_BGEEmbeddings())

    if mode == "api":
        # API can handle some concurrency; keep low to avoid rate limits
        run_config = RunConfig(max_workers=2, timeout=60, max_retries=3)
    else:
        # Local GPU: must be serial, long timeout
        run_config = RunConfig(max_workers=1, timeout=600)

    hf_dataset = Dataset.from_list(dataset)
    result = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=emb,
        run_config=run_config,
    )
    df = result.to_pandas()
    for col in ["faithfulness", "answer_relevancy"]:
        if col in df.columns:
            logger.info("RAGAS %s: %.4f", col, df[col].mean())
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info("RAGAS results saved to %s", output_path)
    return df
