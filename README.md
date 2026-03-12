# PDF Parser RAG

End-to-end RAG (Retrieval-Augmented Generation) system built on the [OmniDocBench](https://github.com/opendatalab/OmniDocBench) dataset. Covers the full pipeline from OCR parsing → chunking → embedding → hybrid retrieval → reranking → evaluation.

---

## Architecture

```
OmniDocBench PDFs
       │
       ▼
┌─────────────────┐
│ PaddleOCR-VL    │  Layout detection + OCR (text, table, formula)
│ 1.5 (VL mode)   │
└────────┬────────┘
         │ structured JSON per page
         ▼
┌─────────────────┐
│ Chunk Manager   │  fixed / recursive / parent-child splitting
└────────┬────────┘
         │ all_chunks.json
         ▼
┌─────────────────┐     ┌──────────────┐
│ BGE-M3          │────▶│  ChromaDB    │  dense vector index
│ (embedding)     │     │  (cosine)    │
└─────────────────┘     └──────┬───────┘
                               │
┌─────────────────┐            │ hybrid RRF
│ BM25Store       │────────────┤
└─────────────────┘            │
                               ▼
                    ┌──────────────────┐
                    │ bge-reranker-v2  │  cross-encoder reranking
                    │ -m3              │
                    └────────┬─────────┘
                             │ top-K chunks
                             ▼
                    ┌──────────────────┐
                    │  RAGEvaluator    │  Hit@K, MRR
                    └──────────────────┘
```

---

## Tech Stack

| Layer | Component | Notes |
|-------|-----------|-------|
| OCR | PaddleOCR-VL 1.5 | Layout detection + VL multimodal OCR |
| Chunking | Custom ChunkManager | fixed / recursive / parent-child |
| Embedding | BAAI/bge-m3 | Local, CLS token + L2 normalize |
| Vector Store | ChromaDB | Cosine similarity, persistent |
| Sparse Retrieval | BM25 (rank-bm25) | Full-text keyword search |
| Hybrid Fusion | RRF (Reciprocal Rank Fusion) | Vector + BM25 |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder, sigmoid scoring |
| QA Generation | Qwen3-32B (BNB 4-bit) | Generates eval QA from chunks |
| Evaluation | Hit@K, MRR | Chunk-ID based retrieval evaluation |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download models
```bash
# Embedding
modelscope download --model BAAI/bge-m3 --local_dir models/bge-m3

# Reranker
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir models/bge-reranker-v2-m3

# LLM for QA generation (pre-quantized 4-bit)
modelscope download --model unsloth/Qwen3-32B-bnb-4bit --local_dir models/Qwen3-32B-bnb-4bit
```

### 3. Run full pipeline
```bash
# Full pipeline (OCR → chunk → embed → generate QA → eval)
python scripts/run_full_pipeline.py

# Skip completed stages
python scripts/run_full_pipeline.py --skip-ocr --skip-chunk --skip-embed --skip-gold

# Run specific experiment with prefix
python scripts/run_full_pipeline.py --skip-ocr --skip-chunk --skip-embed --skip-gold --eval-prefix base_rerank
```

### 4. Generate experiment report
```bash
python scripts/generate_report.py
# Output: results/experiment_report.md
```

### 5. Plot metrics
```bash
python scripts/plot_metrics.py
# Output: results/plot/metrics_plot.png, eval_timing_plot.png
```

---

## Configuration

All parameters in `config/config.yaml`:

| Section | Key Parameters |
|---------|---------------|
| `ocr` | PaddleOCR-VL device, layout threshold |
| `chunking` | type (fixed/recursive/parent_child), chunk_size, overlap |
| `embedding` | BGE-M3 local path, batch_size |
| `retrieval` | method (vector/bm25/hybrid), rrf_k |
| `reranker` | enabled, top_r, model path |
| `llm.local` | model_name, batch_size, max_tokens, load_in_4bit |
| `evaluation` | top_k |

---

## Evaluation Results (Baseline)

> Dataset: OmniDocBench | Chunking: fixed (800/200) | Retrieval: vector only | Reranker: off

| Metric | Value |
|--------|-------|
| Hit@1 | 63.5% |
| Hit@3 | 83.0% |
| Hit@5 | 87.0% |
| MRR | 0.73 |
| Embed latency | ~12 ms/query |
| Total latency | ~15 ms/query |

---

## Project Structure

```
pdf_parser_rag/
├── config/
│   └── config.yaml              # All parameters
├── data/
│   ├── raw/OmniDocBench/        # Source PDFs + OmniDocBench.json
│   └── answers/
│       ├── gold_answers.csv     # QA pairs for evaluation
│       └── qa_pairs.jsonl       # Raw generated QA
├── models/                      # Local model weights (not tracked)
├── results/
│   ├── chunk_results/           # all_chunks.json, all_parent_chunks.json
│   ├── ocr_structured/          # Per-page OCR structured JSON
│   ├── plot/                    # PNG plots
│   └── experiment_report.md    # Auto-generated comparison report
├── scripts/
│   ├── run_full_pipeline.py     # Main entry point
│   ├── generate_qa_from_chunks.py  # LLM-based QA generation
│   ├── generate_report.py       # Experiment comparison report
│   ├── plot_metrics.py          # Visualize metrics
│   └── run_evaluator.py         # Standalone evaluator
├── src/
│   ├── ocr_processor.py         # PaddleOCR-VL wrapper
│   ├── chunk_manager.py         # Chunking strategies
│   ├── vector_store.py          # BGE-M3 + ChromaDB
│   ├── bm25_store.py            # BM25 retrieval
│   ├── reranker.py              # BGE cross-encoder reranker
│   ├── evaluator.py             # RAGEvaluator (Hit@K, MRR)
│   └── ocr_evaluator/          # OCR quality metrics
└── vectors/
    └── chroma_db/               # ChromaDB persistent store
```

---

## Experiment Comparison

Run multiple configurations and compare automatically:

```bash
# 1. Baseline (vector only)
python scripts/run_full_pipeline.py --skip-ocr --skip-chunk --skip-embed --skip-gold --eval-prefix base

# 2. With reranker
# config: reranker.enabled: true
python scripts/run_full_pipeline.py --skip-ocr --skip-chunk --skip-embed --skip-gold --eval-prefix base_rerank

# 3. Hybrid retrieval
# config: retrieval.method: hybrid
python scripts/run_full_pipeline.py --skip-ocr --skip-chunk --skip-embed --skip-gold --eval-prefix base_hybrid

# 4. Generate comparison report
python scripts/generate_report.py
```
