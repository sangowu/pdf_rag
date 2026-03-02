# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end RAG (Retrieval-Augmented Generation) system for OCR-based PDF document understanding using the OmniDocBench dataset. The pipeline: scanned PDF → PaddleOCR → text chunking → Qwen3-Embedding vectorization → ChromaDB storage → vLLM inference → Hit@K evaluation.

Primary language: Python. Target documents: Chinese-language scanned PDFs.

## Setup & Commands

```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download PaddleOCR models (first time)
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='ch')"

# Pipeline execution (3 stages, run in order)
python scripts/run_ocr_and_chunking.py --pdf_dir data/raw/ --output_db vectors/chroma_db --chunk_size 512 --confidence_threshold 0.8
python scripts/build_gold_answers.py --omnidoc_json data/raw/OmniDocBench/annotations/OmniDocBench.json --db_dir vectors/chroma_db --output_csv data/answers/gold_answers.csv
python scripts/run_rag_test.py --gold_answers data/answers/gold_answers.csv --db_dir vectors/chroma_db --num_samples 10 --output_csv results/metrics.csv

# Lint & format
black src/ scripts/
pylint src/ scripts/
```

## Architecture

### Three-Stage Pipeline

1. **OCR + Chunking** (`scripts/run_ocr_and_chunking.py`): PDF → PaddleOCR text extraction → LangChain chunking → Qwen3-Embedding → ChromaDB
2. **Gold Standard** (`scripts/build_gold_answers.py`): OmniDocBench.json QA pairs → match against ChromaDB chunks → `gold_answers.csv`
3. **RAG Evaluation** (`scripts/run_rag_test.py`): Query ChromaDB → compute Hit@K → vLLM answer generation → BLEU/RougeL → `metrics.csv`

### Core Modules (src/)

| Module | Class/Function | Purpose |
|--------|---------------|---------|
| `ocr_processor.py` | `OCRProcessor` | PaddleOCR wrapper: PDF→images→text with confidence filtering |
| `chunk_manager.py` | `ChunkManager` | Text splitting (LangChain) + deterministic chunk_id generation |
| `validators.py` | `verify_*` functions | Data alignment checks: chunk_id consistency, ChromaDB sync, gold_answers validity |
| `evaluator.py` | `RAGEvaluator` | Hit@K computation, vLLM answer generation, BLEU/RougeL scoring |
| `utils.py` | Helpers | Shared utility functions |

### Key Data Flow

- **Chunk IDs** must be deterministic (`file_hash + chunk_index`) for reproducibility
- **ChromaDB** is the single source of truth for chunks — no CSV intermediate layer
- **Metadata** per chunk: `file_name`, `page_no`, `chunk_index`, `ocr_confidence`
- **Vector dimension**: 3072 (Qwen3-Embedding output)

## Critical Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| OCR confidence threshold | 0.8 | Filters low-quality results |
| chunk_size | 512 | Tuned for Chinese document semantics |
| chunk_overlap | 128 | Prevents boundary information loss |
| Chinese separators | `["\n\n", "\n", "。", "，", "、", ""]` | Priority order for text splitting |
| Retrieval top-k | 5 | Default for Hit@K evaluation |

## Tech Stack

- **OCR**: PaddleOCR (lang='ch' for Chinese)
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`
- **Embeddings**: Qwen3-Embedding (3072-dim, API-based)
- **Vector DB**: ChromaDB (cosine similarity)
- **LLM**: vLLM (local inference)
- **Evaluation**: NLTK (BLEU), Rouge (RougeL)

## MVP Scope

**In scope**: OCR extraction, chunking, ChromaDB storage, vLLM inference, Hit@K evaluation, CSV output only.

**Out of scope**: FastAPI service, CSV intermediate chunk storage, multi-format output, advanced reranking.

**Success criteria**: 100+ PDFs processed, ≥500 chunks in ChromaDB, gold_answers validity ≥80%, Hit@K > 0.

## Troubleshooting

When Hit@K = 0, check in order: (1) `verify_gold_answers_validity()` — are gold chunk_ids in ChromaDB? (2) `verify_chromadb_sync()` — enough chunks? (3) Query embedding quality — inspect raw query results.

When gold_answers validity is low: likely OCR text mismatch with OmniDocBench answers — adjust text matching tolerance.
