# RAG 项目简明说明书

## 1. 项目概述

基于 OmniDocBench 数据集构建端到端 RAG 系统，评估不同 OCR 解析、Chunking 策略和检索配置对检索质量的影响。

**核心流程：**
```
PDF → OCR（PaddleOCR-VL）→ Chunking → Embedding → Hybrid Retrieval → Reranking → 评估
```

---

## 2. 技术栈

| 层级 | 组件 | 说明 |
|-----|------|------|
| **OCR** | PaddleOCR-VL 1.5 | 版面检测 + VL 多模态 OCR，支持文字/表格/公式 |
| **Chunking** | 自实现 ChunkManager | fixed / recursive / parent-child 三种策略 |
| **Embedding** | BAAI/bge-m3 | 本地推理，CLS token + L2 normalize，dim=1024 |
| **向量存储** | ChromaDB | cosine 相似度，持久化 |
| **稀疏检索** | BM25 (rank-bm25) | 关键词全文检索 |
| **混合检索** | RRF 融合 | vector + BM25 互补 |
| **重排序** | BAAI/bge-reranker-v2-m3 | cross-encoder，sigmoid 打分 |
| **QA 生成** | Qwen3-32B BNB 4-bit | 从 chunk 自动生成评测用 QA 对 |
| **评估** | Hit@K、MRR | 基于 chunk_id 的检索评估 |

---

## 3. 核心模块说明

### 3.1 OCR 处理器 (`src/ocr_processor.py`)
- 调用 PaddleOCR-VL pipeline，输出结构化 JSON（每页包含 text/table/figure_caption 等 block）
- 支持批量处理，结果缓存避免重复推理

### 3.2 Chunk 管理器 (`src/chunk_manager.py`)
- **fixed**: 固定大小滑动窗口（chunk_size=800, overlap=200）
- **recursive**: 按语义分隔符递归切分（中英文混合友好）
- **parent_child**: 大 chunk 索引 + 小 chunk 检索，平衡精度与上下文

### 3.3 向量存储 (`src/vector_store.py`)
- BGE-M3 本地推理（bfloat16）
- 建索引和查询均使用 CLS token + L2 normalize（必须一致）
- ChromaDB cosine 距离检索

### 3.4 重排序器 (`src/reranker.py`)
- BGE-reranker-v2-m3：AutoModelForSequenceClassification
- 先检索 top_r（默认 20），reranker 重排后返回 top evaluation.top_k

### 3.5 QA 生成 (`scripts/generate_qa_from_chunks.py`)
- 自动检测 chunk 语言（中文比例 >15% → 中文 prompt，否则英文 prompt）
- 预过滤低质量 chunk（< 50 字符、纯标题）
- post-validation：answer 必须是 chunk 原文子串，语言须一致
- Qwen3-32B BNB 4-bit + Flash Attention 2 + batch generation

### 3.6 评估框架 (`src/evaluator.py`)
- 纯检索指标：Hit@K、MRR（无 LLM）
- 支持 vector / bm25 / hybrid 三种检索方式
- 延迟分阶段统计（embed/chroma/rerank/total）

---

## 4. 评估结果（基线）

> 配置：fixed chunking (800/200) | BGE-M3 vector only | reranker off

| 指标 | 数值 |
|------|------|
| Hit@1 | 63.5% |
| Hit@3 | 83.0% |
| Hit@5 | 87.0% |
| MRR | 0.73 |
| 检索延迟 | ~15 ms/query |

---

## 5. 项目结构

```
pdf_parser_rag/
├── config/config.yaml           # 所有参数配置
├── data/
│   ├── raw/OmniDocBench/        # 数据集
│   └── answers/gold_answers.csv # 评测 QA 对
├── models/                      # 本地模型权重
├── results/                     # 评测结果、图表、报告
├── scripts/
│   ├── run_full_pipeline.py     # 主入口
│   ├── generate_qa_from_chunks.py
│   ├── generate_report.py       # 实验对比报告
│   └── plot_metrics.py
├── src/
│   ├── ocr_processor.py
│   ├── chunk_manager.py
│   ├── vector_store.py          # BGE-M3 + ChromaDB
│   ├── bm25_store.py
│   ├── reranker.py              # BGE cross-encoder
│   └── evaluator.py             # Hit@K, MRR
└── vectors/chroma_db/           # ChromaDB 持久化
```

---

## 6. 待实现

- RAG 端到端答案生成 + EM/ROUGE-L 评测
- Gradio 交互 Demo
