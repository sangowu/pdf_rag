# RAG项目简明说明书

## 1. 项目需求

使用OmniDocBench数据集建立一个端到端的RAG系统，用于扫描PDF的文字识别和智能检索。

**核心任务：**
- 输入：OmniDocBench中的PDF文件（图片扫描版本）
- 处理：OCR识别 → 文本切片 → 向量化存储 → 检索推理 → 评估
- 输出：量化的检索和生成质量指标（Hit@P、BLEU等）

**数据集特性：** PDF中的文本是基于图像的，需要OCR提取

---

## 2. 技术栈

| 层级 | 组件 | 用途 | 调用方式 |
|-----|------|------|---------|
| **文本提取** | PaddleOCR | 识别扫描PDF中的文字 | API调用 |
| **文本切片** | LangChain RecursiveCharacterTextSplitter | 将长文本分割成chunks | 库调用 |
| **向量化** | Qwen3-Embedding | 将文本转换为向量 | API调用 |
| **向量存储** | ChromaDB | 存储和检索向量 | 库调用 |
| **LLM推理** | vLLM | 基于检索结果生成答案 | 库调用 |
| **评估框架** | 自实现 | 计算Hit@K、BLEU等指标 | 自实现 |
| **数据处理** | pandas, numpy | 数据操作 | 库调用 |

---

## 3. 必须手动实现的模块

这些模块是项目的核心，需要深入理解和自己编写：

### 3.1 PaddleOCR处理器 (`src/ocr_processor.py`)

**需要实现的功能：**
```python
class OCRProcessor:
    def process_pdf(self, pdf_path: str, confidence_threshold: float = 0.8):
        """
        输入：PDF文件路径
        输出：提取的文本 + OCR元数据
        
        关键步骤：
        1. 加载PDF每一页为图像
        2. 用PaddleOCR识别文字
        3. 过滤置信度<threshold的结果
        4. 保留坐标信息用于排版恢复
        5. 生成结构化的文本块
        """
        pass
```

**必须理解的：**
- OCR输出格式：`[[文本, [坐标], 置信度], ...]`
- 置信度阈值的选择（推荐0.8）
- 如何处理多列文本排版混乱
- 如何合并相邻行的文本

### 3.2 Chunk管理器 (`src/chunk_manager.py`)

**需要实现的功能：**
```python
class ChunkManager:
    def generate_chunks(self, file_name: str, text: str, 
                       chunk_size: int = 512, overlap: int = 128):
        """
        输入：文件名、文本
        输出：chunks列表 + chunk_id
        
        关键步骤：
        1. 用LangChain的分割器切片
        2. 生成确定性的chunk_id（file_hash + chunk_index）
        3. 保留metadata：file_name、page_no、chunk_index
        """
        pass
```

**必须理解的：**
- chunk_id的生成规则（必须可重现）
- chunk_size和overlap的影响
- 中文文本的分割符选择
- page_no的提取逻辑

### 3.3 数据对齐检查 (`src/validators.py`)

**需要实现的功能：**
```python
def verify_chunk_id_consistency():
    """检查chunk_id是否可重现"""
    pass

def verify_chromadb_sync():
    """检查ChromaDB中的chunks是否完整"""
    pass

def verify_gold_answers_validity():
    """检查gold_answers中的chunk_id是否都存在"""
    pass
```

**必须理解的：**
- 如何快速定位数据对齐问题
- Hit@K=0时的排查流程
- 数据一致性的检查方法

### 3.4 评估框架 (`src/evaluator.py`)

**需要实现的功能：**
```python
class RAGEvaluator:
    def evaluate(self, query: str, gold_chunk_ids: set, k: int = 5):
        """
        输入：问题、正确的chunk_ids
        输出：Hit@K、答案、质量指标
        
        关键步骤：
        1. 从ChromaDB检索top-k
        2. 计算Hit@K（重叠比例）
        3. 用vLLM生成答案
        4. 计算答案质量（BLEU/RougeL）
        """
        pass
```

**必须理解的：**
- Hit@K的准确定义和计算
- Prompt工程（如何组织检索结果给LLM）
- 答案质量指标的含义

### 3.5 金标准构建 (`scripts/build_gold_answers.py`)

**需要实现的功能：**
```python
def build_gold_answers_from_omnidoc(omnidoc_json_path: str, 
                                    retriever, 
                                    output_csv_path: str):
    """
    输入：OmniDocBench.json + ChromaDB检索器
    输出：gold_answers.csv（问题 → 正确chunks）
    
    关键步骤：
    1. 解析OmniDocBench.json中的QA对
    2. 对每个answer，在ChromaDB中查找对应chunks
    3. 处理特殊情况（answer跨chunk、OCR错误等）
    4. 生成gold_chunk_ids列表
    """
    pass
```

**必须理解的：**
- OmniDocBench的数据schema
- 文本匹配的容差度（处理OCR错误）
- 页码对齐的逻辑

---

## 4. 可以使用外部接口的模块

这些模块可以直接调用库，不需要自己实现。但要理解它们的参数和输出。

### 4.1 文本切片 - LangChain

**需要理解：**
- chunk_size=512是为了OmniDoc中文文档的语义完整性
- overlap=128避免分割线处信息丢失
- 分割符的优先级（先按段落，再按句子，最后按字符）

### 4.2 向量化 - Qwen3-Embedding

**需要理解：**
- 向量维度是3072
- 如何批量调用API
- API的速率限制

### 4.3 向量存储 - ChromaDB

**需要理解：**
- metadata的设计（包括ocr_confidence）
- 相似度计算方式（余弦相似度）

### 4.4 LLM推理 - vLLM或transformers

**需要理解：**
- vLLM相比transformers的优势（速度、显存）
- 如何构建prompt


---

## 5. MVP原则

### 第一版本（必做）

**包括：**
- PaddleOCR文本提取
- LangChain文本切片
- ChromaDB向量存储（不要CSV中间层）
- vLLM推理
- Hit@P评估指标

### 完成标准

第一版本完成的标志：
- ✅ 能处理100+个PDF文件
- ✅ all_chunks在ChromaDB中 ≥500条
- ✅ gold_answers.csv有效率 ≥80%
- ✅ 平均Hit@P > 0（至少检索到部分chunks）

---

## 6. 评估方法

### 6.1 Hit@P（检索质量）

### 6.2 答案质量（可选）

使用BLEU或RougeL对比生成答案与gold答案

### 6.3 数据对齐检查

每次运行前检查：
```python
# 1. chunk_id可重现吗？
verify_chunk_id_consistency()

# 2. ChromaDB与all_chunks同步吗？
verify_chromadb_sync()

# 3. gold_answers的chunk_id有效吗？
verify_gold_answers_validity()
```

---

## 7. 管道跑通后的可选优化方向

**这些是后续改进方向，不要在MVP阶段做：**

### 7.1 检索优化

- **调整chunk_size：** 影响检索粒度
  - 当前512，可尝试256或1024
  - 监控Hit@P变化

- **多路召回：** 组合多个检索策略
  - BM25 + 向量相似度
  - 层级检索（先粗召回，再精排）

- **Reranking：** 用更强的模型重排检索结果
  - 使用交叉编码器（Cross-Encoder）
  - 提高top-1的精度

### 7.2 生成优化

- **Prompt优化：** 改进指令工程
  - 添加more-shot examples
  - 优化检索结果的组织方式

- **模型切换：** 尝试更强的LLM
  - 监控生成质量变化

### 7.3 OCR优化

- **置信度阈值调优：** 

- **排版恢复：** 使用坐标信息重新排序
  - 处理多列文本

### 7.4 系统优化

- **服务化：** 添加FastAPI接口
  - 提供HTTP API
  - 支持并发查询

- **缓存：** 缓存检索结果和生成答案
  - 加快重复查询

- **批处理：** 批量处理多个查询
  - 提高吞吐量

---


# 8. 项目结构

```
project/
├── data/
│   ├── raw/                      # OmniDoc PDFs
│   ├── answers/
│   │   └── gold_answers.csv      # 金标准（QA对）
│   └── config.yaml               # 配置文件
│
├── src/
│   ├── __init__.py
│   ├── ocr_processor.py           # PaddleOCR处理
│   ├── chunk_manager.py           # Chunk管理
│   ├── evaluator.py               # 评估框架
│   ├── validators.py              # 数据对齐检查
│   └── utils.py                   # 工具函数
│
├── scripts/
│   ├── run_ocr_and_chunking.py    # 第1阶段：OCR + 切片 → ChromaDB
│   ├── build_gold_answers.py      # 第2阶段：构建金标准
│   └── run_rag_test.py            # 第3阶段：RAG评估
│
├── vectors/
│   └── chroma_db/                 # ChromaDB数据库
│
├── results/
│   ├── metrics.csv                # 评估结果
│   └── logs/
│
├── requirements.txt
├── README.md
└── config.yaml                    # 配置文件
```

## 总结

这个项目的核心是：
1. **PaddleOCR** - 解决"扫描PDF识别"问题
2. **数据对齐** - 解决"chunk_id一致性"问题
3. **Hit@K评估** - 量化检索效果

第一版本的目标是**跑通完整流程**，验证这个方案是否可行。

之后才考虑优化。
