# RAG 系统初学者开发指南

> 本指南采用**循序渐进**的开发方式，从数据探索开始，逐步实现每个核心模块。每完成一个模块，确认功能无误后再进行下一步。

---

## 第一阶段：环境准备 & 数据探索

### 目标
- 搭建开发环境
- 理解 OmniDocBench 数据集结构
- 确认 PDF 的实际特征

### 步骤 1.1：环境配置(已完成)
**你需要做：**
- 创建虚拟环境（可选但推荐）
- 安装 requirements.txt 中的依赖
- 准备项目文件夹结构

**检查清单：**
- [ ] 已安装：paddlepaddle、paddleocr、chromadb、langchain、vllm、pandas
- [ ] 已创建项目文件夹结构（参考项目简明说明书第 8 部分）

---

### 步骤 1.2：数据探索脚本
**你需要创建：** `scripts/explore_dataset.py`

**这个脚本的目标：**
1. 列出 OmniDocBench 中有多少个 PDF 文件
2. 打印其中一个 PDF 的基本信息（文件大小、页数等）
3. 检查是否存在 OmniDocBench.json（包含 QA 对）
4. 打印 JSON 中的 QA 对样例

**你需要实现：**
```
函数：list_pdf_files()
  - 扫描 data/raw/ 文件夹
  - 返回所有 PDF 文件列表
  - 打印文件数量

函数：inspect_pdf(pdf_path)
  - 打印文件大小
  - 打印页数（可用 PyPDF2 或 pdf2image）
  - 打印第一页的图片大小

函数：inspect_omnidoc_json(json_path)
  - 读取 OmniDocBench.json
  - 打印 QA 对数量
  - 打印 1-2 个样例
```

**完成标准：**
- [ ] 能正确扫描并列出 PDF 文件
- [ ] 能打印 PDF 的基本信息
- [ ] 能读取并展示 OmniDocBench.json 的内容
- [ ] 没有报错，输出清晰易读

---

## 第二阶段：OCR 处理器实现

### 目标
从扫描 PDF 提取文本，理解 OCR 的输出格式和置信度过滤

### 步骤 2.1：单个 PDF 的 OCR 识别
**你需要创建：** `src/ocr_processor.py`

**这个模块的目标：**
1. 将 PDF 转换为图像列表
2. 用 PaddleOCR 识别每张图像
3. 过滤低置信度的识别结果
4. 返回结构化的文本

**你需要实现：**
```
类：OCRProcessor

方法：__init__(confidence_threshold=0.8)
  - 初始化 PaddleOCR
  - 保存置信度阈值

方法：pdf_to_images(pdf_path)
  - 输入：PDF 文件路径
  - 输出：图像列表（List[PIL.Image]）
  - 使用 pdf2image 或 PyPDF2

方法：extract_text_from_image(image, page_num)
  - 输入：单张图像、页码
  - 输出：该页面的文本 + 置信度信息
  - 过滤置信度 < threshold 的识别结果
  - 按 Y 坐标排序（从上到下）

方法：process_pdf(pdf_path)
  - 输入：PDF 文件路径
  - 输出：Dict，包含：
    {
      "文件名": xxx,
      "页数": xxx,
      "pages": [
        {
          "page_num": 0,
          "text": "...",
          "confidence": 0.92
        },
        ...
      ]
    }
```

**开发提示：**
- PaddleOCR 的输出格式：`[[[x1,y1],[x2,y2],...], "文本", 置信度], ...]`
- 按 Y 坐标排序是为了保持读取顺序
- 置信度过滤有助于去除垃圾识别结果

**完成标准：**
- [ ] 能成功读取 1-2 个 PDF
- [ ] 能打印每页的识别文本
- [ ] 置信度过滤正常工作（能看到数值在 0.8-1.0 之间）
- [ ] 没有内存溢出问题

---

### 步骤 2.2：批量处理多个 PDF
**你需要创建：** `scripts/run_ocr_batch.py`

**这个脚本的目标：**
1. 遍历 data/raw/ 中的所有 PDF
2. 调用 OCRProcessor 处理每个 PDF
3. 保存识别结果为 JSON 文件

**你需要实现：**
```
函数：process_all_pdfs(input_dir, output_dir)
  - 遍历 input_dir 中的所有 PDF
  - 对每个 PDF 调用 OCRProcessor.process_pdf()
  - 保存结果到 output_dir/{pdf_name}_ocr.json
  - 打印进度条（可用 tqdm）
  - 如果某个 PDF 失败，记录错误但继续处理
```

**完成标准：**
- [ ] 能处理 5-10 个 PDF（测试用）
- [ ] 生成的 JSON 文件格式正确
- [ ] 输出目录中有对应数量的文件
- [ ] 进度打印清晰

---

## 第三阶段：文本切片与 Chunk 管理

### 目标
将长文本分割成合适大小的 chunks，并为每个 chunk 生成唯一的 ID

### 步骤 3.1：Chunk 生成
**你需要创建：** `src/chunk_manager.py`

**这个模块的目标：**
1. 使用 LangChain 的文本分割器切割文本
2. 为每个 chunk 生成可重现的 ID
3. 保留 metadata（文件名、页码、chunk 索引）

**你需要实现：**
```
类：ChunkManager

方法：__init__(chunk_size=512, overlap=128)
  - 初始化 LangChain 的 RecursiveCharacterTextSplitter
  - 保存参数

方法：generate_chunk_id(file_name, page_num, chunk_index)
  - 输入：文件名、页码、chunk 索引
  - 输出：唯一的 chunk_id（字符串）
  - 规则：hash(file_name + str(page_num) + str(chunk_index))
  - 返回形式：f"{file_name}_p{page_num}_c{chunk_index}"

方法：split_text(file_name, text, page_num=0)
  - 输入：文件名、文本内容、页码
  - 输出：List[Dict]，每个 Dict 包含：
    {
      "chunk_id": "xxx",
      "text": "...",
      "file_name": "xxx",
      "page_num": 0,
      "chunk_index": 0,
      "char_count": 256
    }
  - 用 LangChain 进行分割
  - 为每个 chunk 生成 metadata
```

**开发提示：**
- chunk_size=512 是为了保证语义完整性
- overlap=128 是为了避免在分割线处丢失上下文
- chunk_id 必须可重现（同一个文本产生的 ID 必须相同）

**完成标准：**
- [ ] 能成功分割文本
- [ ] chunk_id 格式正确且可重现
- [ ] metadata 完整
- [ ] 处理中文文本无乱码

---

### 步骤 3.2：批量生成 All Chunks
**你需要创建：** `scripts/generate_all_chunks.py`

**这个脚本的目标：**
1. 读取 OCR 的 JSON 结果
2. 对每个 PDF 的每一页进行 chunking
3. 生成 all_chunks.json（所有 chunks 的列表）

**你需要实现：**
```
函数：generate_all_chunks_from_ocr(ocr_output_dir, output_file)
  - 遍历 ocr_output_dir 中的所有 _ocr.json 文件
  - 对每个文件的每一页调用 ChunkManager.split_text()
  - 将所有 chunks 合并到一个列表
  - 保存为 data/answers/all_chunks.json
  - 打印统计信息：总 chunk 数、平均 chunk 大小等

函数：verify_chunks(chunks_list)
  - 检查是否有重复的 chunk_id
  - 检查 chunk_id 的格式是否正确
  - 打印警告（如果有问题）
```

**完成标准：**
- [ ] 生成的 all_chunks.json 包含 ≥500 个 chunks
- [ ] 所有 chunk_id 唯一且格式正确
- [ ] 能打印统计信息（总数、大小范围等）
- [ ] 没有重复或格式错误

---

## 第四阶段：向量存储（ChromaDB）

### 目标
将 chunks 转换为向量并存储到 ChromaDB，支持后续检索

### 步骤 4.1：向量化与存储
**你需要创建：** `src/vector_store.py`

**这个模块的目标：**
1. 调用 Embedding 模型将文本转换为向量
2. 将向量存储到 ChromaDB
3. 支持后续的相似度检索

**你需要实现：**
```
类：VectorStore

方法：__init__(persist_dir="vectors/chroma_db", embedding_model="qwen")
  - 初始化 ChromaDB 客户端
  - 初始化 Embedding 模型（可用 Qwen API 或本地模型）

方法：embed_text(text)
  - 输入：文本
  - 输出：向量（List[float]）
  - 调用 Embedding 模型

方法：add_chunks(chunks_list)
  - 输入：chunks 列表
  - 对每个 chunk：
    1. 提取文本
    2. 调用 embed_text() 获取向量
    3. 将 chunk_id、向量、metadata 存储到 ChromaDB
  - 打印进度条

方法：search(query_text, k=5)
  - 输入：查询文本、返回的 top-k 数量
  - 输出：List[Dict]，包含：
    {
      "chunk_id": "xxx",
      "text": "...",
      "distance": 0.15,  # 距离（越小越相似）
      "file_name": "xxx",
      "page_num": 0
    }
  - 返回相似度最高的 k 个 chunks
```

**开发提示：**
- 如果 Embedding API 有速率限制，可以加 sleep() 或批量调用
- ChromaDB 会自动处理向量的存储和索引
- 距离计算用余弦相似度

**完成标准：**
- [ ] 能成功将 chunks 添加到 ChromaDB
- [ ] ChromaDB 数据库文件生成成功
- [ ] 能进行简单查询并返回结果
- [ ] 返回的 chunks 数量正确

---

### 步骤 4.2：验证 ChromaDB 同步
**你需要创建：** `src/validators.py`

**这个模块的目标：**
1. 检查 all_chunks.json 和 ChromaDB 是否一致
2. 发现数据对齐问题

**你需要实现：**
```
函数：verify_chromadb_sync(chunks_file, vector_store)
  - 读取 all_chunks.json
  - 遍历每个 chunk_id
  - 从 ChromaDB 查询是否存在
  - 如果不存在，记录为"缺失"
  - 如果 ChromaDB 中有 chunks.json 中没有的，记录为"冗余"
  - 打印报告

函数：verify_chunk_id_consistency(chunks_file)
  - 检查是否有重复的 chunk_id
  - 检查 chunk_id 格式是否正确
  - 打印报告
```

**完成标准：**
- [ ] 能读取 all_chunks.json 和 ChromaDB
- [ ] 验证报告清晰（缺失数、冗余数、总数）
- [ ] 如果数据同步，显示"✓ 数据同步完成"

---

## 第五阶段：金标准构建

### 目标
从 OmniDocBench.json 中提取 QA 对，找到对应的 chunks，生成评估用的金标准

### 步骤 5.1：解析 OmniDocBench.json
**你需要创建：** `scripts/build_gold_answers.py`

**这个脚本的目标：**
1. 读取 OmniDocBench.json
2. 理解其数据结构
3. 提取 QA 对和对应的文档

**你需要实现：**
```
函数：load_omnidoc_json(json_path)
  - 输入：OmniDocBench.json 路径
  - 输出：解析后的数据（Dict 或 List）
  - 打印数据结构样例

函数：analyze_omnidoc_structure(data)
  - 分析数据的键值结构
  - 打印：
    - 总 QA 对数
    - 涉及的文档数
    - 一个样例（问题、答案、所属文档）
```

**完成标准：**
- [ ] 能成功读取 JSON
- [ ] 能清晰打印数据结构
- [ ] 理解 QA 对与文档的关系

---

### 步骤 5.2：匹配 Chunks 与答案
**在 `scripts/build_gold_answers.py` 中继续实现：**

**目标：**
1. 对每个 QA 对，在 all_chunks.json 中找到对应的 chunks
2. 生成 gold_answers.csv

**你需要实现：**
```
函数：find_chunks_for_answer(answer_text, chunks_list, similarity_threshold=0.8)
  - 输入：答案文本、chunks 列表、相似度阈值
  - 输出：匹配的 chunk_id 列表
  - 方法：
    1. 简单方法：用 Python 的 difflib 计算文本相似度
    2. 或调用 Embedding 计算向量相似度
  - 返回相似度 > threshold 的 chunks

函数：build_gold_answers(omnidoc_json, chunks_file, output_csv)
  - 遍历 OmniDocBench.json 中的所有 QA 对
  - 对每个答案调用 find_chunks_for_answer()
  - 生成 gold_answers.csv，包含列：
    - question
    - gold_answer
    - gold_chunk_ids（逗号分隔）
    - num_gold_chunks
    - doc_file_name
  - 打印统计：有效率（能找到 chunks 的 QA 对数 / 总数）
```

**开发提示：**
- 文本相似度可以用 `difflib.SequenceMatcher()`（简单快速）
- 或用向量相似度（更精确但更慢）
- 相似度阈值需要调试（推荐 0.8 起）

**完成标准：**
- [ ] 生成的 gold_answers.csv 包含所有 QA 对
- [ ] 每个答案都有对应的 chunk_ids
- [ ] 有效率 ≥80%（大部分答案找到了 chunks）
- [ ] CSV 格式正确，能用 pandas 读取

---

## 第六阶段：评估框架

### 目标
实现 Hit@K 指标计算，评估检索和生成质量

### 步骤 6.1：Hit@K 评估
**你需要创建：** `src/evaluator.py`

**这个模块的目标：**
1. 对给定问题进行检索
2. 计算 Hit@K 指标
3. 评估检索质量

**你需要实现：**
```
类：RAGEvaluator

方法：__init__(vector_store, chunks_file)
  - 初始化向量存储
  - 加载 all_chunks.json

方法：calculate_hit_at_k(query, gold_chunk_ids, k=5)
  - 输入：查询、正确的 chunk_id 集合、k 值
  - 输出：
    {
      "k": 5,
      "retrieved_chunk_ids": [...],  # 检索到的 chunk_id
      "gold_chunk_ids": [...],       # 正确的 chunk_id
      "hit_count": 2,                # 重叠数
      "hit_rate": 0.4,               # hit_count / len(gold_chunk_ids)
      "hit": True/False              # 是否检索到至少 1 个正确 chunk
    }
  - 逻辑：
    1. 调用 vector_store.search(query, k)
    2. 提取返回的 chunk_ids
    3. 计算与 gold_chunk_ids 的交集
    4. 计算 hit_rate = 交集数 / gold_chunk_ids 数

方法：evaluate_batch(gold_answers_csv, k_list=[1,3,5])
  - 输入：gold_answers.csv、k 值列表
  - 对每个 QA 对调用 calculate_hit_at_k()
  - 生成评估报告，包含：
    - 按 k 值统计的 Hit 数量和百分比
    - 平均 hit_rate
  - 打印表格形式的结果
```

**完成标准：**
- [ ] 能计算单个查询的 Hit@K
- [ ] 能批量评估多个 QA 对
- [ ] 输出的统计信息清晰（表格形式）
- [ ] Hit@1、Hit@3、Hit@5 都能正确计算

---

## 第七阶段：端到端流程验证

### 目标
整合所有模块，完整跑通一次 RAG 系统

### 步骤 7.1：完整流程脚本
**你需要创建：** `scripts/run_full_pipeline.py`

**这个脚本的目标：**
1. 依次调用前面的所有模块
2. 清晰打印每个阶段的进度
3. 生成最终的评估报告

**你需要实现：**
```
函数：run_full_pipeline(config)
  # 阶段 1：OCR
  print("阶段 1：OCR 处理...")
  run_ocr_batch()
  
  # 阶段 2：Chunking
  print("阶段 2：文本切片...")
  generate_all_chunks_from_ocr()
  
  # 阶段 3：向量化
  print("阶段 3：向量化与存储...")
  vector_store = VectorStore()
  chunks = load_all_chunks()
  vector_store.add_chunks(chunks)
  
  # 阶段 4：验证
  print("阶段 4：数据对齐检查...")
  verify_chromadb_sync()
  
  # 阶段 5：金标准
  print("阶段 5：构建金标准...")
  build_gold_answers()
  
  # 阶段 6：评估
  print("阶段 6：RAG 评估...")
  evaluator = RAGEvaluator()
  evaluator.evaluate_batch()
  
  print("✓ 流程完成！")
```

**完成标准：**
- [ ] 脚本能从头到尾运行成功
- [ ] 每个阶段都有清晰的进度提示
- [ ] 没有中间错误

---

### 步骤 7.2：验证 MVP 完成标准
**你需要检查：**

- [ ] **PDF 处理**：能处理 ≥100 个 PDF
- [ ] **Chunks 数量**：all_chunks.json 包含 ≥500 chunks
- [ ] **金标准有效率**：gold_answers.csv 的有效率 ≥80%
- [ ] **Hit@1 > 0**：平均 Hit@1 ≥0.1（说明检索有效）
- [ ] **数据一致**：verify_chromadb_sync() 显示无缺失

---

## 快速参考：模块依赖关系

```
1. explore_dataset.py (独立)
   ↓
2. ocr_processor.py → run_ocr_batch.py
   ↓
3. chunk_manager.py → generate_all_chunks.py
   ↓
4. vector_store.py (需要 all_chunks.json)
   ↓
5. validators.py (检查 4 的结果)
   ↓
6. build_gold_answers.py (需要 all_chunks.json)
   ↓
7. evaluator.py → run_full_pipeline.py (整合 1-6)
```

---

## 常见问题速查

### 如果 Hit@K = 0
**排查步骤：**
1. 运行 verify_chromadb_sync() → 检查数据是否同步
2. 运行 verify_chunk_id_consistency() → 检查 chunk_id 是否重复
3. 手动查询一个答案 → 检查 ChromaDB 中是否有相关内容
4. 检查相似度阈值 → 可能太高了

### 如果生成的 CSV 为空
1. 检查 OmniDocBench.json 是否正确加载
2. 打印样本，看数据结构是否理解正确
3. 检查文件路径是否正确

### 如果速度很慢
1. Embedding API 速率限制 → 添加 batch 处理
2. OCR 处理慢 → 使用 GPU 加速 PaddleOCR
3. ChromaDB 查询慢 → 数据量太大，尝试分片

---

## 开发建议

1. **逐步开发** - 不要一次性写完所有代码，按章节推进
2. **频繁测试** - 每个函数写完就测试，不要等到最后
3. **打印调试** - 用 print() 和日志理解数据流
4. **保存中间结果** - 保存 OCR 结果、chunks、gold_answers，便于调试
5. **记录问题** - 遇到 bug 时记录，最后总结

---

**开始第一步：阅读本指南第一阶段，准备开发环境！**
