# AI 引导用户编程的 RAG 项目开发指南

> **核心原则**：
> 1. 用户手写所有代码，AI 只提示"应该如何做"
> 2. 每个模块完成后，AI 需要详细验证，只有功能完善和无隐性 bug 才能进入下一步
> 3. 采用"学习式"教学，让用户理解每一行代码的含义

---

## AI 的职责

### 代码编写
- ❌ 不能直接修改用户代码
- ❌ 不能完整给出代码模板
- ✅ 提示用户"需要用什么库"、"逻辑应该是什么"
- ✅ 给出伪代码或逻辑流程图
- ✅ 指正错误的实现思路

### 代码验证
- ❌ 不能无条件通过用户代码
- ✅ 运行用户代码，检查输出
- ✅ 提出问题："这个输出看起来对吗？"
- ✅ 让用户自己发现并修复 bug

### 模块完成标准
- AI 必须确认所有"完成标准"都满足才能进入下一步
- 如果有任何不确定，要求用户补充测试

---

## 第一阶段：环境准备 & 数据探索

### 🎯 目标
- 搭建开发环境(已完成)
- 理解 OmniDocBench 数据集结构
- 确认 PDF 的实际特征

### 步骤 1.1：环境配置

**AI 需要做：**
1. 询问用户是否已准备项目文件夹
2. 列出所需的依赖库
3. 引导用户创建 `requirements.txt`

**关键库：**
```
paddlepaddle>=3.0.0
paddleocr>=2.7.0
chromadb>=0.4.0
langchain>=0.1.0
vllm>=0.3.0
pandas>=2.0.0
numpy>=1.24.0
pdf2image>=1.16.0
PyPDF2>=3.0.0
```

**检查清单：**
- [ ] 虚拟环境已创建
- [ ] requirements.txt 已生成
- [ ] 依赖已安装
- [ ] `import paddleocr` 无报错
- [ ] 项目文件夹结构已创建（data/, src/, scripts/, vectors/, results/)

**AI 的对话示例：**
```
你好！我们开始第一个步骤 - 环境配置。
首先，请问：
1. 你已经准备好一个空的项目文件夹了吗？
2. 你已经创建虚拟环境了吗？（可选但推荐）

完成后，我会帮你列出需要安装的库。
```

---

### 步骤 1.2：数据探索脚本

**AI 需要做：**

**第一轮：确认需求**
```
现在我们来写第一个脚本：explore_dataset.py
这个脚本的目标是理解数据集的结构。

你需要创建：scripts/explore_dataset.py

这个脚本需要做三件事：
1. 列出 data/raw/ 中有多少个 PDF 文件
2. 检查 PDF 的基本信息（文件大小、页数等）
3. 读取 OmniDocBench.json 并打印其中的 QA 对

你打算用什么库来读取 PDF 的页数？
选项 A：PyPDF2（更简单）
选项 B：pdf2image（可以生成图像，但更重）
```

**第二轮：逐步引导**

如果用户选择 PyPDF2：
```
好的，用 PyPDF2 是个不错的选择。

现在，你需要写 3 个函数：

1. list_pdf_files(input_dir)
   - 功能：扫描 input_dir 中的所有 PDF 文件
   - 返回值：PDF 文件路径的列表
   - 提示：用 os.listdir() 和 os.path.join()
   
2. inspect_pdf(pdf_path)
   - 功能：打印单个 PDF 的信息
   - 返回值：Dict，包含 {"file_size": xxx, "num_pages": xxx, ...}
   - 提示：PyPDF2.PdfReader() 可以读取 PDF
   
3. inspect_omnidoc_json(json_path)
   - 功能：读取并打印 OmniDocBench.json
   - 返回值：None，直接打印
   - 提示：json.load() 可以读取 JSON 文件

你想先从哪一个函数开始？
```

**第三轮：代码检查**

用户提交代码后：
```
不错！我来运行一下你的代码...

[运行代码]

我看到你列出了 X 个 PDF 文件。
第一个 PDF 的信息是：
- 文件大小：X MB
- 页数：X

这看起来对吗？如果对的话，我们检查一下 OmniDocBench.json 的部分...

[继续验证其他函数]
```

**完成标准检查表：**
```
AI 需要确认以下所有项都满足：
- [ ] list_pdf_files() 返回的列表中有多个 PDF
- [ ] inspect_pdf() 能正确读取页数（与文件关联）
- [ ] inspect_omnidoc_json() 能读取 JSON 并打印结构
- [ ] 没有 FileNotFoundError 或其他异常
- [ ] 输出格式清晰易读
- [ ] 用户理解了这三个函数在做什么

不通过：如果有任何函数无法正常工作，回到"第二轮"的相应函数重新引导。
通过：进入第二阶段。
```

---

## 第二阶段：OCR 处理器实现

### 🎯 目标
从扫描 PDF 提取文本，理解 PaddleOCR 的输出格式和置信度过滤

### 步骤 2.1：单个 PDF 的 OCR 识别

**AI 需要做：**

**第一轮：需求确认**
```
现在我们实现 src/ocr_processor.py，这是项目的第一个核心模块。

OCRProcessor 类需要做这些事：
1. 将 PDF 转换为图像列表
2. 用 PaddleOCR 识别每张图像
3. 过滤低置信度的识别结果
4. 返回结构化的文本

我们分两步走：
- 第一步：实现 pdf_to_images() 方法
- 第二步：实现 extract_text_from_image() 方法

从第一步开始，你打算怎么实现？
```

**第二轮：详细引导**

```
好的，让我们从 pdf_to_images() 开始。

这个方法需要：
- 输入：PDF 文件路径
- 输出：PIL.Image 对象的列表

关键步骤：
1. 用 PyPDF2 或 pdf2image 读取 PDF
2. 将每一页转换为图像（numpy array 或 PIL.Image）
3. 返回图像列表

你可以用 pdf2image.convert_from_path() 一行代码搞定。

尝试实现一下，然后测试：
- 输入一个 PDF 文件
- 看看返回的列表中有多少张图像
```

**第三轮：代码检查**

```
我运行一下你的代码...

你的 pdf_to_images() 返回了 X 张图像。
第一张图像的尺寸是：X x X 像素

这看起来对吗？现在让我们实现下一个方法。
```

**第四轮：extract_text_from_image() 实现**

```
现在实现 extract_text_from_image() 方法。

这个方法是核心，需要：
- 输入：PIL.Image、页码
- 输出：Dict，包含 {text, confidence, raw_results}

关键步骤：
1. 初始化 PaddleOCR：ocr = PaddleOCR(use_gpu=True)
2. 识别：results = ocr.ocr(image)
3. 过滤置信度：只保留 confidence > threshold 的结果
4. 排序：按 Y 坐标从上到下排序
5. 合并：将所有文本连接成一个字符串

PaddleOCR 的输出格式是：
[[[x1,y1],[x2,y2],...], "文本", 置信度], ...]

你需要理解这个格式，然后提取出你需要的信息。

尝试实现一下。
```

**第五轮：整合和测试**

```
现在你有了两个方法。让我们把它们整合到 process_pdf() 中。

process_pdf() 的逻辑：
1. 调用 pdf_to_images()
2. 对每张图像调用 extract_text_from_image()
3. 返回包含所有页面信息的 Dict

实现完后，测试：
- 输入一个真实的 PDF 文件
- 打印前两页的识别结果
- 检查置信度是否都 > 0.8
```

**完成标准检查表：**
```
AI 需要运行用户的代码并验证：
- [ ] pdf_to_images() 返回的图像列表数量正确
- [ ] extract_text_from_image() 返回的文本不为空
- [ ] 置信度过滤有效（没有低于 0.8 的结果）
- [ ] 文本按从上到下的顺序排列
- [ ] process_pdf() 返回的 Dict 结构正确
- [ ] 处理 2-3 个 PDF 都能成功
- [ ] 没有内存溢出或超时

不通过：重新检查哪个环节有问题
通过：进入步骤 2.2
```

---

### 步骤 2.2：批量处理多个 PDF

**AI 需要做：**

**第一轮：需求确认**
```
现在我们写 scripts/run_ocr_batch.py，用来批量处理所有 PDF。

这个脚本需要：
1. 遍历 data/raw/ 中的所有 PDF
2. 对每个 PDF 调用 OCRProcessor.process_pdf()
3. 保存结果为 JSON 文件
4. 记录错误但继续处理

你需要写一个主函数 process_all_pdfs()。

有什么不清楚的吗？
```

**第二轮：逐步引导**

```
process_all_pdfs() 的伪代码：

def process_all_pdfs(input_dir, output_dir):
    ocr = OCRProcessor()  # 初始化
    
    for pdf_file in os.listdir(input_dir):
        if not pdf_file.endswith('.pdf'):
            continue
        
        try:
            result = ocr.process_pdf(pdf_file_path)
            # 保存 result 到 JSON 文件
        except Exception as e:
            # 记录错误
            pass

你需要：
1. 遍历 input_dir 中的 PDF
2. 对每个 PDF 调用 process_pdf()
3. 用 json.dump() 保存结果
4. 用 tqdm 显示进度条

实现一下。
```

**第三轮：代码检查**

```
我运行一下你的代码...

处理了 X 个 PDF，成功 X 个，失败 X 个。
生成的文件：
- pdf_name_1_ocr.json
- pdf_name_2_ocr.json
- ...

我打开其中一个 JSON 文件，结构是这样的：
{
  "文件名": "xxx",
  "页数": 5,
  "pages": [
    {"page_num": 0, "text": "...", "confidence": 0.92},
    ...
  ]
}

这看起来对吗？
```

**完成标准检查表：**
```
AI 需要验证：
- [ ] 能处理 5-10 个 PDF（测试用）
- [ ] 生成的 JSON 文件数量正确
- [ ] JSON 结构符合预期
- [ ] 有进度条或进度提示
- [ ] 错误被捕获和记录
- [ ] 没有中途中断

通过：进入第三阶段
```

---

## 第三阶段：文本切片与 Chunk 管理

### 🎯 目标
将长文本分割成合适大小的 chunks，为每个 chunk 生成唯一 ID

### 步骤 3.1：Chunk 生成

**AI 需要做：**

**第一轮：需求确认和 chunk_id 讲解**
```
现在实现 src/chunk_manager.py，这是第二个核心模块。

首先，理解"chunk_id"的概念很重要。

chunk_id 是什么？
- 每个 chunk 的唯一标识符
- 由 3 部分组成：文件名 + 页码 + chunk 索引
- 例如：docs_p0_c0、docs_p0_c1、docs_p1_c0
- 重要：chunk_id 必须可重现（同一个文本产生的 ID 必须相同）

为什么需要 chunk_id？
- 在 ChromaDB 中标识每个 chunk
- 在 gold_answers.csv 中记录正确的 chunks
- 在评估时匹配检索到的 chunks

现在，你需要实现三个方法：
1. __init__() - 初始化分割器
2. generate_chunk_id() - 生成 chunk_id
3. split_text() - 分割文本

从 generate_chunk_id() 开始怎么样？
```

**第二轮：generate_chunk_id() 实现**

```
generate_chunk_id() 需要：
- 输入：file_name、page_num、chunk_index
- 输出：chunk_id（字符串）

例子：
  generate_chunk_id("docs.pdf", 0, 0) → "docs_p0_c0"
  generate_chunk_id("docs.pdf", 0, 1) → "docs_p0_c1"
  generate_chunk_id("docs.pdf", 1, 0) → "docs_p1_c0"

规则很简单，就是拼接三个部分。

实现一下。
```

**第三轮：split_text() 实现**

```
现在实现 split_text() 方法。

这个方法使用 LangChain 的分割器。关键步骤：

1. 从 langchain.text_splitter 导入 RecursiveCharacterTextSplitter
2. 初始化分割器：
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=512,
       chunk_overlap=128,
       separators=["\\n\\n", "\\n", "。", "！", ""]
   )
3. 分割文本：chunks_text = splitter.split_text(text)
4. 对每个 chunk，调用 generate_chunk_id() 生成 ID
5. 为每个 chunk 添加 metadata

split_text() 的返回值应该是 List[Dict]：
[
  {
    "chunk_id": "docs_p0_c0",
    "text": "...",
    "file_name": "docs.pdf",
    "page_num": 0,
    "chunk_index": 0,
    "char_count": 512
  },
  ...
]

实现一下。
```

**第四轮：代码检查**

```
我测试一下你的代码...

用一个真实的文本测试 split_text()：

输入文本长度：5000 个字符
生成的 chunks 数：X
chunk_id 样例：docs_p0_c0、docs_p0_c1、...

每个 chunk 的字符数在 256-512 之间吗？
```

**完成标准检查表：**
```
AI 需要验证：
- [ ] generate_chunk_id() 生成的 ID 格式正确
- [ ] split_text() 能成功分割文本
- [ ] 返回的 chunk 数量合理
- [ ] chunk_id 可重现（同一文本多次调用得到相同 ID）
- [ ] metadata 完整且格式正确
- [ ] 处理中文文本无乱码

通过：进入步骤 3.2
```

---

### 步骤 3.2：批量生成 All Chunks

**AI 需要做：**

**第一轮：需求和逻辑讲解**
```
现在写 scripts/generate_all_chunks.py。

这个脚本的任务是：
1. 读取前面 OCR 生成的所有 JSON 文件
2. 对每个 JSON 文件的每一页调用 ChunkManager.split_text()
3. 将所有 chunks 合并到一个文件：data/answers/all_chunks.json

为什么需要 all_chunks.json？
- 这是一个"金标准文档库"
- 后面的向量化、评估都依赖它
- 必须确保 all_chunks.json 的一致性

你需要实现：
1. generate_all_chunks_from_ocr() - 主函数
2. verify_chunks() - 验证函数

先实现 generate_all_chunks_from_ocr()。
```

**第二轮：逐步引导**

```
generate_all_chunks_from_ocr() 的逻辑：

def generate_all_chunks_from_ocr(ocr_output_dir, output_file):
    cm = ChunkManager()  # 初始化
    all_chunks = []
    
    for ocr_json_file in os.listdir(ocr_output_dir):
        if not ocr_json_file.endswith('_ocr.json'):
            continue
        
        # 读取 JSON
        ocr_data = json.load(...)
        file_name = ocr_data['文件名']
        
        # 对每一页
        for page_info in ocr_data['pages']:
            text = page_info['text']
            page_num = page_info['page_num']
            
            # 分割成 chunks
            chunks = cm.split_text(file_name, text, page_num)
            all_chunks.extend(chunks)
    
    # 保存
    with open(output_file, 'w') as f:
        json.dump(all_chunks, f)
    
    # 验证
    verify_chunks(all_chunks)

实现一下。
```

**第三轮：verify_chunks() 实现**

```
verify_chunks() 需要检查：
1. 是否有重复的 chunk_id？
2. chunk_id 的格式是否正确？

实现：
def verify_chunks(chunks_list):
    chunk_ids = [c['chunk_id'] for c in chunks_list]
    
    # 检查重复
    duplicates = len(chunk_ids) - len(set(chunk_ids))
    if duplicates > 0:
        print(f"⚠️ 发现 {duplicates} 个重复的 chunk_id")
    
    # 检查格式
    for cid in chunk_ids:
        if not re.match(r'^.*_p\d+_c\d+$', cid):
            print(f"❌ 错误的 chunk_id 格式：{cid}")
    
    print(f"✓ 总共 {len(chunks_list)} 个 chunks")

实现一下。
```

**第四轮：代码检查**

```
我运行一下你的代码...

处理了 X 个 OCR 结果文件。
生成了 X 个 chunks。

chunk_id 样例：
- docs_p0_c0
- docs_p0_c1
- docs_p1_c0
- ...

验证结果：
- 重复 chunk_id 数：0
- 格式错误数：0
- ✓ 数据有效

这看起来对吗？
```

**完成标准检查表：**
```
AI 需要验证：
- [ ] 生成的 all_chunks.json 包含 ≥500 chunks
- [ ] 所有 chunk_id 唯一
- [ ] chunk_id 格式正确
- [ ] 没有报告重复或格式错误
- [ ] 能用 pandas 读取 JSON

通过：进入第四阶段
```

---

## 第四阶段：向量存储（ChromaDB）

### 🎯 目标
将 chunks 转换为向量并存储到 ChromaDB

### 步骤 4.1：向量化与存储

**AI 需要做：**

**第一轮：需求确认**
```
现在实现 src/vector_store.py。

这个模块的任务：
1. 调用 Embedding 模型将文本转换为向量
2. 将向量存储到 ChromaDB
3. 支持后续的相似度检索

需要实现 4 个关键方法：
1. __init__() - 初始化 ChromaDB 和 Embedding 模型
2. embed_text() - 文本转向量
3. add_chunks() - 添加 chunks 到 ChromaDB
4. search() - 查询相似的 chunks

从 __init__() 开始。
```

**第二轮：__init__() 实现**

```
__init__() 需要：
1. 初始化 ChromaDB 客户端
2. 初始化 Embedding 模型

关键代码：
import chromadb
self.client = chromadb.PersistentClient(path=persist_dir)
self.collection = self.client.get_or_create_collection(
    name="chunks",
    metadata={"hnsw:space": "cosine"}
)

对于 Embedding 模型，你有两个选择：
A. 使用 Hugging Face 的本地模型（需要下载，几 GB）
B. 使用 API 服务（如 OpenAI、阿里云等）

哪一个对你更方便？
```

**第三轮：embed_text() 实现**

```
embed_text() 需要：
- 输入：文本
- 输出：向量（List[float]）

如果用本地模型：
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
embeddings = model.encode([text])

如果用 API：
import requests
response = requests.post(api_url, json={"text": text})
embeddings = response.json()['embedding']

实现一下。
```

**第四轮：add_chunks() 实现**

```
add_chunks() 是核心方法。逻辑：

def add_chunks(self, chunks_list):
    for chunk in chunks_list:
        # 提取信息
        chunk_id = chunk['chunk_id']
        text = chunk['text']
        metadata = {
            'file_name': chunk['file_name'],
            'page_num': chunk['page_num'],
            'chunk_index': chunk['chunk_index'],
        }
        
        # 获取向量
        embedding = self.embed_text(text)
        
        # 添加到 ChromaDB
        self.collection.add(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

注意：
- 如果有很多 chunks，Embedding 可能很慢
- 可以添加 batch 处理或 sleep() 来避免 API 限流

实现一下。
```

**第五轮：search() 实现**

```
search() 需要：
- 输入：查询文本、k 值
- 输出：List[Dict]，包含 chunk_id、text、距离等

逻辑：

def search(self, query_text, k=5):
    query_embedding = self.embed_text(query_text)
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # 格式化返回值
    output = []
    for i, chunk_id in enumerate(results['ids'][0]):
        output.append({
            'chunk_id': chunk_id,
            'text': results['documents'][0][i],
            'distance': results['distances'][0][i],
            ...
        })
    
    return output

实现一下。
```

**第六轮：代码检查**

```
我测试一下你的代码...

[运行代码，添加 10-20 个 chunks]

添加成功！ChromaDB 数据库文件已生成在 vectors/chroma_db/。

现在测试 search() 方法...

查询："人工智能"
返回的 top-3 chunks：
1. chunk_id: xxx, distance: 0.12
2. chunk_id: xxx, distance: 0.15
3. chunk_id: xxx, distance: 0.18

距离越小，相关性越高。这些结果看起来对吗？
```

**完成标准检查表：**
```
AI 需要验证：
- [ ] 能成功添加 chunks 到 ChromaDB
- [ ] ChromaDB 数据库文件生成成功
- [ ] search() 能返回结果
- [ ] 返回的 chunks 数量等于 k
- [ ] distance 值在合理范围内（通常 0-1）

通过：进入步骤 4.2
```

---

### 步骤 4.2：验证 ChromaDB 同步

**AI 需要做：**

**第一轮：需求确认**
```
现在实现 src/validators.py。

这个模块很重要！它的作用是检查数据对齐问题。

为什么需要验证？
- 有可能 all_chunks.json 和 ChromaDB 不同步
- 有可能在 Embedding 时丢失了某些 chunks
- 后续 Hit@K=0 的问题，往往源于数据对齐

你需要实现两个验证函数：
1. verify_chromadb_sync() - 检查同步状态
2. verify_chunk_id_consistency() - 检查 chunk_id 重复

先实现 verify_chunk_id_consistency()。
```

**第二轮：verify_chunk_id_consistency() 实现**

```
verify_chunk_id_consistency() 需要：
1. 读取 all_chunks.json
2. 检查是否有重复的 chunk_id
3. 检查 chunk_id 格式是否正确

逻辑：

def verify_chunk_id_consistency(chunks_file):
    chunks = json.load(open(chunks_file))
    chunk_ids = [c['chunk_id'] for c in chunks]
    
    # 检查重复
    duplicates = set([id for id in chunk_ids if chunk_ids.count(id) > 1])
    
    # 检查格式
    bad_format = [id for id in chunk_ids if not re.match(r'^.*_p\d+_c\d+$', id)]
    
    # 打印报告
    if duplicates:
        print(f"❌ 发现 {len(duplicates)} 个重复的 chunk_id：{duplicates}")
    else:
        print(f"✓ 没有重复的 chunk_id")
    
    if bad_format:
        print(f"❌ 发现 {len(bad_format)} 个格式错误：{bad_format[:5]}")
    else:
        print(f"✓ 所有 chunk_id 格式正确")

实现一下。
```

**第三轮：verify_chromadb_sync() 实现**

```
verify_chromadb_sync() 是更复杂的检查。需要：
1. 读取 all_chunks.json
2. 读取 ChromaDB 中的所有 chunks
3. 对比两者

逻辑：

def verify_chromadb_sync(chunks_file, vector_store):
    # 读取 all_chunks
    all_chunks = json.load(open(chunks_file))
    all_chunk_ids = set([c['chunk_id'] for c in all_chunks])
    
    # 读取 ChromaDB
    db_chunks = vector_store.collection.get()
    db_chunk_ids = set(db_chunks['ids'])
    
    # 对比
    missing = all_chunk_ids - db_chunk_ids  # 在 json 但不在 db
    redundant = db_chunk_ids - all_chunk_ids  # 在 db 但不在 json
    
    # 打印报告
    print(f"总共 {len(all_chunk_ids)} 个 chunks")
    if missing:
        print(f"⚠️ 缺失 {len(missing)} 个 chunks")
    if redundant:
        print(f"⚠️ 冗余 {len(redundant)} 个 chunks")
    
    if not missing and not redundant:
        print(f"✓ 数据完全同步！")

实现一下。
```

**第四轮：代码检查**

```
我运行一下验证脚本...

verify_chunk_id_consistency():
✓ 没有重复的 chunk_id
✓ 所有 chunk_id 格式正确

verify_chromadb_sync():
总共 5423 个 chunks
缺失 0 个 chunks
冗余 0 个 chunks
✓ 数据完全同步！

很好！这说明数据对齐没问题。
```

**完成标准检查表：**
```
AI 需要验证：
- [ ] verify_chunk_id_consistency() 能正确检查重复和格式
- [ ] verify_chromadb_sync() 能对比两个数据源
- [ ] 验证报告清晰易读
- [ ] 能正确识别缺失和冗余

通过：进入第五阶段
```

---

## 第五阶段：金标准构建

### 🎯 目标
从 OmniDocBench.json 中提取 QA 对，找到对应的 chunks，生成评估用的金标准

### 步骤 5.1：解析 OmniDocBench.json

**AI 需要做：**

**第一轮：数据探索**
```
现在开始构建金标准。这是 RAG 评估的基础。

首先，你需要深入理解 OmniDocBench.json 的结构。

你还记得在步骤 1.2 中，我们用 explore_dataset.py 读取过这个文件吗？

现在再看一遍，回答这些问题：
1. JSON 中有多少个 QA 对？
2. 一个 QA 对的结构是什么？（包含哪些字段）
3. 问题和答案是分开的字段吗？
4. 是否有字段表示"这个答案来自哪个文档"？

先分析一下数据结构，然后回答这些问题。
```

**第二轮：指导解析**

```
好的，现在你对数据结构有了认识。

接下来，写 scripts/build_gold_answers.py 中的第一个函数：

def load_omnidoc_json(json_path):
    # 读取 JSON 文件
    # 返回解析后的数据
    pass

def analyze_omnidoc_structure(data):
    # 打印数据的统计信息
    # 包括：QA 对数、文档数、一个样例
    pass

实现一下，让我看看数据结构。
```

**第三轮：代码检查**

```
我运行一下你的代码...

[输出]
总 QA 对数：X
涉及的文档数：Y

样例：
问题："...?"
答案："..."
所属文档："xxx.pdf"

这个样例看起来对吗？
```

**完成标准检查表：**
```
AI 需要验证：
- [ ] 能成功读取 JSON
- [ ] analyze_omnidoc_structure() 输出清晰
- [ ] 能看到 QA 对的真实样例
- [ ] 理解了答案与文档的关系

通过：进入步骤 5.2
```

---

### 步骤 5.2：匹配 Chunks 与答案

**AI 需要做：**

**第一轮：策略讨论**
```
现在是核心步骤：匹配答案与 chunks。

这是最难的部分，因为：
- 答案可能很长，跨越多个 chunks
- OCR 可能有错误，文本不完全匹配
- 需要找到"足够相似"的 chunks

有两种匹配策略：

策略 A：文本相似度匹配（简单快速）
  - 用 Python difflib 计算相似度
  - 如果相似度 > 0.8，就认为匹配
  - 缺点：容易受 OCR 错误影响

策略 B：向量相似度匹配（更精确）
  - 用 vector_store.search() 查找相似 chunks
  - 返回距离最小的 chunks
  - 优点：容错性强

你想用哪种策略？
```

**第二轮：实现 find_chunks_for_answer()**

```
现在实现 find_chunks_for_answer()。

逻辑（以文本相似度为例）：

def find_chunks_for_answer(answer_text, chunks_list, similarity_threshold=0.8):
    from difflib import SequenceMatcher
    
    matching_chunks = []
    
    for chunk in chunks_list:
        chunk_text = chunk['text']
        
        # 计算相似度
        ratio = SequenceMatcher(None, answer_text, chunk_text).ratio()
        
        if ratio > similarity_threshold:
            matching_chunks.append({
                'chunk_id': chunk['chunk_id'],
                'similarity': ratio
            })
    
    # 按相似度排序
    matching_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    
    return [c['chunk_id'] for c in matching_chunks]

实现一下。
```

**第三轮：实现 build_gold_answers()**

```
现在实现主函数 build_gold_answers()。

逻辑：

def build_gold_answers(omnidoc_json, chunks_file, output_csv):
    omnidoc_data = load_omnidoc_json(omnidoc_json)
    chunks = json.load(open(chunks_file))
    
    results = []
    
    for qa_pair in omnidoc_data:
        question = qa_pair['question']
        answer = qa_pair['answer']
        
        # 查找相关 chunks
        chunk_ids = find_chunks_for_answer(answer, chunks)
        
        # 添加到结果
        results.append({
            'question': question,
            'gold_answer': answer,
            'gold_chunk_ids': ','.join(chunk_ids),  # 逗号分隔
            'num_gold_chunks': len(chunk_ids),
            'doc_file_name': qa_pair.get('doc_file_name', '')
        })
    
    # 保存为 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # 统计
    valid = (df['num_gold_chunks'] > 0).sum()
    total = len(df)
    print(f"有效率：{valid}/{total} = {valid/total:.1%}")

实现一下。
```

**第四轮：代码检查和调试**

```
我运行一下你的代码...

[运行]

处理了 X 个 QA 对。
其中 Y 个找到了对应的 chunks。
有效率：Y/X = Z%

生成的 gold_answers.csv 样例：
question | gold_answer | gold_chunk_ids | num_gold_chunks
---------|------------|----------------|-----------------
"...?"   | "..."      | "xxx_p0_c0,xxx_p0_c1" | 2
...

这个有效率看起来合理吗？

如果有效率 < 80%，我们需要调整相似度阈值或检查 chunks 的质量。
```

**完成标准检查表：**
```
AI 需要验证：
- [ ] 生成的 gold_answers.csv 包含所有 QA 对
- [ ] 每个答案都有 gold_chunk_ids
- [ ] 有效率 ≥80%（大部分答案找到了 chunks）
- [ ] CSV 格式正确，能用 pandas 读取
- [ ] 没有有重复的 chunk_id

通过：进入第六阶段
```

---

## 第六阶段：评估框架

### 🎯 目标
实现 Hit@K 指标计算，评估检索质量

### 步骤 6.1：Hit@K 评估

**AI 需要做：**

**第一轮：概念讲解**
```
现在实现 src/evaluator.py，这是最后一个核心模块。

首先，理解"Hit@K"这个指标。

Hit@K 的含义：
- K = 检索的 top-k 个 chunks
- Hit = 是否至少有 1 个正确的 chunk
- Hit@1：检索到的第 1 个 chunk 是否正确？
- Hit@5：检索到的前 5 个 chunks 中是否有正确的？

例子：
- 问题："什么是机器学习？"
- 正确的 chunks：["doc_p1_c0", "doc_p1_c1"]
- 检索到的 top-5：["doc_p1_c0", "doc_p2_c3", "doc_p3_c5", "doc_p4_c1", "doc_p5_c2"]
- Hit@5：1（因为有 1 个正确的）
- Hit@1：1（因为第 1 个也是正确的）

还有一个指标叫 Hit Rate：
- Hit Rate = 检索到的正确 chunks 数 / 应该检索到的 chunks 数
- 例如：检索到 1 个，应该检索到 2 个，Hit Rate = 0.5

现在理解了吗？
```

**第二轮：实现 RAGEvaluator 类**

```
实现 src/evaluator.py 中的 RAGEvaluator 类。

关键方法：

1. __init__():
   - 初始化 vector_store
   - 加载 all_chunks.json
   - 建立 chunk_id → chunk 的映射（便于查找）

2. calculate_hit_at_k():
   - 输入：query、gold_chunk_ids、k
   - 输出：
     {
       "k": 5,
       "retrieved_ids": ["doc_p1_c0", ...],  # 检索到的 chunk_id
       "gold_ids": ["doc_p1_c0", "doc_p1_c1"],  # 正确的 chunk_id
       "hit_count": 1,  # 交集数
       "hit_rate": 0.5,  # 1 / 2
       "hit": True  # 是否 ≥ 1 个正确
     }
   
   逻辑：
   - retrieved_ids = vector_store.search(query, k) 的 chunk_ids
   - hit_count = len(set(retrieved_ids) & set(gold_chunk_ids))
   - hit_rate = hit_count / len(gold_chunk_ids)
   - hit = (hit_count > 0)

实现一下。
```

**第三轮：实现 evaluate_batch()**

```
实现 evaluate_batch() 方法，用来批量评估。

逻辑：

def evaluate_batch(self, gold_answers_csv, k_list=[1, 3, 5]):
    df = pd.read_csv(gold_answers_csv)
    
    # 对每个 k 值，记录统计
    results = {k: {'hits': 0, 'total': 0} for k in k_list}
    
    for _, row in df.iterrows():
        question = row['question']
        gold_ids = set(row['gold_chunk_ids'].split(','))
        
        for k in k_list:
            eval_result = self.calculate_hit_at_k(question, gold_ids, k)
            
            if eval_result['hit']:
                results[k]['hits'] += 1
            results[k]['total'] += 1
    
    # 打印结果
    for k in k_list:
        hit_rate = results[k]['hits'] / results[k]['total']
        print(f"Hit@{k}: {results[k]['hits']}/{results[k]['total']} = {hit_rate:.1%}")

实现一下。
```

**第四轮：代码检查**

```
我运行一下你的代码...

[运行评估]

评估结果：
Hit@1: 150/500 = 30.0%
Hit@3: 250/500 = 50.0%
Hit@5: 300/500 = 60.0%

这个结果看起来合理。Hit@1 最低，因为要求最严格；Hit@5 较高，因为有更多机会找到正确 chunks。

如果 Hit@1 = 0%，说明检索有严重问题。我们需要排查：
1. 向量模型是否正确？
2. gold_answers.csv 中的 chunk_ids 是否有效？
3. chunks 的质量是否有问题？

你的结果是什么样的？
```

**完成标准检查表：**
```
AI 需要验证：
- [ ] 能计算单个查询的 Hit@K
- [ ] 能批量评估多个 QA 对
- [ ] Hit@1 ≤ Hit@3 ≤ Hit@5（必须满足）
- [ ] Hit@1 > 0%（说明检索有效）
- [ ] 输出的统计信息清晰

通过：进入第七阶段
```

---

## 第七阶段：端到端流程验证

### 🎯 目标
整合所有模块，完整跑通一次 RAG 系统

### 步骤 7.1：完整流程脚本

**AI 需要做：**

**第一轮：需求确认**
```
现在写最后一个脚本：scripts/run_full_pipeline.py。

这个脚本的作用是：
1. 依次调用所有前面的模块
2. 清晰打印每个阶段的进度
3. 生成最终的评估报告

流程：
阶段 1：OCR 处理
阶段 2：文本切片
阶段 3：向量化存储
阶段 4：数据验证
阶段 5：金标准构建
阶段 6：RAG 评估

实现这个流程。
```

**第二轮：逐步引导**

```
主函数的伪代码：

def run_full_pipeline(config_file):
    config = load_config(config_file)
    
    print("=" * 50)
    print("开始 RAG 系统初始化")
    print("=" * 50)
    
    # 阶段 1
    print("\\n[阶段 1] OCR 处理...")
    process_all_pdfs(...)
    
    # 阶段 2
    print("\\n[阶段 2] 文本切片...")
    generate_all_chunks_from_ocr(...)
    
    # 阶段 3
    print("\\n[阶段 3] 向量化与存储...")
    vector_store = VectorStore(...)
    chunks = json.load(...)
    vector_store.add_chunks(chunks)
    
    # 阶段 4
    print("\\n[阶段 4] 数据验证...")
    verify_chunk_id_consistency(...)
    verify_chromadb_sync(...)
    
    # 阶段 5
    print("\\n[阶段 5] 金标准构建...")
    build_gold_answers(...)
    
    # 阶段 6
    print("\\n[阶段 6] RAG 评估...")
    evaluator = RAGEvaluator(...)
    evaluator.evaluate_batch(...)
    
    print("\\n" + "=" * 50)
    print("✓ RAG 系统初始化完成！")
    print("=" * 50)

实现一下。
```

**第三轮：代码检查**

```
我运行一下你的代码...

[运行]

==================================================
开始 RAG 系统初始化
==================================================

[阶段 1] OCR 处理...
处理了 100 个 PDF，成功 100 个

[阶段 2] 文本切片...
生成了 5423 个 chunks

[阶段 3] 向量化与存储...
添加了 5423 个 chunks 到 ChromaDB

[阶段 4] 数据验证...
✓ 没有重复的 chunk_id
✓ 数据完全同步！

[阶段 5] 金标准构建...
处理了 500 个 QA 对，有效率 85%

[阶段 6] RAG 评估...
Hit@1: 150/500 = 30.0%
Hit@3: 250/500 = 50.0%
Hit@5: 300/500 = 60.0%

==================================================
✓ RAG 系统初始化完成！
==================================================

太棒了！整个流程都通了！
```

**完成标准检查表：**
```
AI 需要验证：
- [ ] 脚本能从头到尾运行成功
- [ ] 每个阶段都有清晰的进度提示
- [ ] 没有中间报错
- [ ] 最终能生成评估结果

通过：验证 MVP 完成标准
```

---

### 步骤 7.2：验证 MVP 完成标准

**AI 需要做：**

**最终检查清单：**
```
✓ MVP 完成标准检查表

【数据处理】
- [ ] 能处理 ≥100 个 PDF 文件
- [ ] all_chunks.json 包含 ≥500 chunks
- [ ] 没有重复的 chunk_id

【向量存储】
- [ ] ChromaDB 成功创建
- [ ] all_chunks 和 ChromaDB 完全同步
- [ ] search() 能返回相关结果

【金标准】
- [ ] gold_answers.csv 包含所有 QA 对
- [ ] 有效率 ≥80%（大部分答案找到了 chunks）

【评估】
- [ ] Hit@1 > 0%（说明检索有效）
- [ ] Hit@1 ≤ Hit@3 ≤ Hit@5

如果所有项都通过，恭喜！MVP 完成！
```

**AI 的总结语：**
```
🎉 恭喜！你已经成功实现了一个完整的 RAG 系统！

这个 MVP 实现了以下功能：
✓ OCR 文本提取：从扫描 PDF 识别文字
✓ 文本切片：将长文本分割成 chunks
✓ 向量化存储：将 chunks 转换为向量并存储
✓ 数据验证：确保数据一致性
✓ 金标准构建：生成评估用的 QA 对
✓ 检索评估：计算 Hit@K 指标

这个系统虽然是 MVP 版本，但已经展示了 RAG 的核心流程。

后续可以考虑的优化方向（不在本指南范围内）：
- 调整 chunk_size 和 overlap
- 尝试不同的 Embedding 模型
- 实现 BM25 多路召回
- 添加 Reranking 模块
- 优化 Prompt 以改进生成质量

祝你继续深造！
```

---


## 模块开发顺序速查表

```
【阶段 1】数据探索
├─ scripts/explore_dataset.py
│  ├─ list_pdf_files()
│  ├─ inspect_pdf()
│  └─ inspect_omnidoc_json()

【阶段 2】OCR 处理
├─ src/ocr_processor.py
│  ├─ __init__()
│  ├─ pdf_to_images()
│  ├─ extract_text_from_image()
│  └─ process_pdf()
├─ scripts/run_ocr_batch.py
│  └─ process_all_pdfs()

【阶段 3】文本切片
├─ src/chunk_manager.py
│  ├─ __init__()
│  ├─ generate_chunk_id()
│  └─ split_text()
├─ scripts/generate_all_chunks.py
│  ├─ generate_all_chunks_from_ocr()
│  └─ verify_chunks()

【阶段 4】向量存储
├─ src/vector_store.py
│  ├─ __init__()
│  ├─ embed_text()
│  ├─ add_chunks()
│  └─ search()
├─ src/validators.py
│  ├─ verify_chunk_id_consistency()
│  └─ verify_chromadb_sync()

【阶段 5】金标准
├─ scripts/build_gold_answers.py
│  ├─ load_omnidoc_json()
│  ├─ analyze_omnidoc_structure()
│  ├─ find_chunks_for_answer()
│  └─ build_gold_answers()

【阶段 6】评估
├─ src/evaluator.py
│  ├─ __init__()
│  ├─ calculate_hit_at_k()
│  └─ evaluate_batch()

【阶段 7】端到端
└─ scripts/run_full_pipeline.py
   └─ run_full_pipeline()
```

---

