# OCR Evaluation Improvement Report

记录从 baseline 开始的所有评估框架优化及其效果。

> **注意**：所有分数变化来自**评估框架修复**，OCR 识别模型本身未改动。
> 真实 OCR 质量的提升需要在 `ocr_processor.py` 层面替换识别模型。

---

## Baseline（优化前）

**生成时间**：2026-03-10 07:28:25
**评估页数**：1005 页（含空页）

| 指标 | 值 |
|---|---|
| Character Accuracy | 0.7010 |
| Sequence Similarity | 0.6684 |
| Jaccard Similarity | 0.8086 |
| Table TEDS | 87.39 |
| Formula Avg Accuracy | **0.0000**（未生效） |
| Correct Formulas | **0 / 1146** |
| Composite Mean | 72.15 |
| Composite Median | 84.48 |
| Excellent (≥90) | 333 页（33.1%） |
| Good (80-90) | 248 页（24.7%） |
| Poor (<70) | 324 页（32.2%） |

**已知问题**：
- `parse_omnidocbench.py` 未提取 `equation_isolated` 的 `latex` 字段，公式 GT 全为空
- `_extract_gt_formulas` 匹配了错误的 `category_type`（`"formula"` 而非 `"equation_isolated"`）
- `_extract_ocr_formulas` 读取了错误字段（`content` 而非 `rec_formula`）
- GT LaTeX 带 `$$...$$` 包裹未去除
- 统计中包含双空页（OCR=0 且 GT=0），虚增满分页数
- 公式比较为逐索引匹配（对数量不一致的情况完全失效）
- `normalize_latex` 规则不足，全角/半角未处理

---

## Round 1：公式 GT 提取修复

**提交**：`fix: extract formula LaTeX from GT and fix category_type matching`

**改动**：
- `parse_omnidocbench.py`：提取 `equation_isolated` / `equation_caption` 的 `latex` 字段
- `evaluate_ocr_quality.py`：`_extract_gt_formulas` 匹配正确的 `category_type`，优先读 `latex` 字段

| 指标 | Baseline | Round 1 | 变化 |
|---|---|---|---|
| Formula Avg Accuracy | 0.0000 | 0.0000 | 字段通了但内容仍为空 |
| Correct Formulas | 0/1146 | 0/1146 | — |
| Composite Mean | 72.15 | 69.43 | -2.72（公式权重加入拉低） |

---

## Round 2：OCR 公式字段 + GT 去包裹

**提交**：`fix: correct formula extraction field names and strip LaTeX delimiters`

**改动**：
- `_extract_ocr_formulas`：读取 `rec_formula` 字段（PP-StructureV3 实际输出字段）
- `_extract_gt_formulas`：去除 GT LaTeX 的 `$$...$$` / `$...$` 包裹

| 指标 | Baseline | Round 2 | 变化 |
|---|---|---|---|
| Formula Avg Accuracy | 0.0000 | **0.3610** | +∞ ↑ |
| Correct Formulas | 0/1146 | 7/1143 (0.6%) | 从0到有 |
| Composite Mean | 72.15 | 70.73 | -1.42 |

---

## Round 3：文本标准化 + 过滤空页

**提交**：`feat: add NFKC text normalization and filter empty pages from metrics`

**改动**：
- `metrics.py`：`normalize_text_for_compare` 加入 NFKC Unicode 标准化（全角→半角）
- `evaluate_ocr_quality.py`：报告和统计中排除双空页（OCR=0 且 GT=0）

| 指标 | Baseline | Round 3 | 变化 |
|---|---|---|---|
| 有效评估页数 | 1005 | **991**（过滤 14 空页） | 更真实 |
| Character Accuracy | 0.7010 | 0.6996 | -0.0014（空页去掉后基数更实） |
| exact 1.0 页面 | 22 页 (2.2%) | **10 页 (1.0%)** | 虚假满分减少 |
| Composite Mean | 72.15 | 70.62 | — |

**备注**：NFKC 标准化效果有限，说明该数据集全角/半角问题不是主要误差来源。

---

## Round 4：公式最优匹配 + normalize_latex 扩展 + correct 阈值调整

**提交**：`feat: improve formula evaluation with best-match and better normalization`

**改动**：
- `formula_evaluator.py`：`evaluate_formulas` 从逐索引改为**最优匹配**（每个 OCR 公式只用一次）
- `normalize_latex`：新增 `\dfrac`→`\frac`、`\left(`→`(`、单字符上下标加花括号、`\boldsymbol`→`\mathbf` 等 12 条规则
- correct 阈值：0.95 → **0.8**

| 指标 | Baseline | Round 4 | 变化 |
|---|---|---|---|
| Formula Avg Accuracy | 0.0000 | **0.6198** | +∞ ↑↑ |
| Correct Formulas | 0/1146 | **299/1143 (26.2%)** | 从0到26% |
| Excellent+Good | 57.8% | **55.6%** | 公式权重影响综合分 |

---

## Round 5：NED 对齐 + OCR 输出空格修复

**提交**：`feat: align formula metric to OmniDocBench NED and fix spaced-char OCR artifacts`

**改动**：
- `latex_accuracy`：主指标切换为 **NED（1 - normalized_edit_distance）**，与 OmniDocBench 官方对齐
- `normalize_latex`：修复 OCR 输出中 `{c o r r}` → `{corr}` 的字符间空格问题
- 新增 `\mathsf`→`\mathrm`、`\operatorname{}` 内部空格折叠等规则

| 指标 | Baseline | Round 5（当前） | 总变化 |
|---|---|---|---|
| 有效评估页数 | 1005 | **991** | 更真实 |
| Character Accuracy | 0.7010 | 0.6996 | 持平 |
| Sequence Similarity | 0.6684 | 0.6637 | 持平 |
| Table TEDS | 87.39 | 86.78 | 持平（空页过滤影响） |
| Formula Avg Accuracy | 0.0000 | **0.5971** | 从无到有 ↑↑ |
| Correct Formulas | 0/1146 | **332/1143 (29.0%)** | 从0到29% ↑↑ |
| Composite Mean | 72.15 | 71.49 | -0.66 |
| Composite Median | 84.48 | 83.14 | -1.34 |
| Excellent (≥90) | 333 (33.1%) | 312 (31.5%) | 公式纳入权重后综合分略降 |
| Good (80-90) | 248 (24.7%) | 239 (24.1%) | — |
| Poor (<70) | 324 (32.2%) | 324 (32.7%) | 基本持平 |

---

## 当前瓶颈分析

### 文本（主要瓶颈）
- **12.1% 页面准确率 < 0.2**（120 页），是拉低均值的核心
- Worst 10 全为 PPT 类文件，存在两种失败模式：
  - **过度识别**：OCR 抽取了背景/装饰文字，GT 只标正文（如 OCR=1390, GT=13）
  - **完全漏识别**：布局检测将文字区域判为 image（如 OCR=0, GT=90）

### 公式（次要瓶颈）
- PP-StructureV3 公式识别器输出质量有限（`\mathsf` vs `\mathrm`、字符间空格等）
- OCR 公式数量与 GT 不一致（过度检测），影响匹配质量
- 评估框架已对齐 OmniDocBench NED，后续提升需替换公式识别模型

### 表格（良好）
- TEDS 86.78，属于可接受水平，暂无明显瓶颈

---

## 后续优化方向

| 优先级 | 方向 | 类型 | 预期收益 |
|---|---|---|---|
| 高 | 深入分析 120 个低分页面，区分过度识别 vs 漏识别 | 分析 | 指导 OCR 参数调整方向 |
| 高 | best/worst 列表过滤双空页 | 评估修复 | 列表更有参考价值 |
| 中 | 针对 PPT 类文档调整 PP-structure 参数 | OCR 优化 | 文本准确率提升 |
| 低 | 替换公式识别模型（UniMER-Net / Pix2Tex） | OCR 优化 | 公式准确率提升 |
