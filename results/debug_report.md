# 失败页面调试报告

## Text Underflow 页面（OCR < GT）

**总数**: 67页

### 1. PPT_1637904769_page_002@page2
- OCR文本: 9 字符
- GT文本: 179 字符
- 比例: 5.0%

**Block标签分布**:
  - text: 2个

**文本块详情**:
  - text: 10 字符
    预览: WHa, Yd H ...
  - text: 0 字符
    预览: ...


**PP-StructureV3检测结果**:
  - text: 3个 (置信度: 0.4780)

**原因分析**:
  🟠 文本块存在但内容过少
  → OCR识别准确度低
  → 文本可能被部分分割或过滤

### 2. PPT_1637904769_page_008@page8
- OCR文本: 27 字符
- GT文本: 191 字符
- 比例: 14.1%

**Block标签分布**:
  - text: 2个

**文本块详情**:
  - text: 25 字符
    预览: TEACHING/LEARNING ENGISH ...
  - text: 2 字符
    预览: y ...


**PP-StructureV3检测结果**:
  - text: 8个 (置信度: 0.4944)
  - paragraph_title: 1个 (置信度: 0.3307)

**原因分析**:
  🟠 文本块存在但内容过少
  → OCR识别准确度低
  → 文本可能被部分分割或过滤

### 3. PPT_CalculusReview_page_009@page9
- OCR文本: 0 字符
- GT文本: 90 字符
- 比例: 0.0%

**Block标签分布**:
  - chart: 1个
  - figure_title: 1个

**文本块详情**:
  ⚠️ 没有识别出text块！


**PP-StructureV3检测结果**:
  - figure_title: 1个 (置信度: 0.8063)
  - chart: 1个 (置信度: 0.7923)
  - formula: 1个 (置信度: 0.7165)

**原因分析**:
  🔴 Critical: parsing_res_list中没有text块
  → PP-StructureV3检测失败，未识别出文本区域
  → 或文本被错误标记为table/image

### 4. PPT_Catalysis@page16
❌ 数据未找到

### 5. PPT_EnglishtoAmericanTransition_page_003@page3
- OCR文本: 73 字符
- GT文本: 282 字符
- 比例: 25.9%

**Block标签分布**:
  - doc_title: 1个
  - paragraph_title: 1个
  - text: 2个

**文本块详情**:
  - text: 2 字符
    预览: y ...
  - paragraph_title: 8 字符
    预览: In 1979,...
  - text: 27 字符
    预览: Even Price Charles claimed ...


**PP-StructureV3检测结果**:
  - text: 6个 (置信度: 0.4825)
  - doc_title: 1个 (置信度: 0.5021)
  - paragraph_title: 1个 (置信度: 0.3464)

**原因分析**:
  🟠 文本块存在但内容过少
  → OCR识别准确度低
  → 文本可能被部分分割或过滤

### 6. PPT_EnglishtoAmericanTransition_page_012@page12
- OCR文本: 60 字符
- GT文本: 150 字符
- 比例: 40.0%

**Block标签分布**:
  - paragraph_title: 2个
  - text: 2个

**文本块详情**:
  - paragraph_title: 31 字符
    预览: Changes to English (in America)...
  - paragraph_title: 24 字符
    预览: □Change7 (pronunciation)...
  - text: 0 字符
    预览: ...


**PP-StructureV3检测结果**:
  - text: 2个 (置信度: 0.5974)
  - paragraph_title: 2个 (置信度: 0.5604)

**原因分析**:

### 7. PPT_Keuk Chan Narith_page_005@page5
- OCR文本: 46 字符
- GT文本: 172 字符
- 比例: 26.7%

**Block标签分布**:
  - text: 2个

**文本块详情**:
  - text: 47 字符
    预览: An EmergenceofEnglishlanguagevarietyinCambodia ...
  - text: 0 字符
    预览: ...


**PP-StructureV3检测结果**:
  - text: 3个 (置信度: 0.6052)

**原因分析**:
  🟠 文本块存在但内容过少
  → OCR识别准确度低
  → 文本可能被部分分割或过滤

### 8. PPT_esea-app101_page_015@page15
- OCR文本: 16 字符
- GT文本: 136 字符
- 比例: 11.8%

**Block标签分布**:
  - paragraph_title: 1个
  - text: 1个

**文本块详情**:
  - paragraph_title: 10 字符
    预览: Questions:...
  - text: 6 字符
    预览: TnrLE ...


**PP-StructureV3检测结果**:
  - text: 1个 (置信度: 0.8742)
  - paragraph_title: 1个 (置信度: 0.5519)

**原因分析**:
  🟠 文本块存在但内容过少
  → OCR识别准确度低
  → 文本可能被部分分割或过滤

### 9. PPT_lecture1_page_005@page5
- OCR文本: 191 字符
- GT文本: 409 字符
- 比例: 46.7%

**Block标签分布**:
  - doc_title: 1个
  - formula: 1个
  - text: 2个

**文本块详情**:
  - text: 77 字符
    预览: A collection of one or more equations involving t ...
  - text: 85 字符
    预览: Hence it is a system of equations involving the va...


**PP-StructureV3检测结果**:
  - text: 6个 (置信度: 0.5860)
  - formula: 2个 (置信度: 0.5720)
  - paragraph_title: 1个 (置信度: 0.7644)

**原因分析**:

### 10. PPT_lecture1_page_007@page7
- OCR文本: 51 字符
- GT文本: 391 字符
- 比例: 13.0%

**Block标签分布**:
  - formula: 1个
  - text: 1个

**文本块详情**:
  - text: 52 字符
    预览: Matrix representation of system of linear equation...


**PP-StructureV3检测结果**:
  - text: 1个 (置信度: 0.8287)
  - formula: 1个 (置信度: 0.3019)

**原因分析**:
  🟠 文本块存在但内容过少
  → OCR识别准确度低
  → 文本可能被部分分割或过滤

### 统计汇总

- 没有text块的页面: **13** (19.4%)
- 低置信度检测的页面: **14** (20.9%)

## Text Overflow 页面（OCR > GT）

**总数**: 46页

- GT长度=0的页面: **0**页 (应该没有文本)
- GT长度>0的页面: **46**页 (文本被误识别)


## 改进建议

### 优先级1: 修复Text Underflow (文本完全丢失)

发现: PP-StructureV3检测到3个text区域
平均置信度: 0.4780

### 优先级2: 优化Text Overflow (空白页处理)

当前权重调整(text权重0.2)已生效，但源头问题未解决
建议继续关注Text Underflow的修复
