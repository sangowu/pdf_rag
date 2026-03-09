# 失败页面诊断报告

## 摘要

- 总页数: 1005
- Text Overflow (OCR > GT): **81页 (8.1%)**
- Text Underflow (OCR < GT): **67页 (6.7%)**
- 可接受 (差异<50%): 843页 (83.9%)
- 空页面 (都为0): 14页 (1.4%)

## Text Overflow 页面（表格混入文本）

**问题**: OCR提取文本 > GT 50%以上，可能包含表格HTML
**数量**: 81页

### 最严重的10页
| 页面 | OCR长度 | GT长度 | 比例 | Char Acc |
|-----|--------|--------|------|----------|
| book_en_[搬书匠#20][HTML5 Canvas].2011.英文版_page_208@page208 | 39 | 0 | infx | 0.0000 |
| docstructbench_dianzishu_zhongwenzaixian-o.O-60403612.pdf_179@page179 | 10 | 0 | infx | 0.0000 |
| docstructbench_dianzishu_zhongwenzaixian-o.O-61520553.pdf_328@page328 | 69 | 0 | infx | 0.0000 |
| docstructbench_dianzishu_zhongwenzaixian-o.O-61520779.pdf_76@page76 | 20 | 0 | infx | 0.0000 |
| docstructbench_dianzishu_zhongwenzaixian-o.O-61521429.pdf_358@page358 | 23 | 0 | infx | 0.0000 |
| docstructbench_dianzishu_zhongwenzaixian-o.O-61571259.pdf_239@page239 | 57 | 0 | infx | 0.0000 |
| docstructbench_dianzishu_zhongwenzaixian-o.O-63709763.pdf_72@page72 | 30 | 0 | infx | 0.0000 |
| docstructbench_dianzishu_zhongwenzaixian-o.O-63711094.pdf_34@page34 | 33 | 0 | infx | 0.0000 |
| docstructbench_llm-raw-scihub-o.O-dvdy.10165.pdf_7@page7 | 15 | 0 | infx | 0.0000 |
| docstructbench_llm-raw-scihub-o.O-j.apcata.2004.01.008.pdf_6@page6 | 60 | 0 | infx | 0.0000 |

**统计**: 平均比例 infx, 最大 infx

## Text Underflow 页面（文本被过滤）

**问题**: OCR提取文本 < GT 50%以下，可能被误识别或过滤
**数量**: 67页

### 最严重的10页
| 页面 | OCR长度 | GT长度 | 比例 | Char Acc |
|-----|--------|--------|------|----------|
| PPT_CalculusReview_page_009@page9 | 0 | 90 | 0.0x | 0.0000 |
| PPT_Catalysis.ppt_page_016@page16 | 0 | 38 | 0.0x | 0.0000 |
| PPT_session 10_page_022@page22 | 0 | 39 | 0.0x | 0.0000 |
| PPT_session 10_page_023@page23 | 0 | 39 | 0.0x | 0.0000 |
| PPT_session 10_page_028@page28 | 0 | 45 | 0.0x | 0.0000 |
| book_en_[搬书匠#893][Pyomo—Optimization Modeling in Python].2012.英文版_page_016@page16 | 8 | 1822 | 0.0x | 0.0044 |
| color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点五年级英语下册_课本_人教新起点英语5B电子课本_page_046@page46 | 0 | 15 | 0.0x | 0.0000 |
| color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点四年级英语上册_课本_人教新起点英语4A电子课本_page_091@page91 | 0 | 74 | 0.0x | 0.0000 |
| color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点四年级英语下册_课本_人教新起点英语4B电子课本_page_017@page17 | 0 | 13 | 0.0x | 0.0000 |
| color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点四年级英语下册_课本_人教新起点英语4B电子课本_page_043@page43 | 0 | 14 | 0.0x | 0.0000 |

**统计**: 平均比例 0.19x

## 诊断结论

🔴 **主要问题: Text Overflow (55%)**

**原因分析**:
- OCR提取的文本远大于GT，说明包含了表格或图片的HTML内容
- bbox隔离可能未正确过滤表格区域
- 或PP-StructureV3的表格检测不准确

**改进方向**:
1. 调整bbox隔离参数: overlap_threshold 0.2 → 0.05
2. 增加bbox padding范围，扩大表格检测边界
3. 或调整PP-StructureV3检测阈值: score_threshold 0.3 → 0.4
