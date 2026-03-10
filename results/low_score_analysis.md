# Low Score Page Analysis Report

- Threshold: character_accuracy < 0.2
- Total content pages: 991
- Low score pages: **120** (12.1%)

## Failure Mode Distribution

| 失败模式 | 页数 | 占低分页 | 描述 |
|---|---|---|---|
| 过度识别 | 59 | 49.2% | OCR 文字量 > GT 的 2.0x |
| 漏识别 | 41 | 34.2% | OCR 文字量 < GT 的 0.5x |
| 字符错误 | 20 | 16.7% | 数量接近但识别字符错误 |

## Optimization Suggestions

### 过度识别（59 页）

⚠️ 提高 text_det_thresh 或过滤低置信度 block

**文档类型分布：**

| 类型 | 页数 |
|---|---|
| 其他 | 43 |
| 电子书 | 7 |
| PPT | 5 |
| 书籍 | 3 |
| 教材 | 1 |

**文字长度统计：**

| 指标 | OCR | GT |
|---|---|---|
| 平均 | 215 | 23 |
| 最大 | 3552 | 315 |
| 最小 | 4 | 0 |

**最差样本（前10）：**

| filename | page | char_acc | ocr_len | gt_len |
|---|---|---|---|---|
| PPT_13 fallacies LALT2012_page_010 | 10 | 0.0000 | 1390 | 13 |
| PPT_B14_Claeys_Consistent Presentation of Images_v2_page_006 | 6 | 0.0000 | 138 | 25 |
| PPT_B14_Claeys_Consistent Presentation of Images_v2_page_010 | 10 | 0.0000 | 102 | 28 |
| PPT_ch7_page_028 | 28 | 0.0000 | 156 | 45 |
| PPT_ch7_page_132 | 132 | 0.0000 | 1019 | 278 |
| book_en_A.Concise.Introduction.to.Linear.Algebra,.Geza.Schay,.Birkhauser,.2012_page_065 | 65 | 0.0000 | 2050 | 123 |
| book_en_[搬书匠#20][HTML5 Canvas].2011.英文版_page_208 | 208 | 0.0000 | 39 | 0 |
| book_en_搬书匠-3282-Python Web Scraping 2nd Edition-2017-英文版_page_195 | 195 | 0.0000 | 859 | 315 |
| color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点五年级英语下册_课本_人教新起点英语5B电子课本_page_036 | 36 | 0.0000 | 421 | 167 |
| docstructbench_dianzishu_zhongwenzaixian-o.O-60403612.pdf_179 | 179 | 0.0000 | 10 | 0 |

### 漏识别（41 页）

⚠️ 布局检测把文字区域判成 image，或降低 text_det_thresh

**文档类型分布：**

| 类型 | 页数 |
|---|---|
| 其他 | 20 |
| PPT | 9 |
| 教材 | 9 |
| 电子书 | 2 |
| 书籍 | 1 |

**文字长度统计：**

| 指标 | OCR | GT |
|---|---|---|
| 平均 | 22 | 367 |
| 最大 | 298 | 2029 |
| 最小 | 0 | 4 |

**最差样本（前10）：**

| filename | page | char_acc | ocr_len | gt_len |
|---|---|---|---|---|
| PPT_CalculusReview_page_009 | 9 | 0.0000 | 0 | 90 |
| PPT_Catalysis.ppt_page_016 | 16 | 0.0000 | 0 | 38 |
| PPT_session 10_page_022 | 22 | 0.0000 | 0 | 39 |
| PPT_session 10_page_023 | 23 | 0.0000 | 0 | 39 |
| PPT_session 10_page_028 | 28 | 0.0000 | 0 | 45 |
| color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点五年级英语下册_课本_人教新起点英语5B电子课本_page_046 | 46 | 0.0000 | 0 | 15 |
| color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点四年级英语上册_课本_人教新起点英语4A电子课本_page_091 | 91 | 0.0000 | 0 | 74 |
| color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点四年级英语下册_课本_人教新起点英语4B电子课本_page_017 | 17 | 0.0000 | 0 | 13 |
| color_textbook_zhonggaokao_小学_13.人教新起点英语（4-5年级）_人教新起点四年级英语下册_课本_人教新起点英语4B电子课本_page_043 | 43 | 0.0000 | 0 | 14 |
| color_textbook_zhonggaokao_小学_KET听说读写逐项突破_KET听说读写逐项突破之轻松搞定KET写作25分 【10讲 褚连一】_第04讲第四课KET分类词汇训练二_第四课KET分类词汇训练二_page_003 | 3 | 0.0000 | 0 | 1350 |

### 字符错误（20 页）

⚠️ 图像质量差或复杂字体，考虑预处理或换模型

**文档类型分布：**

| 类型 | 页数 |
|---|---|
| 其他 | 10 |
| PPT | 6 |
| 书籍 | 3 |
| 学术论文 | 1 |

**文字长度统计：**

| 指标 | OCR | GT |
|---|---|---|
| 平均 | 622 | 522 |
| 最大 | 5157 | 5410 |
| 最小 | 15 | 9 |

**最差样本（前10）：**

| filename | page | char_acc | ocr_len | gt_len |
|---|---|---|---|---|
| eastmoney_d09a006aa02ddc09299bbb9a1b5efa0d77408191f0c1ff1fca8c80bd6150f806.pdf_17 | 17 | 0.0000 | 93 | 63 |
| yanbaopptmerge_2b8553b00244437fa3e502aa2d3d319ed74459a1e264a4fdd9ecc14ce46609d5.pdf_2 | 2 | 0.0000 | 37 | 21 |
| yanbaopptmerge_abef2a4978ae4d13e931f0392502bd40.pdf_1287 | 1287 | 0.0000 | 2636 | 1737 |
| yanbaopptmerge_yanbaoPPT_4710 | 4710 | 0.0000 | 47 | 27 |
| yanbaor2_yanbaoPPT_4618 | 4618 | 0.0000 | 254 | 185 |
| PPT_linear-algebra primer_page_008 | 8 | 0.0296 | 516 | 270 |
| PPT_ch7_page_053 | 53 | 0.0408 | 377 | 196 |
| book_en_搬书匠-3299-Swift Data Structure and Algorithms-2016-英文版_page_111 | 111 | 0.0453 | 1042 | 662 |
| PPT_lay_linalg5_01_05_page_009 | 9 | 0.0609 | 199 | 115 |
| docstructbench_enbook-zlib-o.O-15322190.pdf_138 | 138 | 0.0690 | 16 | 29 |
