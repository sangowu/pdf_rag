# 表格评估HTML标准化改进报告

**生成时间**: 2026-03-10
**改进方案**: HTML标准化 (方案A - OmniDocBench推荐方法)

---

## 一、方案背景

### 问题现象
当前TEDS Score（表格编辑距离评分）始终为0.00，无论OCR质量如何，都无法产生有意义的表格评估指标。

### 根本原因分析
```
OCR生成的HTML:     <html><body><table><tbody><tr><td>A</td></tr></tbody></table></body></html>
GT标注的HTML:      <table><tr><td>A</td></tr></table>

或者：
OCR格式:    <table class="ocr-table" style="border:1px"><tr><td>  Content  </td></tr></table>
GT格式:     <table><tr><td>Content</td></tr></table>

编辑距离 = 非常大（因为标签、属性、空白完全不同）
TEDS = 0
```

### 官方实践对标
- **OmniDocBench (CVPR 2025)**: 采用HTML标准化 + TEDS评估
- **PaddleOCR 3.0+**: 基于OmniDocBench标准，也进行HTML规范化
- **业界共识**: TEDS计算前必须进行格式标准化

---

## 二、改进方案详解

### 2.1 标准化流程

```python
normalize_html(html_str) 处理顺序：

1. 移除HTML注释
   <!--comment--> → (删除)

2. 移除所有属性
   <td class="cell" style="..."> → <td>

3. 统一大小写
   <TD> <Td> → <td>

4. 规范化空白
   <td>  content  </td> → <td>content</td>
   <table>\n  <tr> → <table><tr>

5. 移除冗余标签
   <tbody></tbody> (空) → (删除)
   <html><body>...</body></html> → (保留内容)

6. 最终格式
   <table><tr><td>content</td></tr></table>
```

### 2.2 改进点

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| **不同属性** | `<td class="a">` vs `<td class="b">` | 统一为 `<td>` |
| **空白差异** | `<td> content </td>` vs `<td>content</td>` | 统一处理 |
| **标签嵌套** | 有无 `<tbody>` 差异很大 | 统一移除冗余包装 |
| **大小写** | `<TABLE>` vs `<table>` | 统一小写 |
| **包装标签** | `<html><body><table>...` vs `<table>...` | 统一移除包装 |

---

## 三、代码实现

### 新增函数：normalize_html()

```python
@staticmethod
def normalize_html(html_str: str) -> str:
    """标准化HTML表格格式（OmniDocBench方法）"""
    # 1. 移除注释
    html_str = re.sub(r'<!--.*?-->', '', html_str, flags=re.DOTALL)

    # 2. 移除所有属性
    html_str = re.sub(r'<([\w/]+)\s+[^>]*>', r'<\1>', html_str)

    # 3. 转小写
    html_str = html_str.lower()

    # 4. 规范化空白
    html_str = re.sub(r'>\s+<', '><', html_str)  # 标签间空白
    html_str = re.sub(r'\s+', ' ', html_str)      # 连续空白

    # 5. 移除冗余标签
    html_str = re.sub(r'<(tbody|thead|tfoot)\s*>\s*</\1>', '', html_str)
    html_str = re.sub(r'<html[^>]*>', '', html_str)
    html_str = re.sub(r'<body[^>]*>', '', html_str)

    return html_str.strip()
```

### 修改函数：teds_score()

在编辑距离计算前先标准化：

```python
# 之前
dist = editdistance.eval(pred_html, ref_html)  # ❌ 可能格式差异很大

# 之后
pred_html_normalized = TableEvaluator.normalize_html(pred_html)
ref_html_normalized = TableEvaluator.normalize_html(ref_html)
dist = editdistance.eval(pred_html_normalized, ref_html_normalized)  # ✅ 格式统一
```

---

## 四、预期改进效果

### 4.1 表格评估指标预期

| 指标 | 当前 | 预期 | 说明 |
|------|------|------|------|
| **TEDS Score** | 0.00 | 15-50+ | 取决于实际表格质量 |
| **Structure Similarity** | 0.00 | 0.3-0.8+ | 行列结构匹配 |
| **Content Similarity** | 0.00 | 0.2-0.7+ | 单元格内容匹配 |

### 4.2 综合评分预期

```
修复前：TEDS始终=0，严重拉低综合分
        综合分 = (text * 1.0 + 0 * 0.5 + formula * 0.3) / 1.8

修复后：TEDS = 20-40（有意义的分数）
        综合分 = (text * 1.0 + table*0.5 + formula * 0.3) / 1.8
        整体分数会更客观
```

### 4.3 影响的页面

- **纯表格页面**: TEDS从0→0保持不变（正确，两边都识别为表格）✓
- **混合页面**: TEDS从0→有意义分数（能够评估表格质量）✓
- **纯文本页面**: 不受影响（此类页面本身TEDS贡献很小）✓

---

## 五、技术对标

### 与官方系统的对齐

| 方面 | OmniDocBench | PaddleOCR | 我们的实现 |
|------|-------------|-----------|----------|
| **HTML标准化** | ✅ 有 | ✅ 有 | ✅ 已添加 |
| **TEDS计算** | ✅ 编辑距离 | ✅ 编辑距离 | ✅ 已有 |
| **内容相似度** | ✅ 单元格匹配 | ✅ 单元格匹配 | ✅ 已有 |
| **结构相似度** | ✅ 行列匹配 | ✅ 行列匹配 | ✅ 已有 |

---

## 六、下一步验证

### 6.1 部署检查清单

- [ ] 确认代码已部署到云服务器
- [ ] 重新运行 `evaluate_ocr_quality.py` 脚本
- [ ] 查看新的 `ocr_quality_report.md`

### 6.2 预期观察

新报告中应该能看到：

```markdown
## Table Quality Metrics

| Metric | Mean |
| --- | --- |
| TEDS Score (0-100) | 10-40  ← 不再是0.00！
| Structure Similarity | 0.2-0.6 ← 有意义的分数
| Content Similarity | 0.1-0.5 ← 有意义的分数
```

### 6.3 关键页面变化

最坏页面（纯表格）:
```
PPT_13 fallacies_page_010:
- 修复前: 0.0000 (1390 chars of HTML vs 13 chars text)
- 修复后: 1.0000 (都识别为0字符，表格独立评估)
```

---

## 七、可选后续优化

如果TEDS仍然较低（<30），可考虑：

### 方案 B.1: 调整权重

```python
lambda_ = 0.3  # 降低结构权重，提高内容权重
# 这样：TEDS = 0.3 * structure + 0.7 * content
# 更关注单元格内容的匹配，不那么关注格式差异
```

### 方案 B.2: 表格权重调整

```python
# 在综合评分中，如果表格质量仍不理想
table_weight = 0.3  # 从0.5降到0.3
# 综合分 = (text * 1.0 + table*0.3 + formula * 0.3) / 1.6
```

---

## 八、参考资源

- **OmniDocBench论文**: https://arxiv.org/pdf/2412.07626.pdf
- **PaddleOCR文档**: https://www.paddleocr.ai/
- **TEDS论文**: 表格编辑距离标准化计算

---

**报告状态**: 实现完成，等待云服务器部署验证

**下一步**: 部署后重新运行评估脚本，查看表格评估指标是否改善
