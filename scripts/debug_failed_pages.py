"""
调试脚本：分析失败页面的OCR输出结构

输出内容：
  1. 失败页面的parsing_res_list详细结构
  2. 每个block的label、content长度、bbox信息
  3. 原因分析：为什么text没被提取
  4. 建议改进方案
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from src.utils import load_config

CONFIG = load_config()
PATHS = CONFIG.get("paths", {})
OCR_STRUCTURED_DIR = PATHS.get("ocr_structured_dir", "results/ocr_structured")
EVALUATION_CSV = "results/ocr_quality_evaluation_full.csv"


class FailedPageDebugger:
    """失败页面调试工具"""

    def __init__(self):
        self.evaluation_results = []
        self.ocr_data = {}
        self._load_data()

    def _load_data(self):
        """加载评估结果和OCR数据"""
        # 加载评估CSV
        if not Path(EVALUATION_CSV).exists():
            print(f"Warning: {EVALUATION_CSV} not found")
            return

        with open(EVALUATION_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.evaluation_results = list(reader)

        print(f"Loaded {len(self.evaluation_results)} evaluation results")

        # 加载OCR数据
        ocr_dir = Path(OCR_STRUCTURED_DIR)
        if not ocr_dir.exists():
            print(f"Warning: OCR dir not found: {ocr_dir}")
            return

        for json_file in ocr_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list) and len(data) > 0:
                    page_item = data[0]  # 每个JSON文件通常只有一页
                    filename = page_item.get("filename", "")
                    page_index = page_item.get("page_index", 0)
                    key = f"{Path(filename).stem}@page{page_index}"
                    self.ocr_data[key] = page_item
            except Exception as e:
                pass

        print(f"Loaded OCR data for {len(self.ocr_data)} pages")

    def classify_failed_pages(self) -> Dict[str, List[Dict]]:
        """分类失败页面"""
        failed = {
            "text_underflow": [],  # OCR < GT
            "text_overflow": [],   # OCR > GT
        }

        for result in self.evaluation_results:
            try:
                filename = result.get("filename", "")
                page = result.get("page_index", "0")
                text_ocr = int(result.get("text_length_ocr", 0))
                text_gt = int(result.get("text_length_gt", 0))

                key = f"{Path(filename).stem}@page{page}"

                # Text Underflow: OCR < GT 50%
                if text_gt > 0 and text_ocr < text_gt * 0.5:
                    failed["text_underflow"].append({
                        "key": key,
                        "filename": filename,
                        "page": page,
                        "text_ocr": text_ocr,
                        "text_gt": text_gt,
                    })

                # Text Overflow: OCR > GT 150%
                elif text_gt > 0 and text_ocr > text_gt * 1.5:
                    failed["text_overflow"].append({
                        "key": key,
                        "filename": filename,
                        "page": page,
                        "text_ocr": text_ocr,
                        "text_gt": text_gt,
                    })
            except (ValueError, KeyError):
                continue

        return failed

    def analyze_parsing_structure(self, page_data: Dict) -> Dict:
        """分析parsing_res_list的结构"""
        core_content = page_data.get("core_content", {})
        parsing_res_list = core_content.get("parsing_res_list", [])

        analysis = {
            "total_blocks": len(parsing_res_list),
            "block_labels": defaultdict(int),
            "block_details": [],
            "text_blocks": [],
            "table_blocks": [],
            "image_blocks": [],
            "other_blocks": [],
        }

        for i, res in enumerate(parsing_res_list):
            block_label = res.get("block_label", "unknown")
            block_content = res.get("block_content", "")
            block_bbox = res.get("block_bbox", [])

            analysis["block_labels"][block_label] += 1

            block_info = {
                "index": i,
                "label": block_label,
                "content_length": len(block_content),
                "content_preview": block_content[:100] if block_content else "",
                "has_bbox": bool(block_bbox),
                "bbox": block_bbox,
            }

            analysis["block_details"].append(block_info)

            # 分类统计
            if block_label in ["text", "paragraph", "title", "list_item", "paragraph_title"]:
                analysis["text_blocks"].append(block_info)
            elif block_label == "table":
                analysis["table_blocks"].append(block_info)
            elif block_label in ["image", "figure"]:
                analysis["image_blocks"].append(block_info)
            else:
                analysis["other_blocks"].append(block_info)

        return analysis

    def analyze_layout_detection(self, page_data: Dict) -> Dict:
        """分析PP-StructureV3的layout检测结果"""
        metadata = page_data.get("metadata", {})
        layout_det_res = metadata.get("layout_det_res", {})
        boxes = layout_det_res.get("boxes", [])

        analysis = {
            "total_detected": len(boxes),
            "by_label": defaultdict(list),
        }

        for box in boxes:
            label = box.get("label", "unknown")
            score = box.get("score", 0.0)
            coordinate = box.get("coordinate", [])

            analysis["by_label"][label].append({
                "score": round(score, 4),
                "coordinate": coordinate,
            })

        # 按label统计
        analysis["detection_count"] = {
            label: len(boxes_list)
            for label, boxes_list in analysis["by_label"].items()
        }

        # 按label和置信度统计
        analysis["detection_stats"] = {}
        for label, boxes_list in analysis["by_label"].items():
            scores = [b["score"] for b in boxes_list]
            analysis["detection_stats"][label] = {
                "count": len(scores),
                "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
                "min_score": round(min(scores), 4) if scores else 0,
                "max_score": round(max(scores), 4) if scores else 0,
            }

        return analysis

    def generate_debug_report(self, failed: Dict) -> str:
        """生成调试报告"""
        report = []
        report.append("# 失败页面调试报告\n")

        # === Text Underflow 分析 ===
        report.append("## Text Underflow 页面（OCR < GT）\n")
        report.append(f"**总数**: {len(failed['text_underflow'])}页\n")

        if failed["text_underflow"]:
            # 分析前10页
            for i, page_info in enumerate(failed["text_underflow"][:10]):
                key = page_info["key"]
                page_data = self.ocr_data.get(key)

                if not page_data:
                    report.append(f"### {i+1}. {key}")
                    report.append(f"❌ 数据未找到\n")
                    continue

                report.append(f"### {i+1}. {key}")
                report.append(f"- OCR文本: {page_info['text_ocr']} 字符")
                report.append(f"- GT文本: {page_info['text_gt']} 字符")
                report.append(f"- 比例: {page_info['text_ocr'] / page_info['text_gt']:.1%}\n")

                # 分析parsing_res_list
                parsing_analysis = self.analyze_parsing_structure(page_data)
                report.append("**Block标签分布**:")
                for label, count in sorted(parsing_analysis["block_labels"].items()):
                    report.append(f"  - {label}: {count}个")
                report.append("")

                # 文本块分析
                report.append("**文本块详情**:")
                if parsing_analysis["text_blocks"]:
                    for block in parsing_analysis["text_blocks"][:3]:
                        report.append(f"  - {block['label']}: {block['content_length']} 字符")
                        report.append(f"    预览: {block['content_preview'][:50]}...")
                else:
                    report.append("  ⚠️ 没有识别出text块！")
                report.append("")

                # 表格块分析
                if parsing_analysis["table_blocks"]:
                    report.append("**表格块数**: " + str(len(parsing_analysis["table_blocks"])))
                    for block in parsing_analysis["table_blocks"][:2]:
                        report.append(f"  - HTML长度: {block['content_length']} 字符")

                # Layout检测分析
                layout_analysis = self.analyze_layout_detection(page_data)
                report.append("\n**PP-StructureV3检测结果**:")
                for label, stats in layout_analysis["detection_stats"].items():
                    report.append(f"  - {label}: {stats['count']}个 (置信度: {stats['avg_score']:.4f})")
                report.append("")

                # 原因分析
                report.append("**原因分析**:")
                if not parsing_analysis["text_blocks"]:
                    report.append("  🔴 Critical: parsing_res_list中没有text块")
                    report.append("  → PP-StructureV3检测失败，未识别出文本区域")
                    report.append("  → 或文本被错误标记为table/image")
                elif parsing_analysis["text_blocks"]:
                    total_text_len = sum(b["content_length"] for b in parsing_analysis["text_blocks"])
                    if total_text_len < page_info['text_gt'] * 0.3:
                        report.append("  🟠 文本块存在但内容过少")
                        report.append("  → OCR识别准确度低")
                        report.append("  → 文本可能被部分分割或过滤")
                report.append("")

            # 统计汇总
            report.append("### 统计汇总\n")
            no_text_blocks = 0
            low_confidence_detect = 0
            for page_info in failed["text_underflow"]:
                key = page_info["key"]
                page_data = self.ocr_data.get(key)
                if page_data:
                    parsing_analysis = self.analyze_parsing_structure(page_data)
                    if not parsing_analysis["text_blocks"]:
                        no_text_blocks += 1

                    layout_analysis = self.analyze_layout_detection(page_data)
                    if layout_analysis["detection_stats"].get("text", {}).get("max_score", 0) < 0.3:
                        low_confidence_detect += 1

            report.append(f"- 没有text块的页面: **{no_text_blocks}** ({no_text_blocks/len(failed['text_underflow'])*100:.1f}%)")
            report.append(f"- 低置信度检测的页面: **{low_confidence_detect}** ({low_confidence_detect/len(failed['text_underflow'])*100:.1f}%)")
            report.append("")

        # === Text Overflow 分析 ===
        report.append("## Text Overflow 页面（OCR > GT）\n")
        report.append(f"**总数**: {len(failed['text_overflow'])}页\n")

        if failed["text_overflow"]:
            # 按GT长度分类
            gt_zero = [p for p in failed["text_overflow"] if p["text_gt"] == 0]
            gt_nonzero = [p for p in failed["text_overflow"] if p["text_gt"] > 0]

            report.append(f"- GT长度=0的页面: **{len(gt_zero)}**页 (应该没有文本)")
            report.append(f"- GT长度>0的页面: **{len(gt_nonzero)}**页 (文本被误识别)\n")

            # 分析GT=0的页面
            if gt_zero:
                report.append("**GT=0的页面（纯图片/表格）**:\n")
                for i, page_info in enumerate(gt_zero[:5]):
                    key = page_info["key"]
                    page_data = self.ocr_data.get(key)

                    if not page_data:
                        continue

                    report.append(f"### {i+1}. {key}")
                    report.append(f"- OCR文本: {page_info['text_ocr']} 字符")
                    report.append(f"- GT: 0 字符\n")

                    parsing_analysis = self.analyze_parsing_structure(page_data)
                    report.append("**Block标签分布**:")
                    for label, count in sorted(parsing_analysis["block_labels"].items()):
                        report.append(f"  - {label}: {count}个")
                    report.append("")

                    # 分析原因
                    if parsing_analysis["text_blocks"]:
                        total_len = sum(b["content_length"] for b in parsing_analysis["text_blocks"])
                        report.append(f"**发现{len(parsing_analysis['text_blocks'])}个text块，总长{total_len}字符**")
                        report.append("→ 这些是表格HTML还是真实文本？\n")

                    if parsing_analysis["table_blocks"]:
                        report.append(f"**发现{len(parsing_analysis['table_blocks'])}个table块**")
                        report.append("→ 表格被提取为text了吗？\n")

            report.append("")

        # === 总体建议 ===
        report.append("## 改进建议\n")
        report.append("### 优先级1: 修复Text Underflow (文本完全丢失)\n")

        if failed["text_underflow"]:
            page_data = self.ocr_data.get(failed["text_underflow"][0]["key"])
            if page_data:
                layout_analysis = self.analyze_layout_detection(page_data)
                if "text" in layout_analysis["detection_stats"]:
                    text_stats = layout_analysis["detection_stats"]["text"]
                    if text_stats["count"] > 0:
                        report.append(f"发现: PP-StructureV3检测到{text_stats['count']}个text区域")
                        report.append(f"平均置信度: {text_stats['avg_score']:.4f}")

                        if text_stats["avg_score"] < 0.3:
                            report.append("\n**问题**: 文本检测置信度太低（<0.3）")
                            report.append("**解决方案**:")
                            report.append("1. 降低score_threshold: 0.3 → 0.25")
                            report.append("2. 或检查是否需要提高文本检测模型质量")
                    else:
                        report.append("**问题**: PP-StructureV3完全未检测出text区域")
                        report.append("**解决方案**:")
                        report.append("1. 检查是否text被标记为table/image")
                        report.append("2. 增加检测模型的recall，可能需要降低score_threshold")

        report.append("\n### 优先级2: 优化Text Overflow (空白页处理)\n")
        report.append("当前权重调整(text权重0.2)已生效，但源头问题未解决")
        report.append("建议继续关注Text Underflow的修复\n")

        return "\n".join(report)

    def save_report(self, report: str, output_path: str = "results/debug_report.md"):
        """保存调试报告"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Debug report saved to {output_path}")


if __name__ == "__main__":
    debugger = FailedPageDebugger()

    # 分类失败页面
    failed = debugger.classify_failed_pages()
    print(f"\nFound {len(failed['text_underflow'])} Text Underflow pages")
    print(f"Found {len(failed['text_overflow'])} Text Overflow pages")

    # 生成调试报告
    report = debugger.generate_debug_report(failed)
    print("\n" + "="*80)
    print(report)
    print("="*80)

    # 保存报告
    debugger.save_report(report)