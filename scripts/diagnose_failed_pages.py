"""
诊断脚本：分析失败页面的根本原因

分类统计：
  - text_overflow: OCR提取的文本远大于GT（可能包含表格HTML）
  - text_underflow: OCR提取的文本远小于GT（可能被过滤或误识别）
  - acceptable: 差异在合理范围内
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

from src.utils import load_config

CONFIG = load_config()
PATHS = CONFIG.get("paths", {})
OCR_STRUCTURED_DIR = PATHS.get("ocr_structured_dir", "results/ocr_structured")
NORMALIZED_DATA_PATH = PATHS.get("normalized_data_path", "data/answers/normalized_data.json")
EVALUATION_CSV = "results/ocr_quality_evaluation_full.csv"


class FailedPageDiagnoser:
    """失败页面诊断工具"""

    def __init__(self):
        self.evaluation_results = []
        self._load_evaluation_results()

    def _load_evaluation_results(self):
        """加载评估结果CSV"""
        if not Path(EVALUATION_CSV).exists():
            print(f"Warning: {EVALUATION_CSV} not found")
            return

        with open(EVALUATION_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.evaluation_results = list(reader)

        print(f"Loaded {len(self.evaluation_results)} evaluation results")

    def classify_pages(self,
                      overflow_threshold: float = 1.5,
                      underflow_threshold: float = 0.5) -> Dict[str, List[Dict]]:
        """
        分类失败页面

        Args:
            overflow_threshold: text_len_ocr / text_len_gt > threshold → overflow
            underflow_threshold: text_len_ocr / text_len_gt < threshold → underflow

        Returns:
            分类结果字典
        """
        classified = {
            "text_overflow": [],      # OCR文本过多（包含表格）
            "text_underflow": [],     # OCR文本过少（被过滤）
            "acceptable": [],         # 可接受范围
            "empty_pages": [],        # 两者都为空
        }

        for result in self.evaluation_results:
            try:
                filename = result.get("filename", "")
                page = result.get("page_index", "0")
                char_acc = float(result.get("character_accuracy", 0))
                text_ocr = int(result.get("text_length_ocr", 0))
                text_gt = int(result.get("text_length_gt", 0))

                key = f"{filename}@page{page}"

                # 都为空
                if text_ocr == 0 and text_gt == 0:
                    classified["empty_pages"].append({
                        "key": key,
                        "text_ocr": text_ocr,
                        "text_gt": text_gt,
                        "char_acc": char_acc,
                        "ratio": 1.0
                    })
                    continue

                # 只有GT有内容
                if text_gt > 0 and text_ocr == 0:
                    classified["text_underflow"].append({
                        "key": key,
                        "text_ocr": text_ocr,
                        "text_gt": text_gt,
                        "char_acc": char_acc,
                        "ratio": 0.0
                    })
                    continue

                # 只有OCR有内容
                if text_ocr > 0 and text_gt == 0:
                    classified["text_overflow"].append({
                        "key": key,
                        "text_ocr": text_ocr,
                        "text_gt": text_gt,
                        "char_acc": char_acc,
                        "ratio": float('inf')
                    })
                    continue

                # 两者都有内容
                ratio = text_ocr / text_gt
                page_info = {
                    "key": key,
                    "text_ocr": text_ocr,
                    "text_gt": text_gt,
                    "char_acc": char_acc,
                    "ratio": round(ratio, 2),
                    "diff": text_ocr - text_gt
                }

                if ratio > overflow_threshold:
                    classified["text_overflow"].append(page_info)
                elif ratio < underflow_threshold:
                    classified["text_underflow"].append(page_info)
                else:
                    classified["acceptable"].append(page_info)

            except (ValueError, KeyError) as e:
                print(f"Error processing result: {e}")
                continue

        return classified

    def generate_report(self, classified: Dict) -> str:
        """生成诊断报告"""
        report = []
        report.append("# 失败页面诊断报告\n")

        # 统计摘要
        overflow = len(classified["text_overflow"])
        underflow = len(classified["text_underflow"])
        acceptable = len(classified["acceptable"])
        empty = len(classified["empty_pages"])
        total = overflow + underflow + acceptable + empty

        report.append("## 摘要\n")
        report.append(f"- 总页数: {total}")
        report.append(f"- Text Overflow (OCR > GT): **{overflow}页 ({overflow/total*100:.1f}%)**")
        report.append(f"- Text Underflow (OCR < GT): **{underflow}页 ({underflow/total*100:.1f}%)**")
        report.append(f"- 可接受 (差异<50%): {acceptable}页 ({acceptable/total*100:.1f}%)")
        report.append(f"- 空页面 (都为0): {empty}页 ({empty/total*100:.1f}%)")
        report.append("")

        # 详细分析
        report.append("## Text Overflow 页面（表格混入文本）\n")
        report.append(f"**问题**: OCR提取文本 > GT 50%以上，可能包含表格HTML")
        report.append(f"**数量**: {overflow}页")
        report.append("")

        if overflow > 0:
            # 按ratio排序，最严重的在前
            sorted_overflow = sorted(
                classified["text_overflow"],
                key=lambda x: x["ratio"],
                reverse=True
            )

            report.append("### 最严重的10页")
            report.append("| 页面 | OCR长度 | GT长度 | 比例 | Char Acc |")
            report.append("|-----|--------|--------|------|----------|")

            for item in sorted_overflow[:10]:
                report.append(
                    f"| {item['key']} | {item['text_ocr']} | {item['text_gt']} | "
                    f"{item['ratio']:.1f}x | {item['char_acc']:.4f} |"
                )

            # 统计分析
            overflow_ratios = [x["ratio"] for x in classified["text_overflow"]]
            report.append("")
            report.append(f"**统计**: 平均比例 {sum(overflow_ratios)/len(overflow_ratios):.2f}x, "
                         f"最大 {max(overflow_ratios):.2f}x")
            report.append("")

        # Text Underflow
        report.append("## Text Underflow 页面（文本被过滤）\n")
        report.append(f"**问题**: OCR提取文本 < GT 50%以下，可能被误识别或过滤")
        report.append(f"**数量**: {underflow}页")
        report.append("")

        if underflow > 0:
            # 按ratio排序，最严重的在前
            sorted_underflow = sorted(
                classified["text_underflow"],
                key=lambda x: x["ratio"]
            )

            report.append("### 最严重的10页")
            report.append("| 页面 | OCR长度 | GT长度 | 比例 | Char Acc |")
            report.append("|-----|--------|--------|------|----------|")

            for item in sorted_underflow[:10]:
                report.append(
                    f"| {item['key']} | {item['text_ocr']} | {item['text_gt']} | "
                    f"{item['ratio']:.1f}x | {item['char_acc']:.4f} |"
                )

            # 统计分析
            underflow_ratios = [x["ratio"] for x in classified["text_underflow"]]
            report.append("")
            report.append(f"**统计**: 平均比例 {sum(underflow_ratios)/len(underflow_ratios):.2f}x")
            report.append("")

        # 结论
        report.append("## 诊断结论\n")

        if overflow > underflow:
            percentage = overflow / (overflow + underflow) * 100 if (overflow + underflow) > 0 else 0
            report.append(f"🔴 **主要问题: Text Overflow ({percentage:.0f}%)**")
            report.append("")
            report.append("**原因分析**:")
            report.append("- OCR提取的文本远大于GT，说明包含了表格或图片的HTML内容")
            report.append("- bbox隔离可能未正确过滤表格区域")
            report.append("- 或PP-StructureV3的表格检测不准确")
            report.append("")
            report.append("**改进方向**:")
            report.append("1. 调整bbox隔离参数: overlap_threshold 0.2 → 0.05")
            report.append("2. 增加bbox padding范围，扩大表格检测边界")
            report.append("3. 或调整PP-StructureV3检测阈值: score_threshold 0.3 → 0.4")
            report.append("")

        elif underflow > overflow:
            percentage = underflow / (overflow + underflow) * 100 if (overflow + underflow) > 0 else 0
            report.append(f"🔴 **主要问题: Text Underflow ({percentage:.0f}%)**")
            report.append("")
            report.append("**原因分析**:")
            report.append("- OCR提取的文本远小于GT，说明文本被过滤或未被识别")
            report.append("- 可能是PP-StructureV3将文本误识别为表格/图片")
            report.append("- bbox隔离可能过于激进，过滤了真实文本")
            report.append("")
            report.append("**改进方向**:")
            report.append("1. 调整PP-StructureV3检测参数，减少误检")
            report.append("2. 降低bbox隔离的激进程度: overlap_threshold 0.2 → 0.3")
            report.append("3. 检查parsing_res_list中text块的标签是否准确")
            report.append("")

        else:
            report.append("⚖️ **问题均衡: Overflow ≈ Underflow**")
            report.append("")
            report.append("**分析**: 两类问题都存在，需要分页面策略")
            report.append("- 对Overflow页面: 加强bbox隔离")
            report.append("- 对Underflow页面: 降低检测阈值")
            report.append("")

        return "\n".join(report)

    def save_report(self, report: str, output_path: str = "results/diagnosis_report.md"):
        """保存诊断报告"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Diagnosis report saved to {output_path}")


if __name__ == "__main__":
    diagnoser = FailedPageDiagnoser()

    # 分类页面
    classified = diagnoser.classify_pages(
        overflow_threshold=1.5,  # OCR > GT 1.5倍以上
        underflow_threshold=0.5   # OCR < GT 0.5倍以下
    )

    # 生成报告
    report = diagnoser.generate_report(classified)
    print(report)

    # 保存报告
    diagnoser.save_report(report)
