"""
低分页面分析脚本
对 character_accuracy < 0.2 的页面进行失败模式分类：
  - 过度识别：OCR 抽取文字远多于 GT
  - 漏识别：OCR 文字远少于 GT
  - 真实错误：数量接近但字符识别错误

用法：
    python scripts/analyze_low_score_pages.py [--threshold 0.2] [--output results/low_score_analysis.md]
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict


LOW_SCORE_THRESHOLD = 0.2
OVER_DETECT_RATIO = 2.0   # OCR > GT * 2.0 → 过度识别
UNDER_DETECT_RATIO = 0.5  # OCR < GT * 0.5 → 漏识别


def classify_page(ocr_len: int, gt_len: int, char_acc: float) -> str:
    """对单页进行失败模式分类"""
    # 两者都为空：空白页（不应出现在低分列表，但做保护）
    if ocr_len == 0 and gt_len == 0:
        return "empty"

    # OCR 没有识别到任何文字，但 GT 有内容
    if ocr_len == 0 and gt_len > 0:
        return "under_detect"

    # OCR 识别到文字，但 GT 为空（纯图片/装饰页被过度识别）
    if ocr_len > 0 and gt_len == 0:
        return "over_detect"

    ratio = ocr_len / gt_len

    if ratio >= OVER_DETECT_RATIO:
        return "over_detect"
    elif ratio <= UNDER_DETECT_RATIO:
        return "under_detect"
    else:
        return "true_error"


def infer_doc_type(filename: str) -> str:
    """从文件名推断文档类型"""
    fn = filename.lower()
    if fn.startswith("ppt_"):
        return "PPT"
    elif "textbook" in fn or "教材" in fn or "课本" in fn:
        return "教材"
    elif "paper" in fn or "academic" in fn:
        return "学术论文"
    elif "book" in fn:
        return "书籍"
    elif "dianzishu" in fn or "电子书" in fn:
        return "电子书"
    else:
        return "其他"


def analyze(csv_path: str, threshold: float, output_path: str) -> None:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # 过滤低分页面（排除双空页）
    low_score = []
    for r in rows:
        ocr_len = int(r.get("text_length_ocr", 0) or 0)
        gt_len  = int(r.get("text_length_gt", 0) or 0)
        char_acc = float(r.get("character_accuracy", 0) or 0)
        if ocr_len == 0 and gt_len == 0:
            continue
        if char_acc < threshold:
            low_score.append(r)

    total_pages = sum(
        1 for r in rows
        if not (int(r.get("text_length_ocr", 0) or 0) == 0
                and int(r.get("text_length_gt", 0) or 0) == 0)
    )

    # 分类
    categories = defaultdict(list)
    for r in low_score:
        ocr_len  = int(r.get("text_length_ocr", 0) or 0)
        gt_len   = int(r.get("text_length_gt", 0) or 0)
        char_acc = float(r.get("character_accuracy", 0) or 0)
        cat = classify_page(ocr_len, gt_len, char_acc)
        categories[cat].append(r)

    # 按文档类型统计
    doc_type_counts = defaultdict(lambda: defaultdict(int))
    for cat, pages in categories.items():
        for r in pages:
            doc_type = infer_doc_type(r["filename"])
            doc_type_counts[cat][doc_type] += 1

    # 生成报告
    lines = []
    lines.append("# Low Score Page Analysis Report")
    lines.append("")
    lines.append(f"- Threshold: character_accuracy < {threshold}")
    lines.append(f"- Total content pages: {total_pages}")
    lines.append(f"- Low score pages: **{len(low_score)}** ({len(low_score)/total_pages*100:.1f}%)")
    lines.append("")

    # 分类汇总
    lines.append("## Failure Mode Distribution")
    lines.append("")
    lines.append("| 失败模式 | 页数 | 占低分页 | 描述 |")
    lines.append("|---|---|---|---|")

    mode_meta = {
        "over_detect":  ("过度识别", f"OCR 文字量 > GT 的 {OVER_DETECT_RATIO}x", "⚠️ 提高 text_det_thresh 或过滤低置信度 block"),
        "under_detect": ("漏识别",   f"OCR 文字量 < GT 的 {UNDER_DETECT_RATIO}x", "⚠️ 布局检测把文字区域判成 image，或降低 text_det_thresh"),
        "true_error":   ("字符错误",  "数量接近但识别字符错误", "⚠️ 图像质量差或复杂字体，考虑预处理或换模型"),
    }

    for cat in ["over_detect", "under_detect", "true_error"]:
        count = len(categories[cat])
        pct = count / len(low_score) * 100 if low_score else 0
        name, desc, _ = mode_meta[cat]
        lines.append(f"| {name} | {count} | {pct:.1f}% | {desc} |")
    lines.append("")

    # 每种失败模式的优化建议
    lines.append("## Optimization Suggestions")
    lines.append("")
    for cat in ["over_detect", "under_detect", "true_error"]:
        count = len(categories[cat])
        if count == 0:
            continue
        name, _, suggestion = mode_meta[cat]
        lines.append(f"### {name}（{count} 页）")
        lines.append("")
        lines.append(f"{suggestion}")
        lines.append("")

        # 文档类型分布
        dt = doc_type_counts[cat]
        if dt:
            lines.append("**文档类型分布：**")
            lines.append("")
            lines.append("| 类型 | 页数 |")
            lines.append("|---|---|")
            for doc_type, cnt in sorted(dt.items(), key=lambda x: -x[1]):
                lines.append(f"| {doc_type} | {cnt} |")
            lines.append("")

        # OCR/GT 长度统计
        ocr_lens = [int(r.get("text_length_ocr", 0) or 0) for r in categories[cat]]
        gt_lens  = [int(r.get("text_length_gt", 0) or 0) for r in categories[cat]]
        lines.append("**文字长度统计：**")
        lines.append("")
        lines.append("| 指标 | OCR | GT |")
        lines.append("|---|---|---|")
        lines.append(f"| 平均 | {sum(ocr_lens)/len(ocr_lens):.0f} | {sum(gt_lens)/len(gt_lens):.0f} |")
        lines.append(f"| 最大 | {max(ocr_lens)} | {max(gt_lens)} |")
        lines.append(f"| 最小 | {min(ocr_lens)} | {min(gt_lens)} |")
        lines.append("")

        # 最差样本（前10）
        worst = sorted(categories[cat],
                       key=lambda r: float(r.get("character_accuracy", 0) or 0))[:10]
        lines.append("**最差样本（前10）：**")
        lines.append("")
        lines.append("| filename | page | char_acc | ocr_len | gt_len |")
        lines.append("|---|---|---|---|---|")
        for r in worst:
            lines.append(
                f"| {r['filename']} | {r['page_index']} "
                f"| {float(r.get('character_accuracy',0) or 0):.4f} "
                f"| {r.get('text_length_ocr',0)} "
                f"| {r.get('text_length_gt',0)} |"
            )
        lines.append("")

    # 控制台打印
    print(f"\n{'='*60}")
    print(f"LOW SCORE PAGE ANALYSIS (threshold < {threshold})")
    print(f"{'='*60}")
    print(f"  Total content pages : {total_pages}")
    print(f"  Low score pages     : {len(low_score)} ({len(low_score)/total_pages*100:.1f}%)")
    print()
    for cat in ["over_detect", "under_detect", "true_error"]:
        count = len(categories[cat])
        pct = count / len(low_score) * 100 if low_score else 0
        name, _, _ = mode_meta[cat]
        print(f"  {name:8s}: {count:4d} 页 ({pct:5.1f}%)")
    print(f"{'='*60}\n")

    # 写入文件
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to: {out}")


def main():
    parser = argparse.ArgumentParser(description="Analyze low score OCR pages")
    parser.add_argument("--threshold", type=float, default=LOW_SCORE_THRESHOLD,
                        help=f"Character accuracy threshold (default {LOW_SCORE_THRESHOLD})")
    parser.add_argument("--csv", default="results/ocr_quality_evaluation_full.csv",
                        help="Path to evaluation CSV")
    parser.add_argument("--output", default="results/low_score_analysis.md",
                        help="Output markdown report path")
    args = parser.parse_args()

    analyze(args.csv, args.threshold, args.output)


if __name__ == "__main__":
    main()
