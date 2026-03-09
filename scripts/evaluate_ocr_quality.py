"""
完整的OCR质量评估脚本
比对OCR提取的文本、表格、公式 vs OmniDocBench.json中的标准内容
计算指标：
  - 文本：字符准确率(CER)、序列相似度
  - 表格：TEDS评分、结构相似度、内容相似度
  - 公式：LaTeX准确率、符号匹配

使用方法：
    python scripts/evaluate_ocr_quality.py \
        [--num-samples 100] \
        [--include-tables] \
        [--include-formulas]
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
import argparse

from tqdm import tqdm

from src.utils import load_config
from src.evaluator import MetricsCalculator, TableEvaluator, FormulaEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# Load config
CONFIG = load_config()
PATHS = CONFIG.get("paths", {})
OCR_STRUCTURED_DIR = PATHS.get("ocr_structured_dir", "results/ocr_structured")
NORMALIZED_DATA_PATH = PATHS.get("normalized_data_path", "data/answers/normalized_data.json")
OUTPUT_CSV = "results/ocr_quality_evaluation_full.csv"
OUTPUT_MD = "results/ocr_quality_report.md"


class OCRQualityEvaluator:
    """OCR质量评估器（包括文本、表格、公式）"""

    def __init__(self, include_tables: bool = True, include_formulas: bool = True):
        self.ocr_structured_dir = Path(OCR_STRUCTURED_DIR)
        self.normalized_data_path = Path(NORMALIZED_DATA_PATH)
        self.include_tables = include_tables
        self.include_formulas = include_formulas

        # 加载数据
        self.ocr_data = self._load_ocr_data()
        self.gt_data = self._load_ground_truth()

    def _load_ocr_data(self) -> dict:
        """加载OCR结构化输出，按文件和页码索引"""
        ocr_data = {}  # {filename: {page_index: data}}

        if not self.ocr_structured_dir.exists():
            logger.warning("OCR structured dir not found: %s", self.ocr_structured_dir)
            return ocr_data

        for json_file in self.ocr_structured_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # OCR输出是列表，包含多个页面的结构化数据
                if isinstance(data, list):
                    for page_item in data:
                        filename = page_item.get("filename", "")
                        page_index = page_item.get("page_index", 0)
                        stem = Path(filename).stem if filename else ""

                        if stem not in ocr_data:
                            ocr_data[stem] = {}
                        ocr_data[stem][page_index] = page_item
            except Exception as e:
                logger.warning(f"Failed to load OCR data from {json_file}: {e}")

        logger.info(f"Loaded OCR data: {len(ocr_data)} files")
        return ocr_data

    def _load_ground_truth(self) -> dict:
        """加载标准文本（OmniDocBench）"""
        gt_data = {}  # {file_stem: {page_no: page_dets}}

        if self.normalized_data_path.exists():
            try:
                with open(self.normalized_data_path, 'r', encoding='utf-8') as f:
                    normalized_data = json.load(f)

                # normalized_data是列表，每个元素代表一页
                for page_item in normalized_data:
                    page_no = page_item.get("page_no", 0)
                    image_path = page_item.get("image_path", "")
                    stem = Path(image_path).stem if image_path else ""

                    if stem not in gt_data:
                        gt_data[stem] = {}
                    gt_data[stem][page_no] = page_item

                logger.info(f"Loaded ground truth: {len(gt_data)} files from normalized_data.json")
                return gt_data
            except Exception as e:
                logger.warning(f"Failed to load normalized_data: {e}")

        return gt_data

    @staticmethod
    def _bbox_overlap(bbox1: list, bbox2: list, overlap_threshold: float = 0.2) -> bool:
        """
        检测两个bbox是否重叠

        Args:
            bbox1: [x1, y1, x2, y2] 第一个边界框
            bbox2: [x1, y1, x2, y2] 第二个边界框
            overlap_threshold: 重叠面积占较小面积的比例阈值

        Returns:
            bool: 是否重叠
        """
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return False

        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]

        # 计算交集区域
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        # 如果没有交集
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return False

        # 计算交集面积和两个bbox的面积
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # 计算重叠率（相对于较小的面积）
        min_area = min(area1, area2)
        if min_area <= 0:
            return False

        overlap_ratio = inter_area / min_area
        return overlap_ratio >= overlap_threshold

    def _extract_ocr_text(self, ocr_page_data: dict, use_bbox_isolation: bool = True) -> str:
        """
        从OCR页面数据提取纯文本
        只提取 block_label == "text" 的内容，过滤掉图片、表格等

        Args:
            ocr_page_data: OCR输出页面数据
            use_bbox_isolation: 是否使用bbox隔离来排除与表格重叠的文本块

        Returns:
            提取的纯文本
        """
        text_parts = []
        core_content = ocr_page_data.get("core_content", {})
        parsing_res_list = core_content.get("parsing_res_list", [])

        # 第一步：收集所有表格的bbox
        table_bboxes = []
        if use_bbox_isolation:
            for res in parsing_res_list:
                if res.get("block_label") == "table":
                    block_bbox = res.get("block_bbox")
                    if block_bbox:
                        table_bboxes.append(block_bbox)

        # 第二步：遍历所有块，提取文本但排除表格重叠区域
        # 注：overlap_threshold=0.05 - 只在明显重叠时过滤，避免过度过滤真实文本
        for res in parsing_res_list:
            block_label = res.get("block_label", "")
            block_content = res.get("block_content", "")
            block_bbox = res.get("block_bbox")

            # 只提取文本块和标题，过滤图片
            if block_label in ["text", "paragraph", "title", "list_item", "paragraph_title"]:
                # 使用bbox隔离：检查是否与表格明显重叠（阈值降低到0.05）
                if use_bbox_isolation and block_bbox and table_bboxes:
                    overlaps_with_table = any(
                        self._bbox_overlap(block_bbox, table_bbox, overlap_threshold=0.05)
                        for table_bbox in table_bboxes
                    )
                    if overlaps_with_table:
                        logger.debug(f"文本块 {block_label} 与表格明显重叠，已排除")
                        continue

                if block_content:
                    text_parts.append(block_content)

            # 对于其他标签，如果识别为文本，记录标签类型但不包含图片OCR结果
            elif block_label in ["doc_title", "header", "footer", "page_num"]:
                # 同样应用bbox隔离（更宽松的阈值）
                if use_bbox_isolation and block_bbox and table_bboxes:
                    overlaps_with_table = any(
                        self._bbox_overlap(block_bbox, table_bbox, overlap_threshold=0.05)
                        for table_bbox in table_bboxes
                    )
                    if overlaps_with_table:
                        continue

                if block_content:
                    text_parts.append(block_content)

            # 完全过滤掉图片、表格等二进制内容
            elif block_label in ["image", "figure", "table"]:
                continue

        return "\n".join(text_parts).strip()

    def _extract_ocr_tables(self, ocr_page_data: dict) -> List[Dict]:
        """
        从OCR页面数据提取表格
        支持两种格式：direct tables或从parsing_res_list中提取
        """
        core_content = ocr_page_data.get("core_content", {})

        # 方法1：直接提取tables字段
        tables = core_content.get("tables", [])
        if tables:
            return tables

        # 方法2：从parsing_res_list中提取标记为table的块
        # 此处将table块转换为统一格式
        parsing_res_list = core_content.get("parsing_res_list", [])
        extracted_tables = []
        for res in parsing_res_list:
            if res.get("block_label") == "table":
                block_content = res.get("block_content", "")
                if block_content:
                    extracted_tables.append({"html": block_content})

        return extracted_tables if extracted_tables else []

    def _tables_to_html(self, tables: List[Dict]) -> str:
        """
        将表格列表转换为HTML格式
        支持多种表格格式
        """
        if not tables:
            return ""

        html_parts = []
        for table in tables:
            if isinstance(table, dict):
                # 如果已有html字段，直接使用
                if "html" in table:
                    html_parts.append(table["html"])
                # 如果有rows字段，转换为HTML
                elif "rows" in table:
                    rows = table.get("rows", [])
                    if rows:
                        html_parts.append("<table>")
                        for row in rows:
                            html_parts.append("<tr>")
                            for cell in row:
                                html_parts.append(f"<td>{cell}</td>")
                            html_parts.append("</tr>")
                        html_parts.append("</table>")

        return "".join(html_parts)

    def _extract_ocr_formulas(self, ocr_page_data: dict) -> List[str]:
        """从OCR页面数据提取公式"""
        core_content = ocr_page_data.get("core_content", {})
        formulas = core_content.get("formulas", [])

        # 提取公式文本
        formula_texts = []
        for formula in formulas:
            if isinstance(formula, dict):
                formula_texts.append(formula.get("content", ""))
            else:
                formula_texts.append(str(formula))

        return formula_texts

    def _extract_gt_text(self, gt_page_data: dict) -> str:
        """
        从GT页面数据提取纯文本
        只提取文本、标题等内容，过滤掉图片、表格等
        """
        text_parts = []
        page_dets = gt_page_data.get("page_dets", [])

        for det in page_dets:
            category_type = det.get("category_type", "")
            text = det.get("text", "")

            # 只提取文本、标题等内容
            if category_type in ["text_block", "title", "heading", "list_item", "paragraph"]:
                if text:
                    text_parts.append(text)
            elif category_type in ["figure_caption", "page_header", "page_footer"]:
                if text:
                    text_parts.append(text)
            # 完全过滤掉图片、表格、公式等
            elif category_type in ["figure", "table", "formula"]:
                continue

        return "\n".join(text_parts).strip()

    def _extract_gt_tables(self, gt_page_data: dict) -> List[Dict]:
        """从GT页面数据提取表格"""
        page_dets = gt_page_data.get("page_dets", [])
        tables = []

        for det in page_dets:
            if det.get("category_type") == "table":
                table_html = det.get("table_html", "") or det.get("content", "")
                if table_html:
                    tables.append({"html": table_html})

        return tables

    def _extract_gt_formulas(self, gt_page_data: dict) -> List[str]:
        """从GT页面数据提取公式"""
        page_dets = gt_page_data.get("page_dets", [])
        formulas = []

        for det in page_dets:
            if det.get("category_type") == "formula":
                formula_text = det.get("text", "") or det.get("content", "")
                if formula_text:
                    formulas.append(formula_text)

        return formulas

    def evaluate_batch(self, num_samples: Optional[int] = None) -> None:
        """
        批量评估OCR质量

        Args:
            num_samples: 限制评估的样本数；None表示全部评估
        """
        results = []

        # 获取可评估的文件列表
        filenames = set(self.ocr_data.keys()) & set(self.gt_data.keys())
        filenames = sorted(list(filenames))

        if num_samples:
            filenames = filenames[:num_samples]

        logger.info(f"Evaluating {len(filenames)} files")

        for filename in tqdm(filenames, desc="OCR Quality Evaluation"):
            ocr_pages = self.ocr_data[filename]
            gt_pages = self.gt_data[filename]

            # 获取共同的页码
            page_indices = sorted(set(ocr_pages.keys()) & set(gt_pages.keys()))

            for page_idx in page_indices:
                # 文本评估
                ocr_text = self._extract_ocr_text(ocr_pages[page_idx])
                gt_text = self._extract_gt_text(gt_pages[page_idx])

                char_acc = MetricsCalculator.character_accuracy(ocr_text, gt_text)
                char_err = MetricsCalculator.character_error_rate(ocr_text, gt_text)
                seq_sim = MetricsCalculator.sequence_similarity(ocr_text, gt_text)
                jaccard_sim = MetricsCalculator.jaccard_similarity(ocr_text, gt_text)

                result = {
                    "filename": filename,
                    "page_index": page_idx,
                    "text_length_ocr": len(ocr_text),
                    "text_length_gt": len(gt_text),
                    "character_accuracy": round(char_acc, 4),
                    "character_error_rate": round(char_err, 4),
                    "sequence_similarity": round(seq_sim, 4),
                    "jaccard_similarity": round(jaccard_sim, 4),
                }

                # 表格评估
                if self.include_tables:
                    ocr_tables = self._extract_ocr_tables(ocr_pages[page_idx])
                    gt_tables = self._extract_gt_tables(gt_pages[page_idx])

                    # 总是计算表格计数
                    result["table_count_pred"] = len(ocr_tables)
                    result["table_count_ref"] = len(gt_tables)

                    if ocr_tables and gt_tables:
                        # 构建HTML格式用于评估
                        ocr_html = self._tables_to_html(ocr_tables)
                        gt_html = self._tables_to_html(gt_tables)

                        if ocr_html and gt_html:
                            table_eval = TableEvaluator.evaluate_tables(ocr_html, gt_html)
                            result.update({
                                "table_teds": table_eval.get("teds_score", 0.0),
                                "table_structure_similarity": table_eval.get("structure_similarity", 0.0),
                                "table_content_similarity": table_eval.get("content_similarity", 0.0),
                                "table_cell_accuracy": table_eval.get("cell_accuracy", 0.0),
                            })
                        else:
                            # 如果HTML构建失败，使用count-based评分
                            count_match = 1.0 if result["table_count_pred"] == result["table_count_ref"] else 0.0
                            result.update({
                                "table_teds": count_match * 100,
                                "table_structure_similarity": count_match,
                                "table_content_similarity": count_match,
                                "table_cell_accuracy": count_match,
                            })
                    elif ocr_tables and not gt_tables:
                        # OCR识别到表格但GT没有 - 过识别
                        result.update({
                            "table_teds": 0.0,
                            "table_structure_similarity": 0.0,
                            "table_content_similarity": 0.0,
                            "table_cell_accuracy": 0.0,
                        })
                    elif not ocr_tables and gt_tables:
                        # GT有表格但OCR没识别到 - 漏识别
                        result.update({
                            "table_teds": 0.0,
                            "table_structure_similarity": 0.0,
                            "table_content_similarity": 0.0,
                            "table_cell_accuracy": 0.0,
                        })
                    else:
                        # 两者都没有表格
                        result.update({
                            "table_teds": None,
                            "table_structure_similarity": None,
                            "table_content_similarity": None,
                            "table_cell_accuracy": None,
                        })

                # 公式评估
                if self.include_formulas:
                    ocr_formulas = self._extract_ocr_formulas(ocr_pages[page_idx])
                    gt_formulas = self._extract_gt_formulas(gt_pages[page_idx])

                    if ocr_formulas and gt_formulas:
                        formula_eval = FormulaEvaluator.evaluate_formulas(
                            ocr_formulas,
                            gt_formulas,
                            formula_type="latex"
                        )
                        result.update({
                            "formula_count": formula_eval.get("total_formulas", 0),
                            "formula_average_accuracy": round(formula_eval.get("average_accuracy", 0.0), 4),
                            "formula_correct_count": formula_eval.get("correct_formulas", 0),
                        })
                    else:
                        result.update({
                            "formula_count": None,
                            "formula_average_accuracy": None,
                            "formula_correct_count": None,
                        })

                # 综合评分（动态权重，改进版）
                # 判断页面类型：是否为纯文本/纯表格/混合/空白
                gt_text_len = int(result.get("text_length_gt", 0))
                gt_table_count = int(result.get("table_count_ref", 0))

                # 基础：文本评估（最重要，权重1.0）
                # 但如果GT中没有文本内容，降低文本权重（因为期望本就为0）
                if gt_text_len == 0 and gt_table_count == 0:
                    # 纯图片或空白页：不应该有文本，降低文本权重
                    component_weights = {"text": 0.2}
                else:
                    # 正常页面：文本权重正常
                    component_weights = {"text": 1.0}

                score_components = {"text": char_acc}

                # 表格评估（如果有）
                if self.include_tables and "table_teds" in result and result["table_teds"] is not None:
                    # TEDS已经是0-100的分数，转换为0-1
                    table_score = result["table_teds"] / 100.0
                    score_components["table"] = table_score
                    component_weights["table"] = 0.5

                # 公式评估（如果有）
                if self.include_formulas and "formula_average_accuracy" in result and result["formula_average_accuracy"] is not None:
                    score_components["formula"] = result["formula_average_accuracy"]
                    component_weights["formula"] = 0.3

                # 计算加权平均
                if score_components:
                    total_weight = sum(component_weights.values())
                    weighted_sum = sum(score * component_weights[key] for key, score in score_components.items())
                    composite = (weighted_sum / total_weight) * 100 if total_weight > 0 else 0.0
                else:
                    composite = 0.0

                result["composite_score"] = round(composite, 2)

                results.append(result)

        # 保存结果
        self._save_results(results)

        # 保存Markdown报告
        self._save_markdown_report(results, OUTPUT_MD)

        # 打印统计信息
        self._print_statistics(results)

    def _save_results(self, results: list) -> None:
        """保存评估结果为CSV"""
        output_path = Path(OUTPUT_CSV)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not results:
            logger.warning("No results to save")
            return

        fieldnames = results[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info("Results saved to %s", output_path)

    def _save_markdown_report(self, results: list, path: str) -> None:
        """
        保存OCR评估结果的Markdown报告。

        Args:
            results: 每一页的评估结果列表。
            path: Markdown报告输出路径。
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not results:
            logger.warning("No results to generate markdown report")
            output_path.write_text("# OCR Quality Evaluation Report\n\nNo results.\n", encoding="utf-8")
            return

        char_accs = [r["character_accuracy"] for r in results]
        seq_sims = [r["sequence_similarity"] for r in results]
        jaccard_sims = [r["jaccard_similarity"] for r in results]
        composite_scores = [
            r.get("composite_score", 0.0) for r in results if r.get("composite_score") is not None
        ]

        table_results = [r for r in results if r.get("table_teds") is not None]
        if table_results:
            teds_scores = [
                r["table_teds"] for r in table_results if r.get("table_teds") is not None
            ]
            struct_sims = [
                r["table_structure_similarity"]
                for r in table_results
                if r.get("table_structure_similarity") is not None
            ]
            content_sims = [
                r["table_content_similarity"]
                for r in table_results
                if r.get("table_content_similarity") is not None
            ]
        else:
            teds_scores = []
            struct_sims = []
            content_sims = []

        formula_results = [r for r in results if r.get("formula_average_accuracy") is not None]
        if formula_results:
            formula_accs = [
                r["formula_average_accuracy"]
                for r in formula_results
                if r.get("formula_average_accuracy") is not None
            ]
            correct_counts = [
                r["formula_correct_count"]
                for r in formula_results
                if r.get("formula_correct_count") is not None
            ]
            total_formulas = sum(
                [
                    r["formula_count"]
                    for r in formula_results
                    if r.get("formula_count") is not None
                ]
            )
        else:
            formula_accs = []
            correct_counts = []
            total_formulas = 0

        lines = []

        # 总览
        lines.append("# OCR Quality Evaluation Report")
        lines.append("")
        lines.append(
            f"- Generated at: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(f"- Total evaluated pages: {len(results)}")
        lines.append("")

        # 文本质量
        lines.append("## Text Quality Metrics")
        lines.append("")
        lines.append("| Metric | Mean |")
        lines.append("| --- | --- |")
        lines.append(f"| Character Accuracy | {sum(char_accs) / len(char_accs):.4f} |")
        lines.append(f"| Sequence Similarity | {sum(seq_sims) / len(seq_sims):.4f} |")
        lines.append(f"| Jaccard Similarity | {sum(jaccard_sims) / len(jaccard_sims):.4f} |")
        lines.append("")

        # 表格质量
        if table_results and teds_scores:
            lines.append("## Table Quality Metrics")
            lines.append("")
            lines.append(f"- Pages with tables: **{len(table_results)}**")
            lines.append("")
            lines.append("| Metric | Mean |")
            lines.append("| --- | --- |")
            lines.append(f"| TEDS Score (0-100) | {sum(teds_scores) / len(teds_scores):.2f} |")
            if struct_sims:
                lines.append(
                    f"| Structure Similarity | {sum(struct_sims) / len(struct_sims):.4f} |"
                )
            if content_sims:
                lines.append(
                    f"| Content Similarity | {sum(content_sims) / len(content_sims):.4f} |"
                )
            lines.append("")

        # 公式质量
        if formula_results and formula_accs and total_formulas > 0:
            lines.append("## Formula Quality Metrics")
            lines.append("")
            lines.append(f"- Pages with formulas: **{len(formula_results)}**")
            lines.append(f"- Total formulas: **{total_formulas}**")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            lines.append(
                f"| Average Accuracy | {sum(formula_accs) / len(formula_accs):.4f} |"
            )
            lines.append(
                f"| Correct Formulas | {sum(correct_counts)}/{total_formulas} |"
            )
            lines.append(
                f"| Accuracy Rate (%) | {sum(correct_counts) / total_formulas * 100:.1f} |"
            )
            lines.append("")

        # 综合评分
        if composite_scores:
            mean_score = sum(composite_scores) / len(composite_scores)
            median_score = sorted(composite_scores)[len(composite_scores) // 2]
            std_score = (
                sum((s - mean_score) ** 2 for s in composite_scores) / len(composite_scores)
            ) ** 0.5

            lines.append("## Composite Score Distribution (0-100)")
            lines.append("")
            lines.append("| Stat | Value |")
            lines.append("| --- | --- |")
            lines.append(f"| Mean | {mean_score:.2f} |")
            lines.append(f"| Median | {median_score:.2f} |")
            lines.append(f"| Std Dev | {std_score:.2f} |")
            lines.append(f"| Min | {min(composite_scores):.2f} |")
            lines.append(f"| Max | {max(composite_scores):.2f} |")
            lines.append("")

            excellent = sum(1 for s in composite_scores if s >= 90)
            good = sum(1 for s in composite_scores if 80 <= s < 90)
            fair = sum(1 for s in composite_scores if 70 <= s < 80)
            poor = sum(1 for s in composite_scores if s < 70)

            lines.append("| Grade | Pages | Ratio |")
            lines.append("| --- | --- | --- |")
            lines.append(
                f"| Excellent (≥90) | {excellent} | {excellent / len(composite_scores) * 100:5.1f}% |"
            )
            lines.append(
                f"| Good (80-90) | {good} | {good / len(composite_scores) * 100:5.1f}% |"
            )
            lines.append(
                f"| Fair (70-80) | {fair} | {fair / len(composite_scores) * 100:5.1f}% |"
            )
            lines.append(
                f"| Poor (<70) | {poor} | {poor / len(composite_scores) * 100:5.1f}% |"
            )
            lines.append("")

        # 最差 / 最好页面列表
        worst = sorted(results, key=lambda r: r["character_accuracy"])[:10]
        best = sorted(results, key=lambda r: r["character_accuracy"], reverse=True)[:10]

        lines.append("## Worst 10 Pages by Character Accuracy")
        lines.append("")
        lines.append("| filename | page | char_acc | text_len_ocr | text_len_gt |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in worst:
            lines.append(
                f"| {r['filename']} | {r['page_index']} | {r['character_accuracy']:.4f} | "
                f"{r['text_length_ocr']} | {r['text_length_gt']} |"
            )
        lines.append("")

        lines.append("## Best 10 Pages by Character Accuracy")
        lines.append("")
        lines.append("| filename | page | char_acc | text_len_ocr | text_len_gt |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in best:
            lines.append(
                f"| {r['filename']} | {r['page_index']} | {r['character_accuracy']:.4f} | "
                f"{r['text_length_ocr']} | {r['text_length_gt']} |"
            )
        lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown report saved to %s", output_path)

    def _print_statistics(self, results: list) -> None:
        """打印详细统计信息"""
        if not results:
            logger.warning("No results to print")
            return

        char_accs = [r["character_accuracy"] for r in results]
        seq_sims = [r["sequence_similarity"] for r in results]
        jaccard_sims = [r["jaccard_similarity"] for r in results]
        composite_scores = [r.get("composite_score", 0) for r in results if r.get("composite_score")]
        # 为后续诊断建议准备默认值，避免未定义变量
        teds_scores = []
        formula_accs = []

        print("\n" + "="*80)
        print("OCR QUALITY EVALUATION - COMPLETE REPORT".center(80))
        print("="*80)
        print(f"\n📊 Evaluation Summary:")
        print(f"  Total evaluated pages: {len(results)}")
        print(f"  Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 文本质量
        print(f"\n📝 TEXT QUALITY METRICS:")
        print(f"  ├─ Character Accuracy:    {sum(char_accs)/len(char_accs):.4f} (Mean)")
        print(f"  ├─ Sequence Similarity:   {sum(seq_sims)/len(seq_sims):.4f} (Mean)")
        print(f"  └─ Jaccard Similarity:    {sum(jaccard_sims)/len(jaccard_sims):.4f} (Mean)")

        # 表格质量
        table_results = [r for r in results if r.get("table_teds") is not None]
        if table_results:
            teds_scores = [r["table_teds"] for r in table_results if r.get("table_teds") is not None]
            struct_sims = [r["table_structure_similarity"] for r in table_results if r.get("table_structure_similarity") is not None]
            content_sims = [r["table_content_similarity"] for r in table_results if r.get("table_content_similarity") is not None]

            print(f"\n📋 TABLE QUALITY METRICS ({len(table_results)} pages with tables):")
            print(f"  ├─ TEDS Score:                {sum(teds_scores)/len(teds_scores):.2f} (Mean, 0-100)")
            print(f"  ├─ Structure Similarity:      {sum(struct_sims)/len(struct_sims):.4f} (Mean)" if struct_sims else "  ├─ Structure Similarity:      N/A (no data)")
            print(f"  └─ Content Similarity:        {sum(content_sims)/len(content_sims):.4f} (Mean)" if content_sims else "  └─ Content Similarity:        N/A (no data)")

            # 表格统计
            table_count_mismatch = sum(1 for r in table_results if r.get("table_count_pred") != r.get("table_count_ref"))
            print(f"\n  Table Count Statistics:")
            print(f"  ├─ Perfect matches: {len(table_results) - table_count_mismatch} pages")
            print(f"  └─ Mismatches: {table_count_mismatch} pages")

        # 公式质量
        formula_results = [r for r in results if r.get("formula_average_accuracy") is not None]
        if formula_results:
            formula_accs = [r["formula_average_accuracy"] for r in formula_results if r.get("formula_average_accuracy") is not None]
            correct_counts = [r["formula_correct_count"] for r in formula_results if r.get("formula_correct_count") is not None]
            total_formulas = sum([r["formula_count"] for r in formula_results if r.get("formula_count") is not None])

            print(f"\n📐 FORMULA QUALITY METRICS ({len(formula_results)} pages with formulas):")
            print(f"  ├─ Average Accuracy:      {sum(formula_accs)/len(formula_accs):.4f} (Mean)")
            print(f"  ├─ Correct Formulas:      {sum(correct_counts)}/{total_formulas}")
            print(f"  └─ Accuracy Rate:         {sum(correct_counts)/total_formulas*100:.1f}%")

        # 综合评分
        print(f"\n⭐ COMPOSITE SCORE (0-100):")
        if composite_scores:
            mean_score = sum(composite_scores) / len(composite_scores)
            median_score = sorted(composite_scores)[len(composite_scores)//2]

            print(f"  ├─ Mean:                  {mean_score:.2f}")
            print(f"  ├─ Median:                {median_score:.2f}")
            print(f"  ├─ Std Dev:               {(sum((s - mean_score)**2 for s in composite_scores) / len(composite_scores))**0.5:.2f}")
            print(f"  ├─ Min:                   {min(composite_scores):.2f}")
            print(f"  └─ Max:                   {max(composite_scores):.2f}")

            # 分级统计
            excellent = sum(1 for s in composite_scores if s >= 90)
            good = sum(1 for s in composite_scores if 80 <= s < 90)
            fair = sum(1 for s in composite_scores if 70 <= s < 80)
            poor = sum(1 for s in composite_scores if s < 70)

            print(f"\n  Grade Distribution:")
            print(f"  ├─ Excellent (≥90):  {excellent:4d} pages ({excellent/len(composite_scores)*100:5.1f}%) ████████████")
            print(f"  ├─ Good (80-90):     {good:4d} pages ({good/len(composite_scores)*100:5.1f}%) ████████")
            print(f"  ├─ Fair (70-80):     {fair:4d} pages ({fair/len(composite_scores)*100:5.1f}%) ████")
            print(f"  └─ Poor (<70):       {poor:4d} pages ({poor/len(composite_scores)*100:5.1f}%) ██")

        # 诊断建议
        print(f"\n💡 DIAGNOSTIC RECOMMENDATIONS:")
        if char_accs and len(char_accs) > 0 and sum(char_accs)/len(char_accs) < 0.80:
            print(f"  ⚠️  Text accuracy < 80%: Check OCR confidence_threshold or image quality")
        if teds_scores and len(teds_scores) > 0 and sum(teds_scores)/len(teds_scores) < 70:
            print(f"  ⚠️  Table TEDS < 70: Table structure or content extraction needs improvement")
        if formula_accs and len(formula_accs) > 0 and sum(formula_accs)/len(formula_accs) < 0.85:
            print(f"  ⚠️  Formula accuracy < 85%: LaTeX parsing or OCR formula recognition needs work")

        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR quality against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all files with all metrics
  python scripts/evaluate_ocr_quality.py

  # Evaluate first 100 files, text only
  python scripts/evaluate_ocr_quality.py --num-samples 100

  # Evaluate with tables and formulas
  python scripts/evaluate_ocr_quality.py --include-tables --include-formulas
        """
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (None = all)"
    )
    parser.add_argument(
        "--include-tables",
        action="store_true",
        default=True,
        help="Include table quality evaluation"
    )
    parser.add_argument(
        "--include-formulas",
        action="store_true",
        default=True,
        help="Include formula quality evaluation"
    )
    args = parser.parse_args()

    evaluator = OCRQualityEvaluator(
        include_tables=args.include_tables,
        include_formulas=args.include_formulas
    )
    evaluator.evaluate_batch(num_samples=args.num_samples)


if __name__ == "__main__":
    main()
