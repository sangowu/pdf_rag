"""
表格评估模块
实现TEDS (Table Edit Distance Score) 的简化版本
比对表格结构和内容的准确性
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from html.parser import HTMLParser
import editdistance

from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class HTMLTableParser(HTMLParser):
    """简单的HTML表格解析器"""

    def __init__(self):
        super().__init__()
        self.tables = []
        self.current_table = None
        self.current_row = None
        self.current_cell = None
        self.in_table = False
        self.in_row = False
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.in_table = True
            self.current_table = {'rows': []}
        elif tag in ['tr']:
            if self.in_table:
                self.in_row = True
                self.current_row = []
        elif tag in ['td', 'th']:
            if self.in_row:
                self.in_cell = True
                self.current_cell = ''

    def handle_endtag(self, tag):
        if tag == 'table' and self.in_table:
            self.in_table = False
            if self.current_table:
                self.tables.append(self.current_table)
                self.current_table = None
        elif tag == 'tr' and self.in_row:
            self.in_row = False
            if self.current_row and self.current_table:
                self.current_table['rows'].append(self.current_row)
                self.current_row = None
        elif tag in ['td', 'th'] and self.in_cell:
            self.in_cell = False
            if self.current_row is not None:
                self.current_row.append(self.current_cell.strip())
                self.current_cell = None

    def handle_data(self, data):
        if self.in_cell and self.current_cell is not None:
            self.current_cell += data

    def get_tables(self):
        return self.tables


class TableEvaluator:
    """表格评估器"""

    @staticmethod
    def parse_html_table(html_str: str) -> List[Dict]:
        """
        解析HTML表格

        Args:
            html_str: HTML字符串

        Returns:
            表格列表，每个表格为 {'rows': [['cell1', 'cell2', ...], ...]}
        """
        if not html_str:
            return []

        try:
            parser = HTMLTableParser()
            parser.feed(html_str)
            return parser.get_tables()
        except Exception as e:
            logger.warning(f"Failed to parse HTML table: {e}")
            return []

    @staticmethod
    def extract_table_structure(table: Dict) -> Tuple[int, int]:
        """
        提取表格结构（行数、列数）

        Returns:
            (num_rows, num_cols)
        """
        rows = table.get('rows', [])
        num_rows = len(rows)
        num_cols = max(len(row) for row in rows) if rows else 0
        return num_rows, num_cols

    @staticmethod
    def flatten_table(table: Dict) -> str:
        """
        展平表格为纯文本字符串（用于编辑距离计算）
        按行优先顺序拼接所有单元格
        """
        rows = table.get('rows', [])
        cells = []

        for row in rows:
            for cell in row:
                cells.append(str(cell))

        return ' '.join(cells)

    @staticmethod
    def structure_similarity(pred_table: Dict, ref_table: Dict) -> float:
        """
        表格结构相似度
        基于行列数的匹配

        Returns:
            相似度 (0-1)
        """
        pred_rows, pred_cols = TableEvaluator.extract_table_structure(pred_table)
        ref_rows, ref_cols = TableEvaluator.extract_table_structure(ref_table)

        if pred_rows == 0 and ref_rows == 0:
            return 1.0
        if pred_rows == 0 or ref_rows == 0:
            return 0.0

        # 行列都匹配才给最高分，否则降分
        row_match = 1.0 if pred_rows == ref_rows else abs(pred_rows - ref_rows) / max(pred_rows, ref_rows)
        col_match = 1.0 if pred_cols == ref_cols else abs(pred_cols - ref_cols) / max(pred_cols, ref_cols)

        # 结构相似度 = (行匹配 + 列匹配) / 2
        structure_sim = (2.0 - row_match - col_match) / 2.0
        return max(0.0, min(1.0, structure_sim))

    @staticmethod
    def content_similarity(pred_table: Dict, ref_table: Dict) -> float:
        """
        表格内容相似度
        基于所有单元格文本的相似度

        Returns:
            相似度 (0-1)
        """
        pred_text = TableEvaluator.flatten_table(pred_table)
        ref_text = TableEvaluator.flatten_table(ref_table)

        if not pred_text and not ref_text:
            return 1.0
        if not pred_text or not ref_text:
            return 0.0

        return MetricsCalculator.sequence_similarity(pred_text, ref_text)

    @staticmethod
    def teds_score(pred_table: Dict, ref_table: Dict, lambda_: float = 0.5) -> float:
        """
        计算 TEDS (Table Edit Distance Score) - 完整版本
        基于递归编辑距离，同时考虑HTML结构和文本内容

        TEDS = (1 - EditDistance(ref_html, pred_html) / max(len(ref_html), len(pred_html))) * 100

        Args:
            pred_table: 预测表格
            ref_table: 参考表格
            lambda_: 权重参数 (0-1)
                     = 0: 只考虑内容（单元格文本）
                     = 1: 只考虑结构（HTML标签）
                     = 0.5: 均衡考虑

        Returns:
            TEDS评分 (0-100)
        """
        if not ref_table.get('rows') and not pred_table.get('rows'):
            return 100.0

        # 方法1：基于HTML结构的编辑距离
        pred_html = TableEvaluator._table_to_html(pred_table)
        ref_html = TableEvaluator._table_to_html(ref_table)

        if pred_html and ref_html:
            dist = editdistance.eval(pred_html, ref_html)
            max_len = max(len(pred_html), len(ref_html))
            structure_score = (1.0 - dist / max_len) if max_len > 0 else 0.0
        else:
            structure_score = TableEvaluator.structure_similarity(pred_table, ref_table)

        # 方法2：基于单元格内容的准确率
        content_score = TableEvaluator.cell_accuracy(pred_table, ref_table)

        # 加权组合：结构和内容
        teds = (lambda_ * structure_score + (1 - lambda_) * content_score) * 100

        return round(max(0.0, min(100.0, teds)), 2)

    @staticmethod
    def _table_to_html(table: Dict) -> str:
        """将表格转换为HTML格式（用于编辑距离计算）"""
        rows = table.get('rows', [])
        if not rows:
            return ""

        html_parts = ['<table>']
        for row in rows:
            html_parts.append('<tr>')
            for cell in row:
                html_parts.append(f'<td>{cell}</td>')
            html_parts.append('</tr>')
        html_parts.append('</table>')

        return ''.join(html_parts)

    @staticmethod
    def cell_accuracy(pred_table: Dict, ref_table: Dict) -> float:
        """
        单元格准确率
        匹配单元格数量 / 总单元格数

        Returns:
            准确率 (0-1)
        """
        pred_rows = pred_table.get('rows', [])
        ref_rows = ref_table.get('rows', [])

        if not ref_rows:
            return 0.0

        # 假设两个表格都有相同的行数，则逐单元格比较
        correct_cells = 0
        total_cells = 0

        for i, ref_row in enumerate(ref_rows):
            if i < len(pred_rows):
                pred_row = pred_rows[i]
                for j, ref_cell in enumerate(ref_row):
                    total_cells += 1
                    if j < len(pred_row):
                        pred_cell = pred_row[j]
                        if str(pred_cell).strip() == str(ref_cell).strip():
                            correct_cells += 1
            else:
                total_cells += len(ref_row)

        return correct_cells / total_cells if total_cells > 0 else 0.0

    @staticmethod
    def evaluate_tables(pred_html: str, ref_html: str) -> Dict:
        """
        完整表格评估 - 支持多表格

        Args:
            pred_html: 预测的HTML表格
            ref_html: 参考的HTML表格

        Returns:
            包含各项指标的字典
        """
        pred_tables = TableEvaluator.parse_html_table(pred_html)
        ref_tables = TableEvaluator.parse_html_table(ref_html)

        if not pred_tables or not ref_tables:
            return {
                'teds_score': 0.0,
                'structure_similarity': 0.0,
                'content_similarity': 0.0,
                'cell_accuracy': 0.0,
                'table_count': len(pred_tables),
                'error': 'Failed to parse tables'
            }

        # 评估所有表格（逐个比对）
        teds_scores = []
        structure_sims = []
        content_sims = []
        cell_accs = []

        # 比对表格（最小匹配）
        num_tables = min(len(pred_tables), len(ref_tables))

        for i in range(num_tables):
            pred_table = pred_tables[i]
            ref_table = ref_tables[i]

            teds = TableEvaluator.teds_score(pred_table, ref_table)
            teds_scores.append(teds)

            structure_sim = TableEvaluator.structure_similarity(pred_table, ref_table)
            structure_sims.append(structure_sim)

            content_sim = TableEvaluator.content_similarity(pred_table, ref_table)
            content_sims.append(content_sim)

            cell_acc = TableEvaluator.cell_accuracy(pred_table, ref_table)
            cell_accs.append(cell_acc)

        # 计算平均值
        avg_teds = sum(teds_scores) / len(teds_scores) if teds_scores else 0.0
        avg_structure = sum(structure_sims) / len(structure_sims) if structure_sims else 0.0
        avg_content = sum(content_sims) / len(content_sims) if content_sims else 0.0
        avg_cell_acc = sum(cell_accs) / len(cell_accs) if cell_accs else 0.0

        # 表格数量匹配惩罚
        table_count_penalty = abs(len(pred_tables) - len(ref_tables)) / max(len(pred_tables), len(ref_tables))
        final_teds = avg_teds * (1 - table_count_penalty * 0.2)  # 表格数量错误降低20%权重

        return {
            'teds_score': round(final_teds, 2),
            'structure_similarity': round(avg_structure, 4),
            'content_similarity': round(avg_content, 4),
            'cell_accuracy': round(avg_cell_acc, 4),
            'table_count_pred': len(pred_tables),
            'table_count_ref': len(ref_tables),
            'table_count_matched': num_tables,
            'individual_teds_scores': [round(s, 2) for s in teds_scores],
        }

    @staticmethod
    def compare_table_blocks(pred_blocks: List[Dict], ref_blocks: List[Dict]) -> float:
        """
        比较表格块列表（当无HTML格式时）

        Args:
            pred_blocks: 预测的表格块列表，每个块为 {'content': '...'}
            ref_blocks: 参考的表格块列表

        Returns:
            平均TEDS评分
        """
        if not ref_blocks:
            return 0.0

        if not pred_blocks:
            return 0.0

        scores = []
        for i, ref_block in enumerate(ref_blocks[:len(pred_blocks)]):
            ref_text = ref_block.get('content', '')
            pred_text = pred_blocks[i].get('content', '') if i < len(pred_blocks) else ''

            # 简化：直接比较文本相似度
            sim = MetricsCalculator.sequence_similarity(pred_text, ref_text)
            scores.append(sim * 100)

        return sum(scores) / len(scores) if scores else 0.0
