"""
公式评估模块
评估数学公式的准确性（LaTeX、MathML等格式）
支持AST树匹配、符号匹配、结构匹配等多种方法
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any

from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)

try:
    from sympy import sympify, simplify, nsimplify
    from sympy.parsing.latex import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("sympy not available, formula AST matching will be limited")


class FormulaEvaluator:
    """公式评估器"""

    @staticmethod
    def normalize_latex(latex_str: str) -> str:
        """
        标准化LaTeX公式字符串
        - 去除多余空格
        - 统一括号格式
        - 标准化命令名

        Args:
            latex_str: LaTeX字符串

        Returns:
            标准化后的LaTeX字符串
        """
        if not latex_str:
            return ""

        # 去除多余空格
        normalized = re.sub(r'\s+', ' ', latex_str.strip())

        # 统一花括号和空格
        normalized = re.sub(r'\{\s+', '{', normalized)
        normalized = re.sub(r'\s+\}', '}', normalized)

        # x^2 → x^{2}（单字符上下标统一加花括号）
        normalized = re.sub(r'\^([^{])', r'^{\1}', normalized)
        normalized = re.sub(r'_([^{])', r'_{\1}', normalized)

        # 转换常见等价变体
        replacements = [
            (r'\\dfrac', r'\\frac'),          # \dfrac → \frac
            (r'\\tfrac', r'\\frac'),           # \tfrac → \frac
            (r'\\operatorname', r'\\mathrm'),  # \operatorname → \mathrm
            (r'\\text\b', r'\\mathrm'),        # \text → \mathrm
            (r'\\boldsymbol', r'\\mathbf'),    # \boldsymbol → \mathbf
            (r'\\bm\b', r'\\mathbf'),          # \bm → \mathbf
            (r'\{\\rm\s+', r'\\mathrm{'),      # {\rm d} → \mathrm{d}
            (r'\\left\s*\(', r'('),            # \left( → (
            (r'\\right\s*\)', r')'),           # \right) → )
            (r'\\left\s*\[', r'['),            # \left[ → [
            (r'\\right\s*\]', r']'),           # \right] → ]
            (r'\\left\s*\{', r'{'),            # \left{ → {
            (r'\\right\s*\}', r'}'),           # \right} → }
            (r'\\,', r' '),                    # 细空格统一为空格
            (r'\\;', r' '),
            (r'\\quad', r' '),
            (r'\\qquad', r' '),
        ]

        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)

        # 再次折叠多余空格
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    @staticmethod
    def normalize_mathml(mathml_str: str) -> str:
        """
        标准化MathML字符串
        - 去除属性值空格
        - 统一标签格式
        """
        if not mathml_str:
            return ""

        # 去除多余空格
        normalized = re.sub(r'>\s+<', '><', mathml_str.strip())
        normalized = re.sub(r'\s+', ' ', normalized)

        # 去除属性中的多余空格
        normalized = re.sub(r'=\s+"', '="', normalized)
        normalized = re.sub(r'"\s+', '"', normalized)

        return normalized

    @staticmethod
    def extract_math_symbols(formula: str) -> set:
        """
        提取公式中的数学符号

        Args:
            formula: 公式字符串（LaTeX或纯文本）

        Returns:
            符号集合
        """
        # 提取 \command 形式的命令
        commands = set(re.findall(r'\\[a-zA-Z]+', formula))

        # 提取变量和常数（字母和数字）
        variables = set(re.findall(r'[a-zA-Z0-9]', formula))

        # 提取操作符
        operators = set(re.findall(r'[+\-*/=<>()[\]{}^_]', formula))

        return commands | variables | operators

    @staticmethod
    def symbol_accuracy(pred_formula: str, ref_formula: str) -> float:
        """
        基于符号的准确率
        计算预测公式中有多少符号与参考公式匹配

        Returns:
            准确率 (0-1)
        """
        pred_symbols = FormulaEvaluator.extract_math_symbols(pred_formula)
        ref_symbols = FormulaEvaluator.extract_math_symbols(ref_formula)

        if not ref_symbols:
            return 0.0 if not pred_symbols else 0.0

        correct_symbols = len(pred_symbols & ref_symbols)
        return correct_symbols / len(ref_symbols)

    @staticmethod
    def structure_similarity(pred_formula: str, ref_formula: str) -> float:
        """
        基于公式结构的相似度
        考虑公式的嵌套深度和主要操作符结构

        Returns:
            相似度 (0-1)
        """
        # 简化：提取主要结构（括号嵌套、分数、根号等）
        def get_structure(formula):
            # 计数括号、花括号、方括号的嵌套
            bracket_depth = 0
            max_depth = 0
            special_ops = 0

            for char in formula:
                if char in '([{':
                    bracket_depth += 1
                    max_depth = max(max_depth, bracket_depth)
                elif char in ')]}':
                    bracket_depth = max(0, bracket_depth - 1)
                elif char in '^_' or '\\frac' in formula or '\\sqrt' in formula:
                    special_ops += 1

            return (max_depth, special_ops, len(formula))

        pred_struct = get_structure(pred_formula)
        ref_struct = get_structure(ref_formula)

        # 结构相似度基于最大深度和特殊操作数
        if pred_struct == ref_struct:
            return 1.0

        depth_diff = abs(pred_struct[0] - ref_struct[0])
        ops_diff = abs(pred_struct[1] - ref_struct[1])

        # 最大差异值
        max_diff = max(depth_diff, ops_diff)

        return max(0.0, 1.0 - max_diff / 5.0)

    @staticmethod
    def sequence_edit_distance(pred_formula: str, ref_formula: str) -> float:
        """
        基于编辑距离的准确率

        Returns:
            准确率 (0-1)
        """
        return MetricsCalculator.character_accuracy(pred_formula, ref_formula)

    @staticmethod
    def parse_latex_to_ast(latex_str: str) -> Optional[Any]:
        """
        将LaTeX公式解析为SymPy AST（语法树）

        Args:
            latex_str: LaTeX字符串

        Returns:
            SymPy表达式对象或None（如果解析失败）
        """
        if not SYMPY_AVAILABLE or not latex_str:
            return None

        try:
            # 预处理：删除$符号
            latex_clean = latex_str.strip('$').strip()
            # 尝试解析为SymPy表达式
            expr = parse_expr(latex_clean, transformations='all')
            return expr
        except Exception as e:
            logger.debug(f"Failed to parse LaTeX to AST: {latex_str}, error: {e}")
            return None

    @staticmethod
    def ast_distance(pred_ast: Any, ref_ast: Any) -> float:
        """
        计算两个AST之间的距离（基于数值等价性）

        Returns:
            距离值 (0-1)，0表示完全相同，1表示完全不同
        """
        if not SYMPY_AVAILABLE:
            return 0.5

        if pred_ast is None or ref_ast is None:
            return 1.0

        try:
            # 尝试检查数值等价性
            diff = simplify(pred_ast - ref_ast)
            if diff == 0:
                return 0.0  # 完全相同
            else:
                return 0.5  # 结构不同但可能数值相关
        except Exception:
            return 0.5

    @staticmethod
    def ast_similarity(pred_latex: str, ref_latex: str) -> float:
        """
        基于AST的相似度（如果sympy可用）

        Returns:
            相似度 (0-1)
        """
        if not SYMPY_AVAILABLE:
            return 0.5

        pred_ast = FormulaEvaluator.parse_latex_to_ast(pred_latex)
        ref_ast = FormulaEvaluator.parse_latex_to_ast(ref_latex)

        if pred_ast is None or ref_ast is None:
            # 降级到字符串比较
            return MetricsCalculator.sequence_similarity(pred_latex, ref_latex)

        distance = FormulaEvaluator.ast_distance(pred_ast, ref_ast)
        return 1.0 - distance

    @staticmethod
    def latex_accuracy(pred_latex: str, ref_latex: str) -> float:
        """
        计算LaTeX公式的准确率 - 完整版本

        采用多层级评估：
        1. 完全相同：1.0
        2. AST等价：0.95-1.0
        3. 结构相似：加权组合多个指标

        Args:
            pred_latex: 预测的LaTeX字符串
            ref_latex: 参考的LaTeX字符串

        Returns:
            准确率 (0-1)
        """
        # 标准化
        pred_norm = FormulaEvaluator.normalize_latex(pred_latex)
        ref_norm = FormulaEvaluator.normalize_latex(ref_latex)

        # 如果完全相同
        if pred_norm == ref_norm:
            return 1.0

        # 尝试AST匹配（如果sympy可用）
        if SYMPY_AVAILABLE:
            ast_sim = FormulaEvaluator.ast_similarity(pred_norm, ref_norm)
            if ast_sim >= 0.95:
                return round(ast_sim, 4)

        # 综合多个指标
        sym_acc = FormulaEvaluator.symbol_accuracy(pred_norm, ref_norm)
        struct_sim = FormulaEvaluator.structure_similarity(pred_norm, ref_norm)
        edit_acc = FormulaEvaluator.sequence_edit_distance(pred_norm, ref_norm)

        # 加权组合：编辑距离 60%，符号 20%，结构 20%
        accuracy = (edit_acc * 0.6 + sym_acc * 0.2 + struct_sim * 0.2)

        return round(accuracy, 4)

    @staticmethod
    def mathml_accuracy(pred_mathml: str, ref_mathml: str) -> float:
        """
        计算MathML的准确率

        Args:
            pred_mathml: 预测的MathML字符串
            ref_mathml: 参考的MathML字符串

        Returns:
            准确率 (0-1)
        """
        # 标准化
        pred_norm = FormulaEvaluator.normalize_mathml(pred_mathml)
        ref_norm = FormulaEvaluator.normalize_mathml(ref_mathml)

        # 如果完全相同
        if pred_norm == ref_norm:
            return 1.0

        # 简化：直接计算序列相似度
        accuracy = MetricsCalculator.sequence_similarity(pred_norm, ref_norm)

        return round(accuracy, 4)

    @staticmethod
    def formula_accuracy(
        pred_formula: str,
        ref_formula: str,
        formula_type: str = "latex"
    ) -> float:
        """
        通用公式准确率计算

        Args:
            pred_formula: 预测公式
            ref_formula: 参考公式
            formula_type: 公式类型 ("latex", "mathml", "plain")

        Returns:
            准确率 (0-1)
        """
        if not pred_formula and not ref_formula:
            return 1.0

        if not pred_formula or not ref_formula:
            return 0.0

        if formula_type.lower() == "latex":
            return FormulaEvaluator.latex_accuracy(pred_formula, ref_formula)
        elif formula_type.lower() == "mathml":
            return FormulaEvaluator.mathml_accuracy(pred_formula, ref_formula)
        else:
            # 纯文本公式
            return MetricsCalculator.character_accuracy(pred_formula, ref_formula)

    @staticmethod
    def evaluate_formulas(
        pred_formulas: List[str],
        ref_formulas: List[str],
        formula_type: str = "latex"
    ) -> Dict:
        """
        评估多个公式

        Args:
            pred_formulas: 预测公式列表
            ref_formulas: 参考公式列表
            formula_type: 公式类型

        Returns:
            评估结果字典
        """
        if not ref_formulas:
            return {
                'total_formulas': 0,
                'average_accuracy': 0.0,
                'correct_formulas': 0,
                'error': 'No reference formulas'
            }

        accuracies = []
        correct_count = 0
        used_pred_indices = set()

        # 最优匹配：对每个 GT 公式，找相似度最高的 OCR 公式（每个 OCR 公式只用一次）
        for ref_formula in ref_formulas:
            best_acc = 0.0
            best_idx = -1

            for j, pred_formula in enumerate(pred_formulas):
                if j in used_pred_indices:
                    continue
                acc = FormulaEvaluator.formula_accuracy(pred_formula, ref_formula, formula_type)
                if acc > best_acc:
                    best_acc = acc
                    best_idx = j

            if best_idx >= 0:
                used_pred_indices.add(best_idx)
            # 若 OCR 没有匹配到任何公式，best_acc 保持 0.0

            accuracies.append(best_acc)

            if best_acc >= 0.8:  # 认为准确（阈值从 0.95 降至 0.8）
                correct_count += 1

        average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

        return {
            'total_formulas': len(ref_formulas),
            'average_accuracy': round(average_accuracy, 4),
            'correct_formulas': correct_count,
            'accuracy_list': [round(a, 4) for a in accuracies],
        }

    @staticmethod
    def extract_formulas_from_text(text: str) -> List[str]:
        """
        从文本中提取LaTeX公式

        支持格式：
        - $...$: 行内公式
        - $$...$$: 块级公式
        - \\(...\\): 行内公式
        - \\[...\\]: 块级公式

        Args:
            text: 包含公式的文本

        Returns:
            公式列表
        """
        formulas = []

        # 提取 $...$ 格式
        formulas.extend(re.findall(r'\$([^$]+)\$', text))

        # 提取 $$...$$ 格式
        formulas.extend(re.findall(r'\$\$([^$]+)\$\$', text))

        # 提取 \(...\) 格式
        formulas.extend(re.findall(r'\\\(([^)]+)\\\)', text))

        # 提取 \[...\] 格式
        formulas.extend(re.findall(r'\\\[([^\]]+)\\\]', text))

        return formulas
