"""
指标计算工具类
提供文本、表格、公式的各种相似度和准确率计算方法
"""

import logging
from difflib import SequenceMatcher
import editdistance
import re

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """通用指标计算器"""

    @staticmethod
    def character_error_rate(pred: str, ref: str) -> float:
        """
        计算字符错误率 (Character Error Rate, CER)
        CER = (插入+删除+替换) / 标准字符数

        Args:
            pred: 预测文本
            ref: 参考文本

        Returns:
            CER值 (0-1)，越小越好
        """
        if not ref:
            return 0.0 if not pred else 1.0

        if not pred:
            return 1.0

        dist = editdistance.eval(pred, ref)
        return min(1.0, dist / len(ref))

    @staticmethod
    def character_accuracy(pred: str, ref: str) -> float:
        """字符准确率 = 1 - CER"""
        return 1.0 - MetricsCalculator.character_error_rate(pred, ref)

    @staticmethod
    def word_error_rate(pred: str, ref: str) -> float:
        """
        计算单词错误率 (Word Error Rate, WER)
        基于空格分割的单词
        """
        pred_words = pred.split()
        ref_words = ref.split()

        if not ref_words:
            return 0.0 if not pred_words else 1.0

        dist = editdistance.eval(pred_words, ref_words)
        return min(1.0, dist / len(ref_words))

    @staticmethod
    def word_accuracy(pred: str, ref: str) -> float:
        """单词准确率 = 1 - WER"""
        return 1.0 - MetricsCalculator.word_error_rate(pred, ref)

    @staticmethod
    def sequence_similarity(pred: str, ref: str) -> float:
        """
        序列相似度 (基于SequenceMatcher)
        范围: 0-1，越高越好
        """
        if not pred and not ref:
            return 1.0
        if not pred or not ref:
            return 0.0

        return SequenceMatcher(None, ref, pred).ratio()

    @staticmethod
    def jaccard_similarity(pred: str, ref: str) -> float:
        """
        Jaccard相似度 (基于字符集)
        J(A,B) = |A∩B| / |A∪B|
        范围: 0-1，越高越好
        """
        if not pred and not ref:
            return 1.0
        if not pred or not ref:
            return 0.0

        pred_chars = set(pred)
        ref_chars = set(ref)
        intersection = len(pred_chars & ref_chars)
        union = len(pred_chars | ref_chars)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def edit_distance(pred: str, ref: str) -> int:
        """
        编辑距离 (Levenshtein Distance)
        返回将pred转换为ref所需的最少操作数
        """
        return editdistance.eval(pred, ref)

    @staticmethod
    def normalized_edit_distance(pred: str, ref: str) -> float:
        """
        标准化编辑距离 (0-1)
        NED = EditDistance / max(len(pred), len(ref))
        """
        if not pred and not ref:
            return 0.0

        dist = editdistance.eval(pred, ref)
        max_len = max(len(pred), len(ref))
        return min(1.0, dist / max_len) if max_len > 0 else 0.0

    @staticmethod
    def bleu_score(pred: str, ref: str, n_gram: int = 2) -> float:
        """
        简化的BLEU分数（n-gram匹配）
        用于序列相似度评估
        """
        if not pred or not ref:
            return 0.0

        pred_ngrams = MetricsCalculator._extract_ngrams(pred, n_gram)
        ref_ngrams = MetricsCalculator._extract_ngrams(ref, n_gram)

        if not ref_ngrams:
            return 0.0

        matches = sum((pred_ngrams & ref_ngrams).values())
        return matches / len(ref_ngrams)

    @staticmethod
    def _extract_ngrams(text: str, n: int) -> dict:
        """提取n-gram"""
        ngrams = {}
        for i in range(len(text) - n + 1):
            gram = text[i:i+n]
            ngrams[gram] = ngrams.get(gram, 0) + 1
        return ngrams

    @staticmethod
    def composite_score(
        char_acc: float,
        seq_sim: float,
        jaccard_sim: float,
        weights: tuple = (0.5, 0.3, 0.2)
    ) -> float:
        """
        综合评分 (0-100)

        Args:
            char_acc: 字符准确率 (0-1)
            seq_sim: 序列相似度 (0-1)
            jaccard_sim: Jaccard相似度 (0-1)
            weights: 权重 (char_weight, seq_weight, jaccard_weight)

        Returns:
            综合评分 (0-100)
        """
        w_char, w_seq, w_jaccard = weights
        total_weight = w_char + w_seq + w_jaccard

        if total_weight == 0:
            return 0.0

        score = (
            char_acc * w_char +
            seq_sim * w_seq +
            jaccard_sim * w_jaccard
        ) / total_weight

        return round(score * 100, 2)
