import Levenshtein
import jellyfish
import numpy as np
from scipy.optimize import linear_sum_assignment


def dict_similarity(
    pred_dict: dict[str, list[str | int | float]],
    true_dict: dict[str, list[str | int | float]],
) -> float:
    """分阶段字典相似度匹配算法"""

    def _header_similarity(h1: str, h2: str) -> float:
        """带语义修正的表头相似度"""
        replacements = [("Number", "#"), ("Date", "Time"), ("%", "Percent")]
        for a, b in replacements:
            h1 = h1.replace(a, b)
            h2 = h2.replace(a, b)
        sim = jellyfish.jaro_similarity(h1.lower(), h2.lower())
        return sim if sim > threadhold else 0

    def _row_similarity(pred_row, true_row, matched_pairs):
        """跨数据类型的行相似度计算"""
        if not matched_pairs:
            return 0
        total_score = 0
        for p_idx, t_idx, h_sim in matched_pairs:
            p_val, t_val = pred_row[p_idx], true_row[t_idx]
            # 自动检测数据类型
            if isinstance(t_val, (int, float)) and isinstance(p_val, (int, float)):
                cell_sim = 1 - abs(p_val - t_val) / (abs(t_val) + 1e-5)
            else:
                str1, str2 = str(p_val), str(t_val)
                cell_sim = 1 - Levenshtein.distance(str1, str2) / max(len(str1), 1)
            total_score += h_sim * (
                cell_sim if cell_sim > threadhold else 0
            )  # 加权相似度
        return total_score / len(matched_pairs)

    # 设置相似度阈值
    threadhold = 0.5
    # 阶段1：表头双向匹配
    header_matrix = [
        [_header_similarity(ph, th) for th in true_dict] for ph in pred_dict
    ]
    row_index1, col_index1 = linear_sum_assignment(1 - np.array(header_matrix))
    matched_pairs = [
        (row, col, header_matrix[row][col])
        for row, col in zip(row_index1, col_index1)
        if header_matrix[row][col] > threadhold
    ]  # 过滤低质量匹配

    # 阶段2：元素行级匹配
    true_rows = list(zip(*true_dict.values()))
    pred_rows = list(zip(*pred_dict.values()))
    row_matrix = [
        [_row_similarity(pred_row, true_row, matched_pairs) for true_row in true_rows]
        for pred_row in pred_rows
    ]
    row_index2, col_index2 = linear_sum_assignment(1 - np.array(row_matrix))
    # 记录有效匹配行
    valid_matches = [
        row_matrix[r][c]
        for r, c in zip(row_index2, col_index2)
        if row_matrix[r][c] > threadhold
    ]

    # 计算元素行匹配的得分和系数，表头匹配的得分和系数
    row_coefficient = (
        1 - (abs(len(pred_rows) - len(valid_matches)) / len(pred_rows))
        if len(pred_rows) > 0
        else float(len(pred_rows) == len(true_rows))
    )
    row_score = sum(valid_matches) / len(pred_rows)
    header_coeffient = (
        1 - (abs(len(pred_dict) - len(true_dict)) / len(pred_dict))
        if len(pred_dict) > 0
        else float(len(pred_dict) == len(true_dict))
    )
    header_score = sum(sim for _, _, sim in matched_pairs) / max(len(pred_dict), 1)
    # print(header_coeffient, header_score, row_coefficient, row_score)
    return 0.4 * header_coeffient * header_score + 0.6 * row_coefficient * row_score