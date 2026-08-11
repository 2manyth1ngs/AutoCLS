"""比较器 zero-shot 排名的纯聚合数学（TECHNICAL_DESIGN.md 模块 8）。

来源对齐：AutoCTS++ exps/random_search.py 的比较器两两比较（单任务、单模型）；
Round-Robin 胜场 + 跨任务 Borda 聚合 + 多种子多数票是本项目对"数据集级 CLS
推荐"的扩展（AutoCTS++ 无此层——它每个任务独立搜索，本项目要给整个目标
数据集推荐一个 CLS，必须跨切片任务聚合）。

任务无关模块（铁律 7）：只吃比较器输出的成对概率矩阵，不 import task_types
——"P(a 优于 b) 的方向语义"由造 label 侧（labeling.make_pairs → is_better）
定义，比较器输出的排名继承该语义，这里是纯 numpy 聚合，无任何任务类型假设。

约定：成对概率矩阵 P (n, n)，P[i, j] = 模型输出的 P(候选 i 优于候选 j)，
对角线 NaN（不自比）。
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np


def symmetrize(P: np.ndarray) -> np.ndarray:
    """双向预测合成单一得分：S[i,j] = (P[i,j] + 1 - P[j,i]) / 2。

    比较器对 (a,b) 与 (b,a) 的预测不保证互补（模型无反对称约束）；评估侧
    历来双向出样（with_reversed），排名侧同理取两个方向的均值，S + Sᵀ = 1
    恒成立（对角除外），保证胜负关系无歧义。
    """
    P = np.asarray(P, dtype=float)
    assert P.ndim == 2 and P.shape[0] == P.shape[1], f"P 应为方阵，得到 {P.shape}"
    return (P + 1.0 - P.T) / 2.0


def wins_from_scores(S: np.ndarray) -> np.ndarray:
    """Round-Robin 胜场：wins[i] = #{j ≠ i : S[i,j] > 0.5}。

    恰 0.5（模型完全无倾向）双方都不得分——不赋假方向，与 make_pairs 的
    严格平局弃样同精神。
    """
    with np.errstate(invalid="ignore"):
        return np.nansum(S > 0.5, axis=1).astype(int)


def pair_vote_matrix(S_list: Sequence[np.ndarray]) -> np.ndarray:
    """多种子（多模型）逐无序对多数票 → 胜负矩阵 V，V[i,j]=1 表示 i 胜 j。

    每个种子对无序对 (i,j) 投一票：S[i,j] > 0.5 票给 i，< 0.5 票给 j，恰 0.5
    弃票。多数者胜；平票（含全弃票）用跨种子平均 S 破，仍平（=0.5）则双方
    都不得分。多数票比平均概率稳健：单个种子的极端置信不会翻转多数方向。
    """
    assert S_list, "至少需要一个得分矩阵"
    n = S_list[0].shape[0]
    stack = np.stack([np.asarray(S, dtype=float) for S in S_list])  # (k, n, n)
    assert stack.shape[1:] == (n, n)
    V = np.zeros((n, n), dtype=int)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 对角全 NaN 的空切片均值
        mean_S = np.nanmean(stack, axis=0)
    for i in range(n):
        for j in range(i + 1, n):
            votes_i = int(np.sum(stack[:, i, j] > 0.5))
            votes_j = int(np.sum(stack[:, i, j] < 0.5))
            if votes_i > votes_j:
                V[i, j] = 1
            elif votes_j > votes_i:
                V[j, i] = 1
            elif mean_S[i, j] > 0.5:
                V[i, j] = 1
            elif mean_S[i, j] < 0.5:
                V[j, i] = 1
            # 完全平局：双方都不得分
    return V


def borda_ranking(
    wins_list: Sequence[np.ndarray],
    tiebreak_scores: np.ndarray,
    keys: Sequence[str],
) -> list[int]:
    """跨任务 Borda 聚合：总胜场降序 → 并列按 tiebreak 得分降序 → 再按 key 升序。

    wins_list: 每任务一条胜场向量（同一候选顺序）；tiebreak_scores: 跨任务/种子
    平均对称化得分（连续量，破并列）；keys: 候选稳定键（最终确定性兜底——
    同分同得分时排名仍逐位可复现）。返回候选下标的排名顺序（第 0 位 = top-1）。
    """
    total = np.sum(np.stack([np.asarray(w) for w in wins_list]), axis=0)
    assert len(total) == len(tiebreak_scores) == len(keys)
    order = sorted(
        range(len(keys)),
        key=lambda i: (-int(total[i]), -float(tiebreak_scores[i]), keys[i]),
    )
    return order
