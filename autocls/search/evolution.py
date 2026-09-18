"""CLS 策略空间的任务无关进化搜索（TECHNICAL_DESIGN.md 模块 8 附属；AutoCTS++ Algorithm 2
适配，reference/AutoCTS++.pdf §3.3 "Search strategy"）。

论文 §3.3 原味语义：
1. 采样 Ks 个候选，两两比较产生 Ks(Ks-1)/2 对，用比较器打分后 Round-Robin 计胜场排序
   （"比较器不保证传递性，因此不能用依赖传递性的高效排序算法"——只能数胜场），取
   top-kp 作初始种群；
2. 每个进化步：种群里每个候选各自以概率 p1 交叉、概率 p2 变异产生后代，后代并入种群，
   再用比较器重新 Round-Robin 排序，截断回种群规模 kp；
3. 最终对种群做一次 Round-Robin 取 top-K。

论文只给出"crossover/mutation probabilities p1/p2"的自然语言描述（原文："Each arch-hyper
has crossover and mutation probabilities p1 and p2, respectively, when generating new
offspring in each evolution step."），没有给出交叉/变异是否互斥的伪代码级分支顺序；
reference/AutoCTS_plusplus 代码库里也无对应实现（只有 exps/random_search.py:550-583 的
"随机采样 + 两两比较 + heap 取 top-k" 兜底逻辑，服务的是论文里更早的、无交叉变异的朴素基线，
不是这里的 Algorithm 2）。

（论文未给全，按 X 语义补全，铁律 5 附注）本文件按 task-3-brief.md 给定的顺序语义补全：
对每个待生成的后代，先掷一次 [0,1) 骰子——命中 [0, p_cross) 走交叉（两个随机父代逐字段
均匀选值）；否则再掷一次骰子，命中 [0, p_mut) 走变异（随机选一个字段，从 space.choices
里重采样该字段取值）；两次都未命中则直接复制一个随机父代。两次骰子的调用顺序固定，
使得整个搜索在给定 rng 下完全确定（determinism 是 interfaces.md 的硬要求）。

任务无关（铁律 7）：本模块只依赖 score_pairs（纯函数：候选向量对 → P(a≻b)）、
CLSSpace、CLSEncoder 三个协作方，不 import task_types、不出现任何数据集/任务概念——
score_pairs 内部要不要闭包 E/ctx 是调用方的事。

CLS 合法性来源（铁律 1）：交叉只在两个已合法父代之间逐字段挑值（不产生新取值，但字段
组合可能撞上门控约束）；变异用 space.choices(field) 重采样该字段（唯一合法取值来源）。
产生的后代一律经 space.validate 校验，非法则丢弃重试（有限次数），不做任何"自动修正
门控字段"的隐式纠正——brief 明确交给 validate + retry 把关，不发明新约束。
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Sequence

import numpy as np

from ..labeling import cls_key
from .rank import symmetrize, wins_from_scores

# score_pairs(vecs_a, vecs_b) -> P(a ≻ b)，均为 (n, d) → (n,)
ScorePairs = Callable[[np.ndarray, np.ndarray], np.ndarray]

_MAX_OFFSPRING_ATTEMPTS = 50       # 单个后代合法性重试上限（brief："有限次数后跳过"）
_SAMPLE_ATTEMPTS_FACTOR = 50       # 去重采样的尝试预算（次数 = n * 该系数）


def _pairwise_wins(
    candidates: Sequence,
    encoder,
    score_pairs: ScorePairs,
    batch_size: int,
) -> np.ndarray:
    """候选集合的全量成对打分 → 对称化 → Round-Robin 胜场（rank.py 复用，不重复实现）。

    n(n-1) 个有序对按 batch_size 切批喂给 score_pairs（brief："在 batches 里跑，默认
    4096 对"）——避免 n 很大（如初始 ks=2000）时一次性构造过大的向量批。
    """
    n = len(candidates)
    vecs = np.stack([encoder.encode(c) for c in candidates])
    P = np.full((n, n), np.nan, dtype=float)
    if n > 1:
        ii, jj = np.where(~np.eye(n, dtype=bool))
        for start in range(0, ii.size, batch_size):
            a_idx = ii[start:start + batch_size]
            b_idx = jj[start:start + batch_size]
            P[a_idx, b_idx] = np.asarray(score_pairs(vecs[a_idx], vecs[b_idx]), dtype=float)
    S = symmetrize(P)
    return wins_from_scores(S)


def _sample_unique_cls(space, rng: np.random.Generator, n: int, exclude_keys=(), *,
                       sampler: Optional[Callable] = None, accept: Optional[Callable] = None) -> list:
    """采样 n 个互不相同（按 cls_key）的合法 CLS。

    space.sample_random 恒产出合法配置（test_sample_10000_all_valid 已验收），这里
    只需要处理"撞库"去重，不需要重复校验合法性。
    WP16 EV1 钩子：`sampler(rng) -> CLS` 换掉 space.sample_random（调用方限定搜索域，如"信任域邻域"）；
    `accept(cls) -> bool` 为额外接受域，不过即丢弃重采。两者缺省时逐位等于旧行为。
    """
    seen = set(exclude_keys)
    out = []
    max_attempts = max(n * _SAMPLE_ATTEMPTS_FACTOR, 200)
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        c = space.sample_random(rng) if sampler is None else sampler(rng)
        if accept is not None and not accept(c):
            continue
        k = cls_key(c)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    if len(out) < n:
        raise RuntimeError(
            f"采样 {n} 个去重 CLS 失败：{attempts} 次尝试后仅得 {len(out)} 个"
            "（策略空间过小或 exclude_keys 排除过多）"
        )
    return out


def _crossover(a, b, rng: np.random.Generator, field_names: Sequence[str]):
    """逐字段均匀选 a/b 之一（可能撞上门控约束，由调用方 validate 把关，不在此修正）。"""
    values = {name: (getattr(a, name) if rng.random() < 0.5 else getattr(b, name))
              for name in field_names}
    return type(a)(**values)


def _mutate(a, rng: np.random.Generator, space):
    """单字段重采样：等概率选一个字段，取值从 space.choices(该字段) 重采（铁律 1）。"""
    field_names = space.field_names
    name = field_names[int(rng.integers(len(field_names)))]
    choices = space.choices(name)
    values = {f: getattr(a, f) for f in field_names}
    values[name] = choices[int(rng.integers(len(choices)))]
    return type(a)(**values)


def _make_offspring(population: Sequence, rng: np.random.Generator, space, p_cross: float, p_mut: float, *,
                    mutator: Optional[Callable] = None, accept: Optional[Callable] = None):
    """按模块 docstring 的顺序语义产生一个合法后代；重试超限仍非法则返回 None。

    WP16 EV1 钩子：`mutator(cls, rng) -> CLS` 换掉单字段全档重采（如局部 ±1/±2 档一步变异）；`accept` 同 _sample_unique_cls
    （交叉 / 变异 / 复制出来的后代都要过它——交叉两个域内父代也可能跳出域）。缺省逐位等于旧行为。
    """
    field_names = space.field_names
    n = len(population)
    for _ in range(_MAX_OFFSPRING_ATTEMPTS):
        if rng.random() < p_cross:
            i, j = int(rng.integers(n)), int(rng.integers(n))
            child = _crossover(population[i], population[j], rng, field_names)
        elif rng.random() < p_mut:
            i = int(rng.integers(n))
            child = _mutate(population[i], rng, space) if mutator is None else mutator(population[i], rng)
        else:
            i = int(rng.integers(n))
            child = population[i]
        if space.validate(child) and (accept is None or accept(child)):
            return child
    return None


def evolutionary_search(
    score_pairs: ScorePairs,
    space,
    encoder,
    *,
    ks: int = 2000,
    kp: int = 50,
    generations: int = 20,
    p_cross: float = 0.8,
    p_mut: float = 0.2,
    top_k: int = 3,
    batch_size: int = 4096,
    rng: np.random.Generator,
    sampler: Optional[Callable] = None,
    mutator: Optional[Callable] = None,
    accept: Optional[Callable] = None,
    seed_population: Optional[Sequence] = None,
) -> list[dict]:
    """AutoCTS++ Algorithm 2 的进化搜索部分（模块 docstring 有完整语义说明）。

    score_pairs: (vecs_a, vecs_b) -> P(a≻b)，调用方通过闭包注入 E/ctx 等任务上下文
        （本函数不知道、也不关心 score_pairs 内部在比什么——铁律 7）。
    space: CLSSpace；encoder: CLSEncoder。ks/kp/generations/p_cross/p_mut/top_k 语义
        见 interfaces.md「Task 3 → Tasks 6, 7」；batch_size 是 score_pairs 单批喂入的
        候选对数上限（brief 新增，默认 4096）。rng 无默认值——调用方必须显式传入
        np.random.Generator 以保证整个搜索在给定种子下逐位确定。
    WP16 EV1 钩子（results/tclsc_ev/PROTOCOL.md §1.2，全部可选、缺省逐位等于旧行为）：
        sampler(rng) -> CLS：初始采样器（限定搜索域）；mutator(cls, rng) -> CLS：变异算子；
        accept(cls) -> bool：接受域（采样与后代都要过）；seed_population：预先放进初始候选池的配置
        （如信任域配置，按 cls_key 去重、不计入 ks；调用方保证其合法且在域内），它们与采样候选一起排序。

    返回按 wins 降序（并列按 cls_key 升序，interfaces.md 的稳定 tie-break）排序的
    top-k 条目：{"cls": asdict(CLS), "cls_key", "wins", "generation_found"}（top_k=kp 即整个最终种群）。
    """
    # 1) 采样 ks 个候选 + Round-Robin 排序 → 初始种群 top-kp（论文 §3.3 第一段）
    seeds: list = []
    seen_seed: set = set()
    for c in (seed_population or ()):
        k = cls_key(c)
        if k not in seen_seed:
            seen_seed.add(k)
            seeds.append(c)
    sampled = seeds + _sample_unique_cls(space, rng, ks, exclude_keys=seen_seed, sampler=sampler, accept=accept)
    wins0 = _pairwise_wins(sampled, encoder, score_pairs, batch_size)
    keys0 = [cls_key(c) for c in sampled]
    order0 = sorted(range(len(sampled)), key=lambda i: (-int(wins0[i]), keys0[i]))

    population = [sampled[i] for i in order0[:kp]]
    pop_keys = [keys0[i] for i in order0[:kp]]
    generation_found: dict[str, int] = {k: 0 for k in pop_keys}

    # 2) 每代交叉/变异产生 kp 个后代，并入种群后重新 Round-Robin 排序截断回 kp
    for gen in range(1, generations + 1):
        offspring, offspring_keys = [], []
        seen = set(pop_keys)
        for _ in range(kp):
            child = _make_offspring(population, rng, space, p_cross, p_mut, mutator=mutator, accept=accept)
            if child is None:
                continue
            k = cls_key(child)
            if k in seen:
                continue
            seen.add(k)
            offspring.append(child)
            offspring_keys.append(k)
            generation_found.setdefault(k, gen)

        pool = population + offspring
        pool_keys = pop_keys + offspring_keys
        wins = _pairwise_wins(pool, encoder, score_pairs, batch_size)
        order = sorted(range(len(pool)), key=lambda i: (-int(wins[i]), pool_keys[i]))

        population = [pool[i] for i in order[:kp]]
        pop_keys = [pool_keys[i] for i in order[:kp]]

    # 3) 最终种群做一次 Round-Robin，取 top-k（wins 降序，并列按 cls_key 升序）
    final_wins = _pairwise_wins(population, encoder, score_pairs, batch_size)
    order = sorted(range(len(population)), key=lambda i: (-int(final_wins[i]), pop_keys[i]))

    return [
        {
            "cls": population[i].to_dict(),
            "cls_key": pop_keys[i],
            "wins": int(final_wins[i]),
            "generation_found": generation_found[pop_keys[i]],
        }
        for i in order[:top_k]
    ]
