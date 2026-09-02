"""进化搜索单测（TECHNICAL_DESIGN.md 模块 8 附属；AutoCTS++ Algorithm 2 适配，铁律 7 任务无关）。

合成场景：真实 CLS 编码向量 + 隐藏线性效用 u(cls) = vec·w，P(a≻b) = sigmoid(u_a - u_b)
——具传递性（无噪声），用来验证"进化确实能找到更优候选"，而不依赖真实比较器的准确率。

全部用 configs/space.yaml 的真实空间 + 小规模 ks/kp/generations，CPU 秒级跑完。
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from autocls.labeling import cls_key
from autocls.search.evolution import (
    _crossover,
    _make_offspring,
    _mutate,
    _sample_unique_cls,
    evolutionary_search,
)
from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLS, CLSSpace


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


@pytest.fixture(scope="module")
def encoder(space):
    return CLSEncoder(space)


def _make_scorer(encoder, seed):
    """隐藏线性效用 + sigmoid 差分打分：具传递性，无评估噪声。"""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(encoder.dim)

    def utility(cls_cfg) -> float:
        return float(encoder.encode(cls_cfg) @ w)

    def score_pairs(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
        u_a = vecs_a @ w
        u_b = vecs_b @ w
        return 1.0 / (1.0 + np.exp(-(u_a - u_b)))

    return score_pairs, utility


SMALL = dict(ks=64, kp=8, generations=5, top_k=3)


# ---------------------------------------------------------------- crossover/mutation 内部契约
class TestOffspringBuildingBlocks:
    def test_crossover_picks_only_parent_values(self, space):
        # 逐字段均匀取 a 或 b 的值，不引入任何第三方取值（铁律 1）
        rng = np.random.default_rng(1)
        a, b = space.sample_random(rng), space.sample_random(rng)
        for _ in range(200):
            child = _crossover(a, b, rng, space.field_names)
            for name in space.field_names:
                assert getattr(child, name) in (getattr(a, name), getattr(b, name))

    def test_mutation_changes_at_most_one_field_from_space_choices(self, space):
        rng = np.random.default_rng(2)
        a = space.sample_random(rng)
        for _ in range(200):
            child = _mutate(a, rng, space)
            diffs = [n for n in space.field_names if getattr(child, n) != getattr(a, n)]
            assert len(diffs) <= 1
            if diffs:
                name = diffs[0]
                assert getattr(child, name) in space.choices(name)

    def test_make_offspring_result_always_valid_when_not_none(self, space):
        # 无论走交叉/变异/复制哪条支路，返回的后代必须合法；非法则重试后要么合法要么 None
        rng = np.random.default_rng(3)
        population = [space.sample_random(rng) for _ in range(10)]
        n_none = 0
        for _ in range(500):
            child = _make_offspring(population, rng, space, p_cross=0.8, p_mut=0.2)
            if child is None:
                n_none += 1
            else:
                assert space.validate(child)
        assert n_none < 500  # 不该系统性地全部失败


# ---------------------------------------------------------------- evolutionary_search
class TestEvolutionarySearch:
    def test_all_results_valid_with_unique_keys(self, space, encoder):
        score_pairs, _ = _make_scorer(encoder, seed=0)
        result = evolutionary_search(
            score_pairs, space, encoder, rng=np.random.default_rng(0), **SMALL
        )
        keys = [r["cls_key"] for r in result]
        assert len(keys) == len(set(keys))
        for r in result:
            c = CLS(**r["cls"])
            assert space.validate(c)
            assert cls_key(c) == r["cls_key"]
            encoder.encode(c)  # 不应抛错

    def test_deterministic_given_rng(self, space, encoder):
        score_pairs, _ = _make_scorer(encoder, seed=0)
        r1 = evolutionary_search(
            score_pairs, space, encoder, rng=np.random.default_rng(0), **SMALL
        )
        r2 = evolutionary_search(
            score_pairs, space, encoder, rng=np.random.default_rng(0), **SMALL
        )
        assert r1 == r2

    def test_top_k_improves_over_initial_sample(self, space, encoder):
        # (c): 最终 top-k 里的最优效用 >= 初始 ks 采样里的最优效用，且至少 1/3 个种子严格更优
        strictly_better = 0
        for seed in range(3):
            score_pairs, utility = _make_scorer(encoder, seed=seed)

            init_rng = np.random.default_rng(seed)
            initial = _sample_unique_cls(space, init_rng, SMALL["ks"])
            best_initial = max(utility(c) for c in initial)

            result = evolutionary_search(
                score_pairs, space, encoder, rng=np.random.default_rng(seed), **SMALL
            )
            best_final = max(utility(CLS(**r["cls"])) for r in result)

            assert best_final >= best_initial - 1e-9
            if best_final > best_initial + 1e-9:
                strictly_better += 1
        assert strictly_better >= 1

    def test_top_k_length_and_ordering(self, space, encoder):
        score_pairs, _ = _make_scorer(encoder, seed=5)
        result = evolutionary_search(
            score_pairs, space, encoder, rng=np.random.default_rng(5), **SMALL
        )
        assert len(result) == SMALL["top_k"]
        wins = [r["wins"] for r in result]
        assert wins == sorted(wins, reverse=True)
        for r in result:
            assert set(r.keys()) == {"cls", "cls_key", "wins", "generation_found"}

    def test_generation_found_is_zero_for_initial_population_member(self, space, encoder):
        # 初代（ks 采样 → top-kp）里活到最终 top-k 的候选，generation_found 必须是 0
        score_pairs, utility = _make_scorer(encoder, seed=7)
        rng = np.random.default_rng(7)
        result = evolutionary_search(
            score_pairs, space, encoder, rng=rng, ks=64, kp=8, generations=0, top_k=3,
        )
        # generations=0：没有进化步，top-k 必然全部来自初代
        assert all(r["generation_found"] == 0 for r in result)
