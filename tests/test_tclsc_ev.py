"""EV1 检验（源侧进化搜索验证，results/tclsc_ev/PROTOCOL.md）——先于实现。覆盖：进化模块三个钩子（采样器 / 变异器 / 接受域；缺省逐位不变）、
搜索域判定（到信任域的最小汉明）、切片选择（14 = 7 × 2）、均匀随机对照（确定性 / 域内 / 去重）、进化一个切片（合成适应度：提名 ∉ 信任域、∈ 域、位次与 above_trust 一致、确定性）、
SetTF field 臂 ArmSpec、C(6,3) 子集期望、切片统计 / 配对块 / 门 / 读数码。零 GPU、不读真实缓存。"""
from __future__ import annotations

import importlib
import itertools

import numpy as np
import pytest

from autocls.labeling import cls_key
from autocls.search import evolution as evo
from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLS, CLSSpace
from scripts import pq_common as pq


def _g77():
    return importlib.import_module("scripts.77_tclsc_ev")


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


@pytest.fixture(scope="module")
def encoder(space):
    return CLSEncoder(space)


@pytest.fixture(scope="module")
def trust(space):
    return _g77().trust_entries(space)          # [(CLS, key)] = shared-10 + GGS


def _hamming(a: CLS, b: CLS, fields) -> int:
    return sum(1 for f in fields if getattr(a, f) != getattr(b, f))


def _linear_scorer(encoder, seed):
    w = np.random.default_rng(seed).standard_normal(encoder.dim)

    def score_pairs(va, vb):
        return 1.0 / (1.0 + np.exp(-((va @ w) - (vb @ w))))
    return score_pairs


SMALL = dict(ks=32, kp=8, generations=3, top_k=8)


class TestEvolutionHooks:
    def test_defaults_bit_identical_with_explicit_default_hooks(self, space, encoder):
        sp = _linear_scorer(encoder, 0)
        a = evo.evolutionary_search(sp, space, encoder, rng=np.random.default_rng(1), **SMALL)
        b = evo.evolutionary_search(sp, space, encoder, rng=np.random.default_rng(1), **SMALL,
                                    sampler=lambda rng: space.sample_random(rng), mutator=lambda c, rng: evo._mutate(c, rng, space),
                                    accept=lambda c: True)
        assert [x["cls_key"] for x in a] == [x["cls_key"] for x in b]
        assert [x["wins"] for x in a] == [x["wins"] for x in b]

    def test_domain_hooks_keep_everything_in_domain(self, space, encoder, trust):
        g = _g77()
        sp = _linear_scorer(encoder, 2)
        tcls = [c for c, _k in trust[:3]]
        acc = lambda c: g.in_domain(space, c, tcls, max_h=2)
        out = evo.evolutionary_search(sp, space, encoder, rng=np.random.default_rng(3), **SMALL,
                                      sampler=lambda rng: g.domain_sampler(space, tcls, rng, max_h=2), mutator=g.local_mutator(space), accept=acc,
                                      seed_population=tcls)
        assert len(out) == SMALL["kp"]
        for x in out:
            assert acc(CLS(**x["cls"]))
        again = evo.evolutionary_search(sp, space, encoder, rng=np.random.default_rng(3), **SMALL,
                                        sampler=lambda rng: g.domain_sampler(space, tcls, rng, max_h=2), mutator=g.local_mutator(space), accept=acc,
                                        seed_population=tcls)
        assert [x["cls_key"] for x in out] == [x["cls_key"] for x in again]


class TestDomainAndControls:
    def test_trust_entries_and_min_hamming(self, space, trust):
        g = _g77()
        assert len(trust) == 11 and len({k for _c, k in trust}) == 11 and trust[-1][1] == pq.cls_key(pq.ggs_cls())
        tcls = [c for c, _k in trust]
        for c, _k in trust:
            assert g.min_hamming(c, tcls, space.field_names) == 0 and g.in_domain(space, c, tcls)
        m = g.local_mutator(space)(tcls[0], np.random.default_rng(0))
        assert g.min_hamming(m, tcls, space.field_names) == min(_hamming(m, t, space.field_names) for t in tcls) == 1

    def test_ev_slices(self):
        g = _g77()
        s = g.ev_slices()
        assert len(s) == 14 and len(set(s)) == 14
        assert all(pq.dataset_of(x) in pq.FOLD_TARGETS for x in s)
        for d in pq.FOLD_TARGETS:
            assert f"{d}_00" in s and f"{d}_x00" in s

    def test_uniform_random_control(self, space, trust):
        g = _g77()
        tcls = [c for c, _k in trust]
        taken = {k for _c, k in trust}
        a = g.uniform_random(space, "Handwriting_00", tcls, taken=set(taken), n=3)
        b = g.uniform_random(space, "Handwriting_00", tcls, taken=set(taken), n=3)
        c = g.uniform_random(space, "Handwriting_x00", tcls, taken=set(taken), n=3)
        assert [cls_key(x) for x in a] == [cls_key(x) for x in b] != [cls_key(x) for x in c]
        keys = {cls_key(x) for x in a}
        assert len(keys) == 3 and not (keys & taken)
        for x in a:
            assert space.validate(x) and g.in_domain(space, x, tcls)


class TestEvolveSlice:
    def test_proposals_outside_trust_inside_domain_ranks_consistent(self, space, encoder, trust):
        g = _g77()
        sp = _linear_scorer(encoder, 5)
        cfg = dict(ks=40, kp=10, generations=3)
        res = g.evolve_slice(sp, space, encoder, trust, seed=11, cfg=cfg)
        tkeys = {k for _c, k in trust}
        tcls = [c for c, _k in trust]
        assert len(res["ranking"]) == cfg["kp"] and len(res["proposals"]) <= 3
        ranks = {k: i for i, k in enumerate(res["ranking"])}
        for p in res["proposals"]:
            assert p["cls_key"] not in tkeys and g.in_domain(space, CLS(**p["cls"]), tcls) and space.validate(CLS(**p["cls"]))
            assert ranks[p["cls_key"]] == p["rank"] and 1 <= p["hamming_to_trust"] <= 3
            assert p["above_trust"] == (res["trust_best_rank"] is None or p["rank"] < res["trust_best_rank"])
        assert res == g.evolve_slice(sp, space, encoder, trust, seed=11, cfg=cfg)

    def test_pick_proposals_skips_trust(self):
        g = _g77()
        out = g.pick_proposals(["t1", "a", "t2", "b", "c", "d"], {"t1", "t2"}, k=3)
        assert [(p["cls_key"], p["rank"], p["above_trust"]) for p in out] == [("a", 1, False), ("b", 3, False), ("c", 4, False)]
        out = g.pick_proposals(["a", "b", "t1"], {"t1"}, k=3)
        assert [(p["cls_key"], p["above_trust"]) for p in out] == [("a", True), ("b", True)]


class TestArm:
    def test_es_field_arm_spec(self):
        m33 = importlib.import_module("scripts.33_pq_comparator_loo")
        s = m33.arm_spec("es_field_hard")
        assert (s.task_branch, s.encoder, s.e_source, s.ctx_kind, s.head, s.split) == ("E+ctx", "settf", "fold", "budget", "field", "S1")
        assert "es_field_hard" in m33.E_ARMS and "es_field_hard" in m33.CLI_ARMS


class TestStats:
    def test_subset_expectation_matches_brute_force(self):
        g = _g77()
        vals = [0.5, 0.7, 0.6, 0.4, 0.65, 0.55]
        e = g.subset_expectation(vals, k=3)
        subs = list(itertools.combinations(vals, 3))
        assert e["max"] == pytest.approx(np.mean([max(s) for s in subs])) and e["mean"] == pytest.approx(np.mean([np.mean(s) for s in subs]))

    def test_slice_stats_and_paired_block_and_gates(self):
        g = _g77()
        truth = {"p1": 0.60, "p2": 0.70, "p3": 0.50}
        st = g.slice_stats(truth, ["p1", "p2", "p3"], ref=0.65, delta=0.04)
        assert st["best3"] == 0.70 and st["mean3"] == pytest.approx(0.6) and st["gap"] == pytest.approx(0.05) and st["over_delta"] is True and st["n"] == 3
        blk = g.paired_block({"s1": 0.7, "s2": 0.6, "s3": 0.8}, {"s1": 0.5, "s2": 0.55, "s3": 0.7})
        assert blk["n"] == 3 and blk["delta"] == pytest.approx(0.35 / 3) and blk["wins"] == 3 and 0 <= blk["p"] <= 1
        assert g.gate_v1([{"gap": 0.05, "over_delta": True}] * 3 + [{"gap": -0.01, "over_delta": False}] * 4 + [{"gap": 0.001, "over_delta": False}] * 7)
        assert not g.gate_v1([{"gap": 0.05, "over_delta": True}] * 2 + [{"gap": 0.01, "over_delta": False}] * 12)
        assert g.gate_v2({"delta": 0.012, "p": 0.01}) and not g.gate_v2({"delta": 0.012, "p": 0.2}) and not g.gate_v2({"delta": 0.005, "p": 0.01})
        assert g.gate_v3({"delta": 0.003, "p": 0.04}) and not g.gate_v3({"delta": -0.003, "p": 0.04})

    def test_verdict_codes(self):
        g = _g77()
        v = lambda et, es, ru: g.verdict_ev({"ev_et": dict(zip(("G-V1", "G-V2", "G-V3"), et)), "ev_es": dict(zip(("G-V1", "G-V2", "G-V3"), es)), "rand_u": {"G-V1": ru}})
        assert v((True, True, True), (False, False, False), False) == "EV-SEARCH-WIN"
        assert v((False, False, False), (True, True, True), False) == "EV-SEARCH-WIN"
        assert v((False, True, True), (False, False, False), False) == "EV-GUIDED-FLAT"
        assert v((True, False, False), (False, False, False), False) == "EV-HEADROOM-RANDOM"
        assert v((False, False, False), (False, False, False), True) == "EV-HEADROOM-RANDOM"
        assert v((False, False, False), (False, False, False), False) == "EV-DEAD"
        assert v((False, True, False), (False, False, True), False) == "EV-DEAD"
