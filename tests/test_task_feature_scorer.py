"""MX-4 WP6-TF：任务特征打分器（scripts/40_task_feature_scorer.py）阶段 0 测试——先于实现（铁律 5）。

计划：~/.claude/plans/ts2vec-e-quirky-bubble.md。覆盖纯函数：
- 现行打分器 s0：shared 取先验 z 均值、random 取任务盲 GBDT；hybrid = s0 + 偏置，与 36 号逐位同排名；
- 名单指标 list_metrics：oracle 入榜、regret@1/@3、前 N 内最好、SH 路径 regret 与成本（6-3-3 = 1050，8-3-3 = 1170）；
- C0 复现比对：与 WP1 per_instance 逐实例一致 → pass，扰动 1e-6 → fail 并指出实例；
- oracle 漏网表：按预算 / 来源 / 数据集计数；
- C1 上界：shared 分 = 同数据集其余切片 z 均值（不含自身、不含其他数据集），身份 one-hot 在特征尾部；
- 0b 交互定位：跨数据集翻转效应的字段 ratio 最大；先验一致 → Spearman 1；
- 嵌套方差分解：纯数据集级交互 → 交互全在数据集级；噪声方差按格内 ddof=1。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib
from dataclasses import asdict

import numpy as np
import pytest

from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLSSpace

g36 = importlib.import_module("scripts.36_prior_sh_sim")
g40 = importlib.import_module("scripts.40_task_feature_scorer")

B3 = (60, 120, 290)


# ---------------------------------------------------------------- 合成数据
def make_rows(rng, datasets, n_slices, shared_keys, n_random, budgets=B3, z_fn=None):
    """合成 rprime 行：shared 跨切片同 CLS，random 每切片独立；z_fn(dataset, slice_i, key, cls) 可定 r_prime。"""
    space = CLSSpace.from_yaml()
    shared_cls = {k: asdict(space.sample_random(rng)) for k in shared_keys}
    rows = []
    for d in datasets:
        for si in range(n_slices):
            tid = f"{d}_{si:02d}"
            cats = [(k, "shared", shared_cls[k]) for k in shared_keys]
            for j in range(n_random):
                cats.append((f"r{d}{si}{j}", "random", asdict(space.sample_random(rng))))
            for k, origin, cls in cats:
                for b in budgets:
                    r = float(rng.uniform(0.2, 0.9)) if z_fn is None else float(z_fn(d, si, k, cls))
                    rows.append({"dataset": d, "task_id": tid, "cls_key": k, "origin": origin,
                                 "iters": b, "r_prime": r, "seconds": 0.1 * b, "cls": cls})
    return rows


class _StubBlind:
    """predict = 行序号 × 0.01（可辨识、确定）。"""

    def __init__(self, base: float = 0.0):
        self.base = base

    def predict(self, X):
        X = np.asarray(X)
        v = self.base + 0.01 * np.arange(X.shape[0])
        return v, np.zeros_like(v)


# ---------------------------------------------------------------- s0 / hybrid
def test_s0_scores_shared_prior_random_blind_and_hybrid_matches_g36():
    rng = np.random.default_rng(0)
    rows = make_rows(rng, ["A"], 1, ["s1", "s2"], 3, budgets=(60,))
    space = CLSSpace.from_yaml()
    enc = CLSEncoder(space)
    prior = {"s1": 0.5, "s2": -0.2}
    blind = _StubBlind(base=1.0)
    sc = g40.s0_scores(rows, prior, blind, enc)
    keys = sorted(r["cls_key"] for r in rows)
    assert sc["s1"] == 0.5 and sc["s2"] == -0.2
    # random 行按 rows 顺序取 stub 预测
    for i, r in enumerate(rows):
        if r["origin"] == "random":
            assert sc[r["cls_key"]] == pytest.approx(1.0 + 0.01 * i)
    hyb = g40.with_bias(sc, rows)
    assert hyb["s1"] == pytest.approx(0.5 + g36.HYBRID_BIAS)
    # 与 36 号 hybrid 构造逐位同排名
    gbdt = {r["cls_key"]: sc[r["cls_key"]] for r in rows}
    ref = {k: (prior[k] + g36.HYBRID_BIAS if k in prior else gbdt[k]) for k in keys}
    assert g36.rank_by(hyb, keys) == g36.rank_by(ref, keys)
    assert g36.rank_by(hyb, keys)[:2] == ["s1", "s2"]


# ---------------------------------------------------------------- list_metrics
TRAJ = {
    "p": {60: 0.90, 120: 0.60, 290: 0.50},
    "q": {60: 0.80, 120: 0.85, 290: 0.95},
    "r": {60: 0.70, 120: 0.80, 290: 0.90},
    "s": {60: 0.10, 120: 0.20, 290: 0.30},
    "t": {60: 0.75, 120: 0.70, 290: 0.60},
}


def test_list_metrics_definitions_match_36_and_37():
    keys = list(TRAJ)
    truth = {k: TRAJ[k][290] for k in keys}
    ranked = ["p", "t", "r", "q", "s"]
    m = g40.list_metrics(keys, truth, TRAJ, B3, ranked, top_n=2, k_top=2, sh_ks=(3, 2, 1))
    assert m["oracle_in_top"] is False                       # oracle q 不在前 2
    assert m["regret1"] == pytest.approx(0.95 - 0.50)
    assert m["regret3"] == pytest.approx(0.95 - max(0.50, 0.60))
    assert m["regret_top"] == pytest.approx(0.95 - 0.60)      # 前 N 内最好
    path = g36.sh_path(ranked, TRAJ, (3, 2, 1), B3)
    pick = g36.sh_pick(path[-1], TRAJ, 290)
    assert m["sh_pick"] == pick and m["sh_regret"] == pytest.approx(0.95 - TRAJ[pick][290])
    assert m["sh_cost"] == g36.path_cost_iters(path, B3)
    m60 = g40.list_metrics(keys, {k: TRAJ[k][60] for k in keys}, TRAJ, (60,), ranked, top_n=2, k_top=2)
    assert m60["sh_regret"] is None and m60["sh_cost"] is None


def test_sh_costs_633_is_1050_and_833_is_1170():
    rng = np.random.default_rng(1)
    keys = [f"k{i:02d}" for i in range(20)]
    traj = {k: {b: float(rng.uniform()) for b in B3} for k in keys}
    truth = {k: traj[k][290] for k in keys}
    m6 = g40.list_metrics(keys, truth, traj, B3, keys, sh_ks=(6, 3, 3))
    m8 = g40.list_metrics(keys, truth, traj, B3, keys, sh_ks=(8, 3, 3))
    assert m6["sh_cost"] == 1050 and m8["sh_cost"] == 1170


def test_oracle_key_tie_breaks_by_key_ascending():
    assert g40.oracle_key({"b": 0.9, "a": 0.9, "c": 0.1}) == "a"


# ---------------------------------------------------------------- C0
def _fake_instances():
    inst = []
    for b in B3:
        inst.append({"fold": "A", "dataset": "A", "slice": "A_00", "budget": b,
                     "metrics": {"prior": {"regret3": 0.01, "sh_regret": 0.02 if b > 60 else None},
                                 "gbdt": {"regret3": 0.03, "sh_regret": None},
                                 "hybrid": {"regret3": 0.01, "sh_regret": None}}})
    return inst


def _fake_wp1(inst):
    per = []
    for r in inst:
        reg = {"prior3": r["metrics"]["prior"]["regret3"], "gbdt3": r["metrics"]["gbdt"]["regret3"],
               "hybrid3": r["metrics"]["hybrid"]["regret3"]}
        if r["budget"] == 290:
            reg["SH6-3-3_prior"] = r["metrics"]["prior"]["sh_regret"]
        per.append({"fold": r["fold"], "slice": r["slice"], "budget": r["budget"], "regret": reg})
    return {"per_instance": per, "by_budget": {"290": {"mean_regret": {"SH6-3-3_prior": 0.02}}}}


def test_c0_compare_pass_then_fail_on_perturbation():
    inst = _fake_instances()
    wp1 = _fake_wp1(inst)
    ok = g40.c0_compare(inst, wp1)
    assert ok["pass"] and ok["n_compared"] == 3 and max(ok["max_abs"].values()) == 0.0
    inst[2]["metrics"]["prior"]["sh_regret"] += 1e-6
    bad = g40.c0_compare(inst, wp1)
    assert not bad["pass"]
    assert bad["mismatches"][0]["slice"] == "A_00" and bad["mismatches"][0]["arm"] == "SH6-3-3_prior"


# ---------------------------------------------------------------- oracle 漏网表
def test_oracle_miss_table_counts_by_budget_origin_dataset():
    inst = [
        {"dataset": "A", "budget": 290, "oracle_origin": "random", "metrics": {"hybrid": {"oracle_in_top": False, "regret_top": 0.05}}},
        {"dataset": "A", "budget": 290, "oracle_origin": "shared", "metrics": {"hybrid": {"oracle_in_top": True, "regret_top": 0.0}}},
        {"dataset": "B", "budget": 290, "oracle_origin": "shared", "metrics": {"hybrid": {"oracle_in_top": False, "regret_top": 0.01}}},
        {"dataset": "B", "budget": 60, "oracle_origin": "random", "metrics": {"hybrid": {"oracle_in_top": False, "regret_top": 0.02}}},
    ]
    t = g40.oracle_miss_table(inst, arm="hybrid")
    assert t["290"]["n"] == 3 and t["290"]["n_miss"] == 2
    assert t["290"]["by_origin"] == {"random": 1, "shared": 1}
    assert t["290"]["by_dataset"] == {"A": 1, "B": 1}
    assert t["290"]["regret_top_mean"] == pytest.approx((0.05 + 0.0 + 0.01) / 3)
    assert t["60"]["n_miss"] == 1 and t["60"]["by_origin"] == {"random": 1}


# ---------------------------------------------------------------- C1 上界
def test_identity_features_onehot_at_tail():
    rng = np.random.default_rng(2)
    rows = make_rows(rng, ["A", "B"], 1, ["s1"], 1, budgets=(60,))
    enc = CLSEncoder(CLSSpace.from_yaml())
    X = g40.identity_features(rows, enc, ("A", "B", "C"))
    base = g36.features(rows, enc, max(B3))
    assert X.shape == (len(rows), base.shape[1] + 3)
    assert np.array_equal(X[:, :base.shape[1]], base)
    for i, r in enumerate(rows):
        assert X[i, -3:].tolist() == [1.0 if r["dataset"] == d else 0.0 for d in ("A", "B", "C")]


def test_ceiling_shared_score_is_mean_over_other_slices_of_same_dataset_only():
    rng = np.random.default_rng(3)
    rows = make_rows(rng, ["A", "B"], 3, ["s1", "s2"], 2)
    # 手工 z：s1 在 A 的三个切片 z = 切片号 (0,1,2)，在 B 全为 9（不得混入）
    for r in rows:
        si = int(r["task_id"].split("_")[1])
        r["z"] = float(si) if r["dataset"] == "A" else 9.0
    enc = CLSEncoder(CLSSpace.from_yaml())
    seen = {}

    def fit_stub(train_rows):
        seen["train_slices"] = sorted({r["task_id"] for r in train_rows})
        return _StubBlind()

    sc = g40.ceiling_scores("A", "A_00", rows, enc, ("A", "B"), seed=0, fit_identity=fit_stub)
    assert set(sc) == set(B3)
    assert sc[60]["s1"] == pytest.approx(np.mean([1.0, 2.0]))     # 只用 A_01/A_02
    assert "A_00" not in seen["train_slices"] and "B_00" in seen["train_slices"]
    # random 键来自 stub（存在且为浮点）
    rand_keys = [r["cls_key"] for r in rows if r["task_id"] == "A_00" and r["origin"] == "random"]
    assert all(isinstance(sc[60][k], float) for k in rand_keys)


# ---------------------------------------------------------------- 0b 交互定位
def test_field_variance_ratio_flags_field_whose_effect_flips_across_datasets():
    rng = np.random.default_rng(4)
    space = CLSSpace.from_yaml()
    temps = space.field_spec("temperature")["choices"]
    datasets = [f"D{i}" for i in range(8)]
    sign = {d: (1.0 if i % 2 == 0 else -1.0) for i, d in enumerate(datasets)}

    def z_fn(d, si, key, cls):
        t_idx = temps.index(cls["temperature"]) / max(1, len(temps) - 1)
        return sign[d] * (t_idx - 0.5) + rng.normal(scale=0.05)

    rows = make_rows(rng, datasets, 3, ["s1", "s2"], 18, budgets=(290,), z_fn=z_fn)
    for r in rows:
        r["z"] = r["r_prime"]
    enc = CLSEncoder(space, mode="binned")
    out = g40.field_variance_ratio(rows, enc, space, n_boot=4, seed=0)
    assert out["rank"][0] == "temperature" and "temperature" in out["top4"]
    assert out["ratio"]["temperature"] > 3 * np.median(list(out["ratio"].values()))


def test_dataset_prior_spearman_is_one_when_pattern_identical():
    rng = np.random.default_rng(5)
    shared = [f"s{i}" for i in range(5)]
    rows = make_rows(rng, ["A", "B", "C"], 2, shared, 1, budgets=(60,))
    for r in rows:
        r["z"] = float(shared.index(r["cls_key"])) if r["origin"] == "shared" else -1.0
    rho = g40.dataset_prior_spearman(rows)
    assert set(rho) == {"A", "B", "C"} and all(v == pytest.approx(1.0) for v in rho.values())


# ---------------------------------------------------------------- 方差分解与噪声
def test_nested_decomposition_pure_dataset_level_interaction():
    inter = np.array([[1.0, -1.0, 0.5, -0.5], [-1.0, 1.0, -0.5, 0.5]])   # 零和交互，2 数据集 × 4 CLS
    groups = ["A", "A", "A", "B", "B", "B"]
    M = np.stack([inter[0 if g == "A" else 1] for g in groups])
    out = g40.nested_decomposition(M, groups)
    assert out["share_task"] == pytest.approx(0.0, abs=1e-12)
    assert out["share_cls"] == pytest.approx(0.0, abs=1e-12)
    assert out["share_inter"] == pytest.approx(1.0)
    assert out["share_inter_dataset"] == pytest.approx(1.0)
    assert out["share_inter_slice"] == pytest.approx(0.0, abs=1e-12)
    assert out["df_inter_dataset"] == 1 * 3 and out["df_inter_slice"] == (6 - 2) * 3


def test_nested_decomposition_shares_sum_to_one_on_random_matrix():
    rng = np.random.default_rng(6)
    M = rng.normal(size=(9, 5))
    out = g40.nested_decomposition(M, ["A"] * 3 + ["B"] * 3 + ["C"] * 3)
    assert out["share_task"] + out["share_cls"] + out["share_inter"] == pytest.approx(1.0)
    assert out["share_inter_dataset"] + out["share_inter_slice"] == pytest.approx(1.0)


def test_noise_variance_cellwise_ddof1():
    cat = [{"task_id": "A_00", "cls_key": "s1", "iters": 290, "r_prime": 0.5, "seed": 0},
           {"task_id": "A_00", "cls_key": "s2", "iters": 290, "r_prime": 0.7, "seed": 0}]
    probe = [{"task_id": "A_00", "cls_key": "s1", "iters": 290, "r_prime": 0.6, "seed": 1},
             {"task_id": "A_00", "cls_key": "s1", "iters": 290, "r_prime": 0.7, "seed": 2},
             {"task_id": "A_00", "cls_key": "s2", "iters": 60, "r_prime": 0.1, "seed": 1}]
    out = g40.noise_variance(cat, probe, 290)
    assert out["n_cells"] == 1
    assert out["sigma2"] == pytest.approx(np.var([0.5, 0.6, 0.7], ddof=1))


# ---------------------------------------------------------------- 折驱动（真 HGB，小数据）
def test_fold_s0_instances_has_all_arms_and_consistent_metrics():
    rng = np.random.default_rng(7)
    rows = make_rows(rng, ["A", "B", "C"], 2, [f"s{i}" for i in range(4)], 4)
    cat_z = g36.zscore_within(rows)
    enc = CLSEncoder(CLSSpace.from_yaml())
    inst = g40.fold_s0_instances("A", cat_z, enc, seed=0)
    assert len(inst) == 2 * len(B3)
    for r in inst:
        assert set(r["metrics"]) >= {"prior", "gbdt", "hybrid", "s0z"}
        assert r["oracle_origin"] in ("shared", "random")
        # hybrid 把 shared 置顶：前 4（= shared 数）与 prior 相同（真数据 shared=10 ≥ N，前 6 全同）
        assert r["metrics"]["hybrid"]["top"][:4] == r["metrics"]["prior"]["top"][:4]
        assert all(k.startswith("s") for k in r["metrics"]["hybrid"]["top"][:4])
        if r["budget"] == 290:
            assert r["metrics"]["prior"]["sh_cost"] == g36.path_cost_iters(
                g36.sh_path(r["ranks"]["prior"], r["traj"], (6, 3, 3), B3), B3)
