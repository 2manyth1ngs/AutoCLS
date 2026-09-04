"""WP7-UCR：任务感知模块（scripts/42_ucr_module.py）纯函数测试——先于实现（铁律 5）。

覆盖：
- 经典分解的趋势/季节强度：正弦 → 季节≈1、斜坡 → 趋势≈1、白噪声 → 两者都小；多通道取均值；
- φ 向量 14 维顺序（40 号 12 维 ⊕ 趋势/季节）；
- 模块：拟合 / 预测形状、冻结存取后预测逐位相同、f̄(c)（跨语料均值）与 dev 分数；
- 孪生 kNN：k=1 取最近数据集的 shared 偏好；
- U1 留一数据集：臂齐全、pooled 先验为训练侧 shared 均值、置换供体 ≠ 自身；
- 门与读数码。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib
from dataclasses import asdict

import numpy as np
import pytest

from autocls.space.cls_space import CLSSpace

g42 = importlib.import_module("scripts.42_ucr_module")
B3 = (60, 120, 290)


# ---------------------------------------------------------------- 分解强度
def test_decomp_strength_sine_ramp_noise():
    rng = np.random.default_rng(0)
    T, n = 240, 6
    t = np.arange(T)
    sine = np.repeat(np.sin(2 * np.pi * t / 24)[None, :, None], n, axis=0) + rng.normal(scale=0.05, size=(n, T, 1))
    ramp = np.repeat((t / T)[None, :, None], n, axis=0) + rng.normal(scale=0.02, size=(n, T, 1))
    noise = rng.normal(size=(n, T, 1))
    s = g42.decomp_strength(sine)
    r = g42.decomp_strength(ramp)
    w = g42.decomp_strength(noise)
    assert s["season_strength"] > 0.9 and s["trend_strength"] < 0.5
    assert r["trend_strength"] > 0.9
    assert w["trend_strength"] < 0.4 and w["season_strength"] < 0.4
    dup = np.concatenate([sine, sine], axis=2)
    assert g42.decomp_strength(dup)["season_strength"] == pytest.approx(s["season_strength"])


def test_phi_cols_order_and_vector():
    assert len(g42.PHI_COLS) == 14 and g42.PHI_COLS[-2:] == ("trend_strength", "season_strength")
    d = {c: float(i) for i, c in enumerate(g42.PHI_COLS)}
    v = g42.phi_vec(d)
    assert v.shape == (14,) and v[0] == 0.0 and v[-1] == 13.0


# ---------------------------------------------------------------- 合成语料
def make_corpus(rng, datasets, shared_keys, n_random=4, budgets=B3, effect=None):
    """rows：每数据集 shared + random × 3 预算；phi：每数据集 14 维；effect(d, cls) 可注入交互。"""
    space = CLSSpace.from_yaml()
    shared_cls = {k: asdict(space.sample_random(rng)) for k in shared_keys}
    rows, phi = [], {}
    for d in datasets:
        tid = f"{d}_00"
        phi[tid] = {c: float(rng.normal()) for c in g42.PHI_COLS}
        phi[tid]["dataset"] = d
        cats = [(k, "shared", shared_cls[k]) for k in shared_keys]
        cats += [(f"r{d}{j}", "random", asdict(space.sample_random(rng))) for j in range(n_random)]
        for k, origin, cls in cats:
            for b in budgets:
                r = float(rng.uniform(0.3, 0.9)) if effect is None else float(effect(d, k, cls, phi[tid]))
                rows.append({"dataset": d, "task_id": tid, "cls_key": k, "origin": origin, "iters": b,
                             "r_prime": r, "seconds": 1.0, "cls": cls})
    return rows, phi


def test_module_fit_predict_freeze_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    rows, phi = make_corpus(rng, [f"D{i}" for i in range(6)], [f"s{i}" for i in range(5)])
    m = g42.UcrModule(kind="hgb").fit(rows, phi)
    test = [r for r in rows if r["dataset"] == "D0"][:10]
    p1 = m.predict(test, phi)
    assert p1.shape == (10,)
    path = tmp_path / "module.pkl"
    m.save(path)
    m2 = g42.UcrModule.load(path)
    assert np.array_equal(m2.predict(test, phi), p1)
    assert m2.cols == g42.PHI_COLS and m2.n_datasets == 6
    # f̄(c)：对同一批候选行，跨语料 φ 的均值预测；dev = f(φ_t) − f̄
    fbar = m2.mean_prediction(test)
    assert fbar.shape == (10,)
    dev = m2.dev_scores(test, phi[test[0]["task_id"]])
    assert np.allclose(dev, p1 - fbar)


def test_twin_knn_scores_nearest_dataset_profile():
    prof = {"A": {"s1": 1.0, "s2": -1.0}, "B": {"s1": -2.0, "s2": 2.0}, "C": {"s1": 0.5, "s2": 0.5}}
    phi = {"A": np.array([0.0, 0.0]), "B": np.array([10.0, 10.0]), "C": np.array([5.0, 5.0])}
    q = np.array([0.5, -0.5])
    sc = g42.twin_knn_scores(phi, prof, q, ["s1", "s2"], k=1)
    assert sc == {"s1": 1.0, "s2": -1.0}
    sc2 = g42.twin_knn_scores(phi, prof, q, ["s1", "s2"], k=2)
    assert sc2["s1"] == pytest.approx(0.75) and sc2["s2"] == pytest.approx(-0.25)


def test_lodo_harness_arms_and_shuffle_donors():
    rng = np.random.default_rng(2)
    ds = [f"D{i}" for i in range(5)]
    rows, phi = make_corpus(rng, ds, [f"s{i}" for i in range(5)], n_random=3)
    out = g42.lodo_ucr(rows, phi, n_shuffle=2, seed=0, folds=ds[:2])
    inst = out["per_instance"]
    assert len(inst) == 2 * len(B3)
    arms = set(inst[0]["metrics"])
    assert {"pooled", "hgb", "ridge_int", "twin_knn", "hgb_shuffle_0", "hgb_shuffle_1"} <= arms
    for r in inst:
        for a, m in r["metrics"].items():
            assert "regret1" in m and "regret3" in m and "oracle_in_top" in m
    for d, donors in out["meta"]["shuffle_donors"].items():
        assert all(x != d for x in donors)


def test_lodo_learns_injected_interaction():
    """交互可学时：hgb 的 regret@3 明显低于 pooled（合成：random 候选的分随 φ[0] 的符号翻转）。"""
    rng = np.random.default_rng(3)
    ds = [f"D{i}" for i in range(24)]
    shared = [f"s{i}" for i in range(6)]

    def effect(d, k, cls, ph):
        sign = 1.0 if ph["log10_T"] > 0 else -1.0
        base = 0.6 + 0.02 * int(k[1:]) if k.startswith("s") else 0.6
        inter = 0.25 * sign * (1.0 if cls["temperature"] < 1.0 else -1.0)
        return base + inter + rng.normal(scale=0.01)

    rows, phi = make_corpus(rng, ds, shared, n_random=6, effect=effect)
    out = g42.lodo_ucr(rows, phi, n_shuffle=1, seed=0)
    s = out["summary"]["pooled_290"]
    assert s["hgb"]["regret3"] < s["pooled"]["regret3"] - 0.02
    assert s["hgb"]["regret3"] < s["hgb_shuffle_0"]["regret3"]


def test_gate_u1_and_verdict():
    g = g42.gate_u1({"hgb_vs_pooled": {"p": 0.01, "mean_a": 0.02, "mean_b": 0.04},
                     "hgb_vs_shuffle_mean": {"p": 0.02, "mean_a": 0.02, "mean_b": 0.035}})
    assert g["pass"] is True
    g2 = g42.gate_u1({"hgb_vs_pooled": {"p": 0.2, "mean_a": 0.04, "mean_b": 0.04},
                      "hgb_vs_shuffle_mean": {"p": 0.5, "mean_a": 0.04, "mean_b": 0.04}})
    assert g2["pass"] is False
    assert g42.verdict_code({"U1": {"pass": False}}) == "UCR-U1-DEAD"
    assert g42.verdict_code({"U1": {"pass": True}, "A1": {"pass": False}, "A2": {"pass": True}}) == "UCR-U1-ONLY"
    assert g42.verdict_code({"U1": {"pass": True}, "A1": {"pass": True}, "A2": {"pass": True}}) == "UCR-TF-WIN"
