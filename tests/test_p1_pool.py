"""WP9-P1 路径一（scripts/46_p1_pool.py + configs/p1.yaml）测试——先于实现（铁律 5）。

协议 results/mx4_p1/PROTOCOL.md。覆盖：配置真源与 35/uea30 一致；候选池采样前缀稳定且与已见键零交集；工单键唯一、
与 WP3 键零碰撞；三源合池 options 硬断言与重键去重；随机搜索解析期望 / 分位；随机起点 SH 与 36 号逐位；Hyperband
η=2 三支架成本手算（含 UCR rung）；离散 TPE 在起点后取 l/g 最大者、门控 None 字段不入乘积；我们的臂定义与成本
（SH6-3-3 与 39.deploy_arms 逐位）；门 E1–E3 四分支；finals 复用去重；UCR 臂成本；消融汇总行。
零 GPU；真实产物金标在缺文件时 skip。
"""
from __future__ import annotations

import importlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from autocls.labeling import cls_key
from autocls.space.cls_space import CLSSpace

g46 = importlib.import_module("scripts.46_p1_pool")
g39 = g46._g39
g36 = g46._g36
g35 = g46._g35
ROOT = Path(__file__).resolve().parents[1]
SIX = ("Handwriting", "HandMovementDirection", "UWaveGestureLibrary", "Cricket", "EthanolConcentration", "PhonemeSpectra")
_PROTOS = {"ts2vec_official": {"iters_rule": "ts2vec_size", "batch_size": 8, "lr": 1e-3, "drop_last": True, "swa": True,
                               "crop_mode": "fixed", "scaler": "train", "svm_grid": "ts2vec"}}
TRAJ = {  # tests/test_prior_sh_sim.py 的公共夹具语义：p 早强晚崩、q 晚熟冠军、r 稳、s 垫底、t 中衰
    "p": {50: 0.90, 100: 0.60, 200: 0.50}, "q": {50: 0.80, 100: 0.85, 200: 0.95}, "r": {50: 0.70, 100: 0.80, 200: 0.90},
    "s": {50: 0.10, 100: 0.20, 200: 0.30}, "t": {50: 0.75, 100: 0.70, 200: 0.60}, "g": {50: 0.60, 100: 0.65, 200: 0.70},
}


@pytest.fixture(scope="module")
def cfg():
    return g46.load_cfg()


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


def _row(dataset, key, accs, *, arm="shared", stage="s1_catalog", seed=0, budgets=(50, 100, 200), error=None, options=None,
         extra=None):
    row = {"dataset": dataset, "cls_key": key, "seed": seed, "stage": stage, "arm": arm, "error": error,
           "protocol": "ts2vec_official", "iters": budgets[-1], "checkpoints": list(budgets),
           "options": dict(options or {"batch_size": 8, "input_mask": "none"}), "acc": (accs[-1] if accs else None),
           "checkpoint_results": ([{"iters": b, "acc": a, "elapsed_pretrain_s": 1.0 * i, "eval_seconds": 0.1}
                                   for i, (b, a) in enumerate(zip(budgets, accs))] if accs else None),
           "cls": {}}
    row.update(extra or {})
    return row


def _traj100(n=100, seed=0):
    """合成 100 条轨迹：终值均匀分布，早截点带噪声；返回 (keys, traj, secs, cls_of)。"""
    rng = np.random.default_rng(seed)
    keys = [f"k{i:03d}" for i in range(n)]
    traj, secs = {}, {}
    for i, k in enumerate(keys):
        v = 0.3 + 0.6 * i / (n - 1)
        traj[k] = {50: round(v - 0.2 + rng.normal(scale=0.05), 6), 100: round(v - 0.1 + rng.normal(scale=0.03), 6), 200: round(v, 6)}
        secs[k] = {50: 1.0, 100: 2.0, 200: 4.0}
    return keys, traj, secs


# ---------------------------------------------------------------- 配置真源
def test_cfg_sizes_iters_match_35_and_uea30(cfg):
    import yaml
    u = yaml.safe_load((ROOT / "configs" / "uea30.yaml").read_text(encoding="utf-8"))["datasets"]
    assert tuple(g46.run_targets(cfg)) == SIX                       # order 0..5
    for d in SIX:
        c = cfg["targets"][d]
        assert (c["n_train"], c["L"], c["C"]) == tuple(g35.SIZES_SURVEY[d]), d
        assert (c["n_train"], c["L"], c["C"], c["n_test"]) == (u[d]["n_train"], u[d]["L"], u[d]["C"], u[d]["n_test"]), d
        assert c["iters"] == g35.resolve_iters_from_size(_PROTOS["ts2vec_official"], c["n_train"] * c["L"] * c["C"]), d
        assert c["nominee_idx"] == list(g35.TARGETS).index(d), d
        assert c["anchor_4090"]["cell_s"] > 0 and c["anchor_4090"]["cell_max_s"] >= c["anchor_4090"]["cell_s"]
    assert g46.sizes_from_cfg(cfg)["Handwriting"] == (150, 152, 3)
    assert g46.budgets_for(cfg, "Handwriting") == [50, 100, 200] and g46.budgets_for(cfg, "Cricket") == [150, 300, 600]
    assert cfg["arms"]["main"] == g46.MAIN_ARM == "SH10-4-3_prior"
    assert cfg["pool"]["n_target"] == 100 and cfg["sim"]["n_draws"] == 200


# ---------------------------------------------------------------- 候选池采样
def test_sample_random_entries_prefix_stable_and_disjoint(space):
    taken = {"deadbeef0001"}
    a = g46.sample_random_entries(space, seed=4500, n=5, taken=taken)
    b = g46.sample_random_entries(space, seed=4500, n=8, taken=taken)
    assert [e["cls_key"] for e in b[:5]] == [e["cls_key"] for e in a]
    assert [e["random_idx"] for e in b] == list(range(8))
    keys = [e["cls_key"] for e in b]
    assert len(set(keys)) == 8 and not (set(keys) & taken)
    for e in b:
        assert space.validate(g46.CLS(**e["cls"])) and e["cls"].get("input_mask", "none") == "none"
        assert cls_key(g46.CLS(**e["cls"])) == e["cls_key"]
    # taken 含某条采样结果 → 该条被跳过，其余前缀不变
    taken2 = taken | {keys[1]}
    c = g46.sample_random_entries(space, seed=4500, n=7, taken=taken2)
    assert [e["cls_key"] for e in c] == [k for k in keys if k != keys[1]][:7]


def test_orders_prior_hybrid_surrogate():
    keys = ["a", "b", "c", "d", "e"]
    prior = {"a": 0.5, "b": 1.0}                                    # 只有 shared/GGS 有先验
    pred = {"a": 0.1, "b": 0.2, "c": 0.9, "d": 0.3, "e": 0.8}
    po, ho, so = g46.pool_orders(keys, prior, pred, prior_keys=["a", "b"])
    assert po == ["b", "a", "c", "d", "e"]                          # 先验降序，其余按 key 垫底
    assert ho == ["b", "a", "c", "e", "d"]                          # 先验置顶 + 其余按 pred_z
    assert so == ["c", "e", "d", "b", "a"]                          # 全部按 pred_z


def test_p1_worklist_keys_unique_no_collision(space):
    rng = np.random.default_rng(0)
    shared = [space.sample_random(rng) for _ in range(3)]
    ggs = space.sample_random(rng)
    cands = {"Handwriting": {"shared": [{"cls_key": cls_key(c), "cls": asdict(c)} for c in shared],
                             "ggs": {"cls_key": cls_key(ggs), "cls": asdict(ggs)}, "nominated": []}}
    sizes = {"Handwriting": (150, 152, 3)}
    wp3 = g39.s1_worklist(_PROTOS, sizes, cands)
    taken = {c["cls_key"] for c in wp3}
    ents = g46.sample_random_entries(space, seed=4500, n=4, taken=taken)
    pool = {"targets": {"Handwriting": {"entries": [{**e, "origin": "random_space", "source": "new", "error": False} for e in ents]
                                        + [{"cls_key": cands["Handwriting"]["shared"][0]["cls_key"], "cls": cands["Handwriting"]["shared"][0]["cls"],
                                            "origin": "shared", "source": "wp3", "error": False}]}}}
    cells = g46.p1_worklist(_PROTOS, sizes, pool)
    assert len(cells) == 4 and {c["arm"] for c in cells} == {"random_space"} and {c["stage"] for c in cells} == {"s1_pool"}
    assert all(c["options"]["input_mask"] == "none" and c["checkpoints"] == list(g39.checkpoints_for(c["iters"])) for c in cells)
    assert len({c["key"] for c in cells}) == 4 and not ({c["key"] for c in cells} & {c["key"] for c in wp3})
    assert all(c["cls_key"] not in taken for c in cells)


# ---------------------------------------------------------------- 三源合池
def test_load_pool_rows_asserts_same_options_and_dedups():
    wp3 = [_row("X", "k1", [0.1, 0.2, 0.3]), _row("X", "k2", [0.1, 0.2, 0.4], error="boom")]
    wp3[1]["checkpoint_results"] = None
    wp8 = [_row("X", "k3", [0.2, 0.3, 0.5], arm="bo_ucb_r1", stage="bo_round", extra={"bo_arm": "bo_ucb", "round": 1}),
           _row("X", "k3", [0.2, 0.3, 0.6], arm="rand_nbr_r2", stage="bo_round", extra={"bo_arm": "rand_nbr", "round": 2})]
    new = [_row("X", "k4", [0.3, 0.4, 0.7], arm="random_space", stage="s1_pool")]
    out = g46.load_pool_rows("X", wp3, wp8, new)
    assert out["seen_keys"] == {"k1", "k2", "k3", "k4"} and out["error_keys"] == {"k2"}
    got = {r["cls_key"]: r for r in out["rows"]}
    assert got["k3"]["arm"] == "bo_ucb_r1" and out["duplicates"] == [{"cls_key": "k3", "kept": "bo_ucb_r1", "dropped": ["rand_nbr_r2"],
                                                                        "acc_final": [0.5, 0.6]}]
    assert out["source_of"]["k4"] == "new" and out["source_of"]["k1"] == "wp3"
    bad = [_row("X", "k9", [0.1, 0.2, 0.3], options={"batch_size": 16, "input_mask": "none"})]
    with pytest.raises(SystemExit):
        g46.load_pool_rows("X", wp3, bad, [])


# ---------------------------------------------------------------- 随机搜索（解析）
def test_random_search_analytic():
    vals = [0.1, 0.4, 0.2, 0.9, 0.5, 0.7]
    for k in (1, 2, 3, 6):
        r = g46.random_search_expect(vals, k, beat=0.55)
        combos = [max(c) for c in itertools.combinations(vals, k)]
        assert r["mean"] == pytest.approx(g46.expected_max_of_k(vals, k)) == pytest.approx(np.mean(combos))
        assert r["median"] == pytest.approx(np.median(combos)) and r["q25"] == pytest.approx(np.quantile(combos, 0.25))
        assert r["p_beat"] == pytest.approx(np.mean([c > 0.55 for c in combos]))
    assert g46.random_search_expect(vals, 1)["mean"] == pytest.approx(np.mean(vals))


# ---------------------------------------------------------------- 随机起点 SH
def test_rand_sh_draws_mean_equals_36():
    keys = list(TRAJ)
    arr = g46.rand_sh_draws(keys, TRAJ, (4, 2, 1), (50, 100, 200), n_draws=30, seed=7)
    assert arr.shape == (30,) and arr.mean() == pytest.approx(g36.expected_random_sh(keys, TRAJ, (4, 2, 1), (50, 100, 200), 30, 7))
    s = g46.summarize_draws(arr, beat=0.9)
    assert set(s) >= {"mean", "median", "q25", "q75", "p_beat"} and 0.0 <= s["p_beat"] <= 1.0
    assert g46.sh_cost_full((4, 2, 1), (50, 100, 200)) == pytest.approx((4 * 50 + 2 * 50 + 1 * 100) / 200)
    assert g46.sh_cost_full((10, 4, 3), (150, 300, 600)) == pytest.approx(5.0)
    assert g46.sh_cost_full((12, 6, 3), (150, 300, 600)) == pytest.approx(6.0)


# ---------------------------------------------------------------- Hyperband
def test_hyperband_bracket_cost_hand_computed():
    b = (50, 100, 200)
    assert g46.hb_bracket_plan(2, b) == ((4, 2, 1), (50, 100, 200))
    assert g46.hb_bracket_plan(1, b) == ((3, 1), (100, 200))
    assert g46.hb_bracket_plan(0, b) == ((3,), (200,))
    assert g46.hb_cost_iters(b, [2, 1, 0]) == 4 * 50 + 2 * 50 + 1 * 100 + 3 * 100 + 1 * 100 + 3 * 200 == 1400
    assert g46.hb_cost_iters(b, [2, 0]) == 1000 and g46.hb_cost_iters(b, [2, 1]) == 800
    assert g46.hb_cost_iters(b, [2, 1, 0, 2, 1, 0]) == 2800
    assert g46.hb_cost_iters((60, 120, 290), [2, 1, 0]) == 1930
    assert g46.hb_n_configs([2, 1, 0]) == 10 and g46.hb_n_configs([2, 0]) == 7
    keys, traj, _ = _traj100(20)
    arr, picks = g46.hyperband_draws(keys, traj, b, [2, 1, 0], n_draws=25, seed=3)
    assert arr.shape == (25,) and all(traj[p][200] == a for p, a in zip(picks, arr))
    assert all(len(set(s)) == 10 for s in g46.hyperband_samples(keys, [2, 1, 0], n_draws=25, seed=3))   # 不放回
    with pytest.raises(SystemExit):
        g46.hyperband_draws(keys[:9], traj, b, [2, 1, 0], n_draws=2, seed=0)


# ---------------------------------------------------------------- TPE 离散池版
def test_tpe_select_argmax_lg_and_gated_none():
    fields = ["f1", "f2", "f3"]
    choices = {"f1": ["a", "b"], "f2": [1, 2, 3], "f3": ["x", "y"]}
    cls_of = {"o1": {"f1": "a", "f2": 1, "f3": "x"}, "o2": {"f1": "b", "f2": 2, "f3": "y"}, "o3": {"f1": "a", "f2": 3, "f3": "y"},
              "c1": {"f1": "a", "f2": 1, "f3": "y"}, "c2": {"f1": "b", "f2": 2, "f3": "x"}, "c3": {"f1": "a", "f2": 1, "f3": None}}
    observed = {"o1": 0.9, "o2": 0.2, "o3": 0.5}                    # γ=1 → good={o1}, bad={o2,o3}
    lg = g46.tpe_ratio(cls_of["c1"], good=["o1"], bad=["o2", "o3"], cls_of=cls_of, fields=fields, choices=choices)
    # l: f1=a (1+1)/(1+2)=2/3, f2=1 2/4, f3=y 1/3；g: f1=a (1+1)/(2+2)=1/2, f2=1 1/5, f3=y 3/4
    assert lg == pytest.approx((2 / 3 * 2 / 4 * 1 / 3) / (1 / 2 * 1 / 5 * 3 / 4))
    lg3 = g46.tpe_ratio(cls_of["c3"], good=["o1"], bad=["o2", "o3"], cls_of=cls_of, fields=fields, choices=choices)
    assert lg3 == pytest.approx((2 / 3 * 2 / 4) / (1 / 2 * 1 / 5))    # f3=None 不入乘积
    assert g46.tpe_gamma(2) == 1 and g46.tpe_gamma(10) == 1 and g46.tpe_gamma(11) == 2 and g46.tpe_gamma(300) == 25
    pick = g46.tpe_select(observed, ["c1", "c2", "c3"], cls_of, fields, choices)
    ratios = {c: g46.tpe_ratio(cls_of[c], good=["o1"], bad=["o2", "o3"], cls_of=cls_of, fields=fields, choices=choices) for c in ("c1", "c2", "c3")}
    assert pick == sorted(ratios, key=lambda c: (-ratios[c], c))[0]
    keys, traj, _ = _traj100(12)
    cls_all = {k: {"f1": choices["f1"][i % 2], "f2": choices["f2"][i % 3], "f3": (None if i % 5 == 0 else choices["f3"][i % 2])} for i, k in enumerate(keys)}
    arr, cost = g46.tpe_draws(keys, cls_all, traj, (50, 100, 200), n=4, fields=fields, choices=choices, n_startup=2, n_draws=10, seed=1)
    assert arr.shape == (10,) and cost == 4 * 200
    assert all(traj[k][200] <= a <= max(t[200] for t in traj.values()) for a in arr for k in [min(traj, key=lambda k: traj[k][200])])


# ---------------------------------------------------------------- 我们的臂
def _pool_d(keys, prior_keys, prior, pred):
    po, ho, so = g46.pool_orders(keys, prior, pred, prior_keys=prior_keys)
    return {"prior_order": po, "hybrid_order": ho, "surrogate_order": so, "prior_keys": list(prior_keys),
            "catalog_keys": list(prior_keys) + ["k020", "k021"], "curated_keys": list(prior_keys) + [f"k{i:03d}" for i in range(20, 60)],
            "random_keys": [f"k{i:03d}" for i in range(60, 100)]}


def test_our_arms_definitions_and_costs():
    keys, traj, secs = _traj100(100)
    prior_keys = keys[:11]
    prior = {k: float(-i) for i, k in enumerate(prior_keys)}         # k000 先验第一
    pred = {k: traj[k][200] for k in keys}                            # 代理 = 真值（理想）
    pool_d = _pool_d(keys, prior_keys, prior, pred)
    a = g46.our_arms(pool_d, traj, secs, (50, 100, 200))
    bf = 200
    assert a["oracle_pool"]["acc"] == pytest.approx(max(t[200] for t in traj.values()))
    assert a["prior1"]["pick"] == "k000" and a["prior1"]["cost_full"] == 1.0 and a["prior3"]["cost_full"] == 3.0
    assert a["SH6-3-3_prior"]["cost_full"] == pytest.approx(3.75) and a["SH10-4-3_prior"]["cost_full"] == pytest.approx(5.0)
    assert a["hybrid16-8-3"]["cost_full"] == pytest.approx(7.5) and a["prior6+sur10-8-3"]["cost_full"] == pytest.approx(7.5)
    assert a["SH16-8-3_surrogate"]["cost_full"] == pytest.approx(7.5)
    assert a["hybrid16-8-3"]["path"][0] == pool_d["hybrid_order"][:16]
    assert a["SH16-8-3_surrogate"]["path"][0] == pool_d["surrogate_order"][:16]
    p6 = a["prior6+sur10-8-3"]["path"][0]
    assert p6[:6] == pool_d["prior_order"][:6] and len(p6) == 16 and len(set(p6)) == 16
    assert a["SH16-8-3_surrogate"]["pick"] == "k099"                  # 理想代理 → 找到池 oracle
    assert a["SH6-3-3_prior"]["regret_pool"] == pytest.approx(a["oracle_pool"]["acc"] - a["SH6-3-3_prior"]["acc"])
    assert a["SH6-3-3_prior"]["cost_seconds"] == pytest.approx(6 * 1.0 + 3 * 1.0 + 3 * 2.0)
    ref = g39.deploy_arms(keys, traj, {"prior": pool_d["prior_order"]}, budgets=(50, 100, 200), random_seed=0, seconds=secs)
    assert a["SH6-3-3_prior"]["acc"] == ref["arms"]["SH6-3-3_prior"] and a["SH6-3-3_prior"]["pick"] == ref["picks"]["SH6-3-3_prior"]
    assert a["SH6-3-3_prior"]["cost_iters"] == ref["cost_iters"]["SH6-3-3_prior"]
    for name in ("oracle_catalog", "oracle_curated", "oracle_random"):
        assert name in a and a[name]["acc"] <= a["oracle_pool"]["acc"]
    assert a["SH100-8-3"]["cost_full"] == pytest.approx(0.25 * 100 + 0.25 * 8 + 0.5 * 3)


# ---------------------------------------------------------------- 门
def _per(ours, base):
    return {"ours": {k: {"regret_pool": v} for k, v in ours.items()}, "baselines": {k: {"regret_mean": v} for k, v in base.items()}}


def test_gates_four_branches(cfg):
    good = {d: _per({"SH10-4-3_prior": 0.02, "SH6-3-3_prior": 0.05, "hybrid16-8-3": 0.01},
                    {"rand-SH10-4-3": 0.06, "rand-SH6-3-3": 0.08, "RS-5": 0.05, "rand-SH12-6-3": 0.04, "HB-5": 0.05, "TPE-5": 0.05,
                     "RS-8": 0.03, "HB-7": 0.02, "rand-SH16-8-3": 0.03, "TPE-8": 0.03}) for d in SIX}
    g = g46.run_gates_p1(good, cfg)
    assert g["E1"]["pass"] and g["E1"]["n_better"] == 6 and g["E1"]["mean_diff"] == pytest.approx(0.04)
    assert g["E2"]["pass"] and g["E2"]["min_baseline"] == "rand-SH12-6-3"
    assert g["E3"]["pass"] and g["E3"]["n_ge_hb"] == 6
    assert g46.verdict_code_p1(g) == "P1-WIN"
    tier1 = json.loads(json.dumps(good))
    for d in SIX:
        tier1[d]["baselines"]["HB-7"]["regret_mean"] = 0.0
    assert g46.verdict_code_p1(g46.run_gates_p1(tier1, cfg)) == "P1-TIER1"
    lst = json.loads(json.dumps(good))
    for d in SIX:
        lst[d]["baselines"]["TPE-5"]["regret_mean"] = 0.0
    assert g46.verdict_code_p1(g46.run_gates_p1(lst, cfg)) == "P1-LIST"
    dead = json.loads(json.dumps(good))
    for d in list(SIX)[:3]:
        dead[d]["ours"]["SH10-4-3_prior"]["regret_pool"] = 0.07          # 只赢 3/6 → E1 ✗
    assert g46.verdict_code_p1(g46.run_gates_p1(dead, cfg)) == "P1-DEAD"
    small = json.loads(json.dumps(good))
    for d in SIX:
        small[d]["baselines"]["rand-SH10-4-3"]["regret_mean"] = 0.025    # 6/6 赢但均值差 0.005 < 0.01 → E1 ✗
    assert not g46.run_gates_p1(small, cfg)["E1"]["pass"]


# ---------------------------------------------------------------- finals 复用
def test_finals_reuse_dedup():
    cells = [{"dataset": "X", "cls_key": "k1", "seed": 1}, {"dataset": "X", "cls_key": "k1", "seed": 2},
             {"dataset": "X", "cls_key": "k2", "seed": 1}, {"dataset": "Y", "cls_key": "k1", "seed": 1}]
    reuse = [_row("X", "k1", [0.5, 0.6, 0.7], seed=1, stage="s2_finals"),
             _row("X", "k1", None, seed=2, stage="s2_finals", error="boom"),
             _row("X", "k2", [0.5, 0.6, 0.7], seed=1, stage="s2_finals", options={"batch_size": 8, "input_mask": "binomial"}),
             _row("Y", "k1", [0.5, 0.6, 0.7], seed=1, stage="bo_finals")]
    todo, reused = g46.finals_reuse(cells, reuse)
    assert [(c["dataset"], c["cls_key"], c["seed"]) for c in todo] == [("X", "k1", 2), ("X", "k2", 1)]
    assert set(reused) == {("X", "k1", 1), ("Y", "k1", 1)} and reused[("X", "k1", 1)]["acc"] == 0.7


# ---------------------------------------------------------------- UCR
def test_ucr_arm_costs_and_ceiling_filter():
    b = (60, 120, 290)
    assert g46.sh_cost_iters((6, 3, 3), b) == 1050 and g46.sh_cost_iters((10, 4, 3), b) == 1350
    assert g46.hb_cost_iters(b, [2, 1, 0]) == 1930
    rows = []
    for d, top in (("A", 0.99), ("B", 0.90), ("C", 0.985)):
        for i in range(20):
            for it in b:
                rows.append({"dataset": d, "task_id": f"{d}_00", "cls_key": f"{d}k{i:02d}", "origin": "shared" if i < 10 else "random",
                             "iters": it, "r_prime": (top if i == 0 and it == 290 else 0.5), "seconds": 1.0 * it, "error": None})
    assert g46.ucr_ceiling_datasets(rows, budget=290, ceiling=0.98) == ["A", "C"]


# ---------------------------------------------------------------- 消融汇总
def test_ablation_rows_from_synthetic_json(tmp_path):
    p = tmp_path / "analysis.json"
    p.write_text(json.dumps({"mean_regret": {"prior1": 0.1, "prior3": 0.05}, "n_instances": 42}), encoding="utf-8")
    rows = g46.ablation_rows([{"block": "A", "name": "test", "path": str(p), "key": "mean_regret", "n_key": "n_instances", "b_final": 290}])
    assert {r["arm"]: r["value"] for r in rows} == {"prior1": 0.1, "prior3": 0.05}
    assert all(r["source_path"] == str(p) and r["n"] == 42 and r["b_final"] == 290 and r["block"] == "A" for r in rows)
    missing = g46.ablation_rows([{"block": "Z", "name": "nope", "path": str(tmp_path / "none.json"), "key": "x"}])
    assert missing == [{"block": "Z", "name": "nope", "arm": None, "metric": "x", "value": None, "n": None, "b_final": None,
                        "source_path": str(tmp_path / "none.json"), "note": "missing"}]


# ---------------------------------------------------------------- 端到端（合成池，零 GPU；不触碰真实基线读数）
def _synthetic_outdir(tmp_path, cfg, space):
    """两靶 × 20 条合成池（11 条带先验 + 9 条随机）+ 三截点 runs.jsonl（全部标 new/s1_pool）→ (cfg2, outdir)。"""
    import copy
    rng = np.random.default_rng(3)
    cfg2 = copy.deepcopy(cfg)
    cfg2["pool"]["wp3_runs"] = "results/__none__/runs.jsonl"
    cfg2["pool"]["wp8_runs"] = "results/__none__/runs.jsonl"
    cfg2["finals"]["reuse"] = []
    cfg2["sim"]["n_draws"] = 5
    outdir = tmp_path / "p1"
    outdir.mkdir()
    pool = {"config_md5": "x", "targets": {}}
    rows = []
    for d in ("Handwriting", "Cricket"):
        budgets = g46.budgets_for(cfg, d)
        cfgs = [space.sample_random(rng) for _ in range(20)]
        keys = [cls_key(c) for c in cfgs]
        prior_keys = keys[:11]
        prior = {k: float(rng.normal()) for k in prior_keys}
        pred = {k: float(rng.normal()) for k in keys}
        po, ho, so = g46.pool_orders(keys, prior, pred, prior_keys=prior_keys)
        entries = [{"cls_key": k, "cls": c.to_dict(), "origin": ("shared" if i < 10 else "ggs" if i == 10 else "random_space"),
                    "source": "new", "prior": prior.get(k), "pred_z": pred[k], "error": False} for i, (k, c) in enumerate(zip(keys, cfgs))]
        pool["targets"][d] = {"order": cfg["targets"][d]["order"], "budgets": budgets, "entries": entries, "prior_keys": prior_keys,
                              "catalog_keys": prior_keys, "curated_keys": prior_keys, "random_keys": keys[11:],
                              "n_random": 9, "prior_order": po, "hybrid_order": ho, "surrogate_order": so}
        for k, c in zip(keys, cfgs):
            v = float(rng.uniform(0.3, 0.9))
            accs = [round(v - 0.2, 4), round(v - 0.1, 4), round(v, 4)]
            rows.append(_row(d, k, accs, arm="random_space", stage="s1_pool", budgets=tuple(budgets), extra={"cls": c.to_dict()}))
    (outdir / "pool.json").write_text(json.dumps(pool), encoding="utf-8")
    (outdir / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return cfg2, outdir


def test_simulate_analyze_end_to_end_synthetic(tmp_path, cfg, space):
    cfg2, outdir = _synthetic_outdir(tmp_path, cfg, space)
    ds = ["Handwriting", "Cricket"]
    sim = g46.cmd_simulate(cfg2, outdir, ds)
    assert (outdir / "simulate.json").exists() and (outdir / "tables.md").exists()
    for d in ds:
        r = sim["per_target"][d]
        assert r["n_pool"] == 20 and r["missing"] == []
        assert set(r["ours"]) >= set(g46.OURS) | {"oracle_pool", "oracle_catalog", "oracle_curated", "oracle_random", "SH100-8-3"}
        b = r["baselines"]
        assert {"RS-5", "RS-8", "rand-SH10-4-3", "rand-SH12-6-3", "rand-SH16-8-3", "HB-7", "HB-5", "HB-lite", "TPE-4", "TPE-5", "TPE-8"} <= set(b)
        assert b["HBx2"]["cost_full"] == pytest.approx(14.0)          # 20 条池恰够 HBx2 不放回抽 20 条
        assert b["HB-7"]["cost_full"] == pytest.approx(7.0) and b["HB-5"]["cost_full"] == pytest.approx(5.0)
        assert b["rand-SH12-6-3"]["cost_full"] == pytest.approx(6.0) and b["TPE-5"]["cost_full"] == pytest.approx(5.0)
        assert all(-1e-9 <= b[n]["regret_mean"] <= 1 for n in b if not n.startswith("_"))   # 全池抽样臂 mean == oracle 到浮点
        assert r["curve"]["ours"] and r["curve"]["baselines"] and "random" not in r["subpools"]   # 随机子池 9 < 12 不报
        assert r["random_stats"]["n"] == 9
    a = g46.cmd_analyze(cfg2, outdir, ds)
    assert a["verdict"] in {"P1-WIN", "P1-TIER1", "P1-LIST", "P1-DEAD"}
    assert set(a["gates"]) == {"E1", "E2", "E3"} and set(a["predictions"]) == {f"P{i}" for i in range(1, 11)}
    picks = json.loads((outdir / "picks.json").read_text(encoding="utf-8"))
    assert set(picks) == set(ds) and {"SH10-4-3_prior", "hybrid16-8-3", "pool_top1", "pool_top2", "pool_top3"} <= set(picks["Handwriting"])
    assert a["finals"]["Handwriting"]["oracle_3seed"] is None       # 无终评行
    assert "P1-" in (outdir / "tables.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- 金标（真实产物）
_WP3 = ROOT / "results" / "mx4_deploy"
_WP8 = ROOT / "results" / "mx4_bo" / "runs.jsonl"
_MASK = ROOT / "results" / "mx4_mask" / "s1_sweep.md"


@pytest.mark.skipif(not ((_WP3 / "runs.jsonl").exists() and (_WP3 / "analysis.json").exists() and _WP8.exists()), reason="缺 WP3/WP8 产物")
def test_c0_sh6_matches_wp3_and_sh10_matches_mask_sweep(cfg):
    c0 = g46.c0_check(cfg)
    wp3 = json.loads((_WP3 / "analysis.json").read_text(encoding="utf-8"))["per_target"]
    for d in SIX:
        assert c0[d]["SH6-3-3_prior"]["acc"] == wp3[d]["arms"]["SH6-3-3_prior"], d
        assert c0[d]["SH6-3-3_prior"]["pick"] == wp3[d]["picks"]["SH6-3-3_prior"], d
    sh10 = {"Handwriting": ("c92058f2", 0.546), "UWaveGestureLibrary": ("8c4aa283", 0.931), "PhonemeSpectra": ("4f7d4135", 0.251),
            "EthanolConcentration": ("cffb5503", 0.297), "Cricket": ("c92058f2", 1.000), "HandMovementDirection": ("9092e36e", 0.378)}
    for d, (k, acc) in sh10.items():
        assert c0[d]["SH10-4-3_prior"]["pick"].startswith(k) and round(c0[d]["SH10-4-3_prior"]["acc"], 3) == pytest.approx(acc), d
    assert c0["_pass"] is True


@pytest.mark.skipif(not ((_WP3 / "candidates.json").exists() and (ROOT / "results" / "mx3_pq" / "rprime_cache.jsonl").exists()),
                    reason="缺 WP3 候选 / 源池缓存")
def test_c0_surrogate_reproduces_wp3_pred_z(cfg):
    cands = json.loads((_WP3 / "candidates.json").read_text(encoding="utf-8"))
    for d in ("Handwriting", "Cricket"):
        model, enc = g46.fit_surrogate(cfg, d)
        for e in cands[d]["nominated"][:3]:
            z = g46.surrogate_predict(model, enc, [e["cls"]])[0]
            assert z == pytest.approx(e["pred_z"], abs=1e-9), (d, e["cls_key"])
