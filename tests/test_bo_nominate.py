"""WP8-BO：目标侧提名回灌搜索（scripts/43_bo_nominate.py）纯函数测试——先于实现（铁律 5）。

覆盖：
- 变异候选池：全部 validate 通过、键唯一、排除集不再出现；
- 一轮提名：ucb 按 pred + κ·std 降序、汉明 ≤ radius（到源池 ∪ 靶侧已测）、去重；rand 不看代理（同 seed 换 κ/bagging 结果不变）；
- 靶侧回灌行：within-靶 @25% z（均值 0 / sd 1）、复制 ×copies、iters 特征 = 60、出错格不入；
- 父本 = 靶上 @25% 前 k；
- 轮次格：stage/arm/seed/完整三截点、arm 入键（两臂同 CLS 不同键）；
- 策略表：成本全跑当量（SH6-3-3 3.75 / SH10-4-3 5.0 / ONESHOT 6.25 / MIX 3.75 / BO 7.5 / RAND 7.5）、SH6-3-3 与 36 号逐位、起跑名单构成、oracle；
- 已测集合并（目录 + 提名 + error 行）；
- 门 B1–B3 与读数码五种。
零 GPU、不读真实缓存。
"""
from __future__ import annotations

import importlib
from dataclasses import asdict

import numpy as np
import pytest

from autocls.labeling import cls_key
from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLS, CLSSpace

g43 = importlib.import_module("scripts.43_bo_nominate")
_g35 = importlib.import_module("scripts.35_protocol_alignment")
_g36 = importlib.import_module("scripts.36_prior_sh_sim")
B3 = (50, 100, 200)
TARGETS6 = ("Handwriting", "UWaveGestureLibrary", "PhonemeSpectra", "EthanolConcentration", "Cricket",
            "HandMovementDirection")


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


@pytest.fixture(scope="module")
def enc(space):
    return CLSEncoder(space, mode="binned")


def _src_rows(space, rng, n_cls=30, n_tasks=4, budgets=(60, 120, 290)):
    cfgs = [space.sample_random(rng) for _ in range(n_cls)]
    rows = []
    for t in range(n_tasks):
        for c in cfgs:
            for b in budgets:
                rows.append({"task_id": f"src{t}", "cls_key": cls_key(c), "cls": asdict(c), "iters": b,
                             "r_prime": float(rng.uniform(0.3, 0.9)), "seconds": 1.0})
    return _g36.zscore_within(rows)


def _measured(space, rng, n=8, budgets=B3):
    out = {}
    for _ in range(n):
        c = space.sample_random(rng)
        out[cls_key(c)] = {"cls": asdict(c), "traj": {b: float(rng.uniform(0.3, 0.9)) for b in budgets},
                           "secs": {b: float(b) for b in budgets}, "source": "catalog"}
    return out


def _row(c, arm, accs, *, stage="s1_catalog", error=None, budgets=B3):
    return {"cls_key": cls_key(c), "cls": asdict(c), "arm": arm, "stage": stage, "seed": 0, "error": error,
            "checkpoint_results": None if error else [
                {"iters": b, "acc": a, "elapsed_pretrain_s": float(b), "eval_seconds": 1.0}
                for b, a in zip(budgets, accs)]}


# ---------------------------------------------------------------- 变异池
def test_mutate_pool_valid_unique_and_excluded(space):
    rng = np.random.default_rng(0)
    parents = [space.sample_random(rng) for _ in range(5)]
    exclude = {cls_key(parents[0]), cls_key(parents[1])}
    pool = g43.mutate_pool(space, parents, pool=300, max_mut=3, rng=np.random.default_rng(1), exclude=exclude)
    assert len(pool) > 50
    assert all(space.validate(c) for c in pool.values())
    assert all(cls_key(c) == k for k, c in pool.items())
    assert not (set(pool) & exclude)


# ---------------------------------------------------------------- 一轮提名
def _setup(space, seed=0):
    rng = np.random.default_rng(seed)
    src = _src_rows(space, rng)
    meas = _measured(space, rng)
    tgt = g43.target_rows(meas, B3[0], task_id="tgt", copies=3)
    parents = [CLS(**meas[k]["cls"]) for k in g43.parents_of(meas, B3[0], k=5)]
    exclude = set(meas) | {r["cls_key"] for r in src}
    return src, tgt, parents, exclude


def test_nominate_round_ucb_orders_by_pred_plus_kappa_std(space, enc):
    src, tgt, parents, exclude = _setup(space)
    out = g43.nominate_round(space, enc, src, tgt, parents=parents, exclude=exclude, n=5, pool=400,
                             radius=3, mode="ucb", kappa=1.0, bagging=3, seed=0)
    assert len(out) == 5
    keys = [e["cls_key"] for e in out]
    assert len(set(keys)) == 5 and not (set(keys) & exclude)
    seen = {r["cls_key"]: CLS(**r["cls"]) for r in list(src) + list(tgt)}
    Xs = np.stack([enc.encode(c) for c in seen.values()])
    for e in out:
        c = CLS(**e["cls"])
        assert space.validate(c) and cls_key(c) == e["cls_key"]
        ham = int(g43.field_hamming(enc.encode(c)[None, :], Xs, enc.field_slices)[0])
        assert e["hamming"] == ham <= 3
        assert e["score"] == pytest.approx(e["pred_z"] + 1.0 * e["std_z"])
        assert e["std_z"] >= 0.0
    scores = [e["score"] for e in out]
    assert scores == sorted(scores, reverse=True)


def test_nominate_round_rand_ignores_surrogate(space, enc):
    src, tgt, parents, exclude = _setup(space)
    a = g43.nominate_round(space, enc, src, tgt, parents=parents, exclude=exclude, n=5, pool=400,
                           radius=3, mode="rand", kappa=0.0, bagging=2, seed=3)
    b = g43.nominate_round(space, enc, src, tgt, parents=parents, exclude=exclude, n=5, pool=400,
                           radius=3, mode="rand", kappa=5.0, bagging=3, seed=3)
    assert [e["cls_key"] for e in a] == [e["cls_key"] for e in b]
    assert len({e["cls_key"] for e in a}) == 5 and not ({e["cls_key"] for e in a} & exclude)
    assert all(e["hamming"] <= 3 and space.validate(CLS(**e["cls"])) for e in a)
    c = g43.nominate_round(space, enc, src, tgt, parents=parents, exclude=exclude, n=5, pool=400,
                           radius=3, mode="rand", kappa=0.0, bagging=2, seed=4)
    assert [e["cls_key"] for e in c] != [e["cls_key"] for e in a]


def test_nominate_round_bad_mode(space, enc):
    src, tgt, parents, exclude = _setup(space)
    with pytest.raises(ValueError):
        g43.nominate_round(space, enc, src, tgt, parents=parents, exclude=exclude, mode="greedy")


# ---------------------------------------------------------------- 靶侧行 / 父本
def test_target_rows_z_and_copies(space):
    meas = _measured(space, np.random.default_rng(1), n=6)
    rows = g43.target_rows(meas, B3[0], task_id="tgt", copies=10, iters_feature=60)
    assert len(rows) == 60
    assert all(r["iters"] == 60 and r["task_id"] == "tgt" for r in rows)
    uniq = {r["cls_key"]: r["z"] for r in rows}
    z = np.array(list(uniq.values()))
    assert abs(z.mean()) < 1e-9 and z.std() == pytest.approx(1.0)
    best = max(meas, key=lambda k: meas[k]["traj"][B3[0]])
    assert uniq[best] == max(uniq.values())
    assert all(r["cls"] == meas[r["cls_key"]]["cls"] for r in rows)
    meas2 = dict(meas)
    meas2["bad"] = {"cls": meas[best]["cls"], "traj": {}, "secs": {}, "source": "bo_ucb_r1"}
    assert "bad" not in {r["cls_key"] for r in g43.target_rows(meas2, B3[0], task_id="tgt", copies=2)}


def test_parents_of_top_k_by_first_checkpoint(space):
    meas = _measured(space, np.random.default_rng(2), n=9)
    order = sorted(meas, key=lambda k: (-meas[k]["traj"][B3[0]], k))
    assert g43.parents_of(meas, B3[0], k=5) == order[:5]
    meas["err"] = {"cls": meas[order[0]]["cls"], "traj": {}, "secs": {}, "source": "bo_ucb_r2"}
    assert "err" not in g43.parents_of(meas, B3[0], k=len(meas))


# ---------------------------------------------------------------- 轮次格
def test_round_cells_full_trajectory_and_arm_in_key(space):
    protos = _g35.load_protocols()
    sizes = {"Handwriting": tuple(_g35.SIZES_SURVEY["Handwriting"])}
    rng = np.random.default_rng(0)
    cfgs = [space.sample_random(rng) for _ in range(3)]
    entries = [{"cls_key": cls_key(c), "cls": asdict(c)} for c in cfgs]
    cells = g43.round_cells("Handwriting", entries, arm="bo_ucb", rnd=1, protos=protos, sizes=sizes,
                            protocol="ts2vec_official")
    assert len(cells) == 3 and len({c["key"] for c in cells}) == 3
    for c in cells:
        assert c["stage"] == "bo_round" and c["arm"] == "bo_ucb_r1" and c["seed"] == 0
        assert c["dataset"] == "Handwriting" and c["protocol"] == "ts2vec_official"
        assert len(c["checkpoints"]) == 3 and c["checkpoints"][-1] == c["iters"]
        assert c["cls"] in [asdict(x) for x in cfgs]
    other = g43.round_cells("Handwriting", entries, arm="rand_nbr", rnd=1, protos=protos, sizes=sizes,
                            protocol="ts2vec_official")
    assert not ({c["key"] for c in cells} & {c["key"] for c in other})


# ---------------------------------------------------------------- 已测集合并
def test_measured_from_rows_merges_catalog_and_rounds(space):
    rng = np.random.default_rng(3)
    cat = [space.sample_random(rng) for _ in range(3)]
    nom = [space.sample_random(rng) for _ in range(2)]
    rows = [_row(c, "s1_catalog", (0.5, 0.6, 0.7)) for c in cat]
    rows += [_row(nom[0], "bo_ucb_r1", (0.4, 0.5, 0.8), stage="bo_round")]
    rows += [_row(nom[1], "bo_ucb_r2", None, stage="bo_round", error="RuntimeError: boom")]
    meas = g43.measured_from_rows(rows)
    assert set(meas) == {cls_key(c) for c in cat + nom}
    assert meas[cls_key(nom[0])]["traj"] == {50: 0.4, 100: 0.5, 200: 0.8}
    assert meas[cls_key(nom[0])]["source"] == "bo_ucb_r1"
    assert meas[cls_key(nom[1])]["traj"] == {} and meas[cls_key(nom[1])]["source"] == "bo_ucb_r2"
    assert meas[cls_key(cat[0])]["secs"][200] == pytest.approx(200.0 + 3.0)


# ---------------------------------------------------------------- 策略表
def test_policy_table_costs_and_sh_reproduction(space):
    rng = np.random.default_rng(5)
    cat = _measured(space, rng, n=21)
    keys = list(cat)
    shared, ggs, nomin = keys[:10], keys[10], keys[11:]
    prior = {k: float(rng.normal()) for k in shared + [ggs]}
    nom_pred = {k: float(rng.normal()) for k in nomin}
    bo = {"bo_ucb": _measured(space, rng, n=15), "rand_nbr": _measured(space, rng, n=15)}
    traj = {k: v["traj"] for k, v in cat.items()}
    secs = {k: v["secs"] for k, v in cat.items()}
    for arm in bo:
        traj.update({k: v["traj"] for k, v in bo[arm].items()})
        secs.update({k: v["secs"] for k, v in bo[arm].items()})
    table = g43.policy_table(traj, secs, budgets=B3, prior=prior, nominated=nom_pred, shared_keys=shared,
                             ggs_key=ggs, bo_keys={a: list(v) for a, v in bo.items()})
    pol = table["policies"]
    assert pol["SH6-3-3_prior"]["cost_full"] == pytest.approx(3.75)
    assert pol["SH10-4-3_prior"]["cost_full"] == pytest.approx(5.0)
    assert pol["ONESHOT-SH"]["cost_full"] == pytest.approx(6.25)
    assert pol["MIX-SH"]["cost_full"] == pytest.approx(3.75)
    assert pol["BO-SH"]["cost_full"] == pytest.approx(7.5)
    assert pol["RAND-SH"]["cost_full"] == pytest.approx(7.5)
    ranked = _g36.rank_by(prior, keys)
    path = _g36.sh_path(ranked, traj, (6, 3, 3), B3)
    assert pol["SH6-3-3_prior"]["pick"] == _g36.sh_pick(path[-1], traj, B3[-1])
    assert pol["SH6-3-3_prior"]["path"] == path
    assert pol["SH6-3-3_prior"]["cost_seconds"] == pytest.approx(_g36.path_cost_seconds(path, B3, secs))
    assert set(pol["BO-SH"]["path"][0]) == set(ranked[:6]) | set(bo["bo_ucb"])
    assert set(pol["RAND-SH"]["path"][0]) == set(ranked[:6]) | set(bo["rand_nbr"])
    assert set(pol["ONESHOT-SH"]["path"][0]) == set(ranked[:6]) | set(nomin)
    mix = {**{k: prior[k] for k in prior}, **nom_pred}
    assert pol["MIX-SH"]["path"][0] == _g36.rank_by(mix, keys)[:6]
    assert table["oracle_catalog"] == pytest.approx(max(traj[k][200] for k in keys))
    assert table["oracle_pool"]["bo_ucb"] == pytest.approx(max(traj[k][200] for k in keys + list(bo["bo_ucb"])))
    for p in pol.values():
        assert p["acc"] == pytest.approx(traj[p["pick"]][200])
        assert p["regret_catalog"] == pytest.approx(table["oracle_catalog"] - p["acc"])
        assert p["regret_pool"] >= -1e-12
    assert table["nominee_max"]["bo_ucb"] == pytest.approx(max(traj[k][200] for k in bo["bo_ucb"]))
    assert table["nominee_mean"]["rand_nbr"] == pytest.approx(np.mean([traj[k][200] for k in bo["rand_nbr"]]))


def test_policy_table_skips_incomplete_trajectories(space):
    rng = np.random.default_rng(6)
    cat = _measured(space, rng, n=12)
    keys = list(cat)
    shared, ggs, nomin = keys[:6], keys[6], keys[7:]
    prior = {k: float(i) for i, k in enumerate(shared + [ggs])}
    traj = {k: v["traj"] for k, v in cat.items()}
    secs = {k: v["secs"] for k, v in cat.items()}
    traj[nomin[0]] = {50: 0.99}          # PS 式：一格 error 只有首截点
    table = g43.policy_table(traj, secs, budgets=B3, prior=prior, nominated={k: 0.0 for k in nomin},
                             shared_keys=shared, ggs_key=ggs, bo_keys={"bo_ucb": [], "rand_nbr": []})
    assert nomin[0] not in table["keys"]
    assert table["policies"]["ONESHOT-SH"]["cost_full"] == pytest.approx((10 * 50 + 3 * 50 + 3 * 100) / 200)


# ---------------------------------------------------------------- 门与读数码
def _pt(cat_or, bo_max, rand_max, bo_mean, rand_mean, m3_bo, m3_rand, m3_sh):
    return {"oracle_catalog": cat_or,
            "nominee_max": {"bo_ucb": bo_max, "rand_nbr": rand_max},
            "nominee_mean": {"bo_ucb": bo_mean, "rand_nbr": rand_mean},
            "oracle_pool": {"bo_ucb": max(cat_or, bo_max), "rand_nbr": max(cat_or, rand_max)},
            "mean3": {"BO-SH": m3_bo, "RAND-SH": m3_rand, "SH6-3-3_prior": m3_sh}}


def test_gates_and_verdict_codes():
    win = {d: _pt(0.50, 0.52, 0.50, 0.45, 0.40, 0.55, 0.50, 0.50) for d in TARGETS6}
    g = g43.gates(win)
    assert g["B1"]["pass"] and g["B2"]["pass"] and g["B3"]["pass"]
    assert g["B1"]["n_pass"] == 6 and g["B2"]["hw_ok"] is True
    assert g43.verdict_code(g) == "BO-WIN"

    headroom = {d: _pt(0.50, 0.52, 0.50, 0.45, 0.40, 0.50, 0.50, 0.50) for d in TARGETS6}
    g = g43.gates(headroom)
    assert g["B1"]["pass"] and not g["B2"]["pass"]
    assert g43.verdict_code(g) == "BO-HEADROOM"

    hw_line = {d: _pt(0.50, 0.52, 0.50, 0.45, 0.40, 0.55, 0.50, 0.50) for d in TARGETS6}
    hw_line["Handwriting"] = _pt(0.50, 0.52, 0.50, 0.45, 0.40, 0.48, 0.45, 0.43)
    g = g43.gates(hw_line)
    assert g["B2"]["n_pass"] == 6 and g["B2"]["hw_ok"] is False and not g["B2"]["pass"]
    assert g43.verdict_code(g) == "BO-HEADROOM"

    luck = {d: _pt(0.50, 0.52, 0.53, 0.40, 0.45, 0.55, 0.55, 0.50) for d in TARGETS6}
    g = g43.gates(luck)
    assert g["B1"]["pass"] and g["B2"]["pass"] and not g["B3"]["pass"]
    assert g43.verdict_code(g) == "BO-LUCK"

    dead = {d: _pt(0.50, 0.505, 0.50, 0.45, 0.40, 0.55, 0.50, 0.50) for d in TARGETS6}
    g = g43.gates(dead)
    assert not g["B1"]["pass"]
    assert g43.verdict_code(g) == "BO-DEAD"

    pending = {d: _pt(0.50, 0.52, 0.50, 0.45, 0.40, None, None, 0.50) for d in TARGETS6}
    g = g43.gates(pending)
    assert g43.verdict_code(g) == "BO-PENDING"
    assert g43.verdict_code(g43.gates({d: win[d] for d in TARGETS6[:4]})) == "BO-PENDING"


def test_gate_b1_needs_three_of_six():
    pt = {d: _pt(0.50, 0.505, 0.50, 0.45, 0.40, 0.55, 0.50, 0.50) for d in TARGETS6}
    for d in TARGETS6[:3]:
        pt[d]["nominee_max"]["bo_ucb"] = 0.52
        pt[d]["oracle_pool"]["bo_ucb"] = 0.52
    g = g43.gates(pt)
    assert g["B1"]["n_pass"] == 3 and g["B1"]["pass"]
