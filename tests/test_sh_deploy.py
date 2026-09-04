"""MX-4 WP3：全量官方协议上的"零样本名单 + 逐次减半"部署（scripts/39_sh_deploy.py）测试——先于实现（铁律 5）。

协议：results/mx4_deploy/PROTOCOL.md。覆盖纯函数：
- 截点 = 官方 iters × {25%, 50%, 100%}（严格递增、末位 = iters）；
- 代理提名：LODO 训练侧拟合、字段汉明 ≤ 3、排除已在目录的键、去重、确定性、按预测 z 降序；
- 先验含 GGS：GGS 的 z 按目录 20 条的组统计量算；
- 轨迹装配：runs 行的 checkpoint_results → {key: {iters: acc}}；
- 部署臂：三级 / 两级 / 只留 1 条 / 纯多保真 / 随机起点，regret 与成本；
- 工单：S1 = 靶 × (shared-10 + GGS + 提名) × seed 0，键唯一、带截点；S2 = 冠军 × seeds {1,2}；
- 门 W1–W4 与读数码。
零 GPU、不读真实缓存、不加载数据集。
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from autocls.search.surrogate import field_hamming
from autocls.space.cls_encoder import CLSEncoder
from autocls.space.cls_space import CLSSpace

g39 = importlib.import_module("scripts.39_sh_deploy")


@pytest.fixture(scope="module")
def space():
    return CLSSpace.from_yaml()


@pytest.fixture(scope="module")
def binned(space):
    return CLSEncoder(space, mode="binned")


# ---------------------------------------------------------------- 截点
def test_checkpoints_for_official_iters():
    assert g39.checkpoints_for(200) == (50, 100, 200)
    assert g39.checkpoints_for(600) == (150, 300, 600)


def test_checkpoints_for_rejects_bad_fractions():
    with pytest.raises(ValueError):
        g39.checkpoints_for(200, fractions=(0.5, 0.25, 1.0))
    with pytest.raises(ValueError):
        g39.checkpoints_for(200, fractions=(0.25, 0.5))          # 末位必须是 1.0
    with pytest.raises(ValueError):
        g39.checkpoints_for(2, fractions=(0.25, 0.5, 1.0))       # round(0.5)=0 → 首截点 <1


# ---------------------------------------------------------------- 提名
def _train_rows(space, n_cls=40, seed=0):
    """合成训练侧：n_cls 个 CLS × 2 任务 × 2 预算，z 由固定线性规则生成。"""
    rng = np.random.default_rng(seed)
    rows = []
    from autocls.labeling import cls_key
    for i in range(n_cls):
        c = space.sample_random(rng)
        d = c.__dict__
        base = (1.0 if c.temperature <= 1.0 else -1.0) + 0.3 * (c.sim_func == "dot")
        for t in ("S_00", "S_01"):
            for b in (60, 290):
                rows.append({"task_id": t, "dataset": "S", "iters": b, "cls_key": cls_key(c),
                             "origin": "random", "cls": dict(d), "r_prime": 0.5,
                             "z": base + rng.normal(scale=0.05)})
    return rows


def test_nominate_is_deterministic_in_hull_and_excludes_catalog(space, binned):
    # 候选由"父本局部变异 1..3 字段"构造，邻域由构造保证（均匀采样在 19 字段空间里筛不出邻居）
    rows = _train_rows(space)
    exclude = {rows[0]["cls_key"], rows[4]["cls_key"]}
    a = g39.nominate(space, binned, rows, exclude_keys=exclude, n=5, pool=300, radius=3, seed=1)
    b = g39.nominate(space, binned, rows, exclude_keys=exclude, n=5, pool=300, radius=3, seed=1)
    assert [x["cls_key"] for x in a] == [x["cls_key"] for x in b]
    assert 0 < len(a) <= 5 and len({x["cls_key"] for x in a}) == len(a)
    assert not ({x["cls_key"] for x in a} & exclude)
    from autocls.space.cls_space import CLS
    Xtr = np.stack([binned.encode(CLS(**r["cls"])) for r in rows])
    for x in a:
        assert x["hamming"] <= 3
        assert field_hamming(binned.encode(CLS(**x["cls"]))[None, :], Xtr, binned.field_slices)[0] <= 3
        assert space.validate(CLS(**x["cls"]))
    preds = [x["pred_z"] for x in a]
    assert preds == sorted(preds, reverse=True)


# ---------------------------------------------------------------- 先验含 GGS
def test_prior_with_ggs_uses_catalog_group_stats():
    cat = [{"task_id": "A_00", "iters": 290, "cls_key": k, "origin": "shared", "r_prime": r}
           for k, r in (("a", 0.2), ("b", 0.4), ("c", 0.6))]
    ggs = [{"task_id": "A_00", "iters": 290, "cls_key": "ggs", "origin": "ggs", "r_prime": 0.6}]
    prior = g39.prior_with_ggs(cat, ggs)
    sd = np.std([0.2, 0.4, 0.6])
    assert prior["ggs"] == pytest.approx(0.2 / sd)
    assert prior["c"] == pytest.approx(0.2 / sd) and prior["a"] == pytest.approx(-0.2 / sd)


# ---------------------------------------------------------------- 轨迹装配
def test_trajectory_from_rows_reads_checkpoint_results():
    rows = [{"dataset": "D", "cls_key": "k1", "seed": 0, "error": None,
             "checkpoint_results": [{"iters": 50, "acc": 0.3, "elapsed_pretrain_s": 1.0, "eval_seconds": 0.5},
                                    {"iters": 200, "acc": 0.5, "elapsed_pretrain_s": 4.0, "eval_seconds": 0.5}]},
            {"dataset": "D", "cls_key": "k2", "seed": 0, "error": "boom", "checkpoint_results": None}]
    traj, secs = g39.trajectory_from_rows(rows)
    assert traj == {"k1": {50: 0.3, 200: 0.5}}
    # 累计口径：到该截点的训练墙钟 + 截止该截点的全部评估耗时（每个截点都评一次）
    assert secs["k1"][50] == pytest.approx(1.5) and secs["k1"][200] == pytest.approx(5.0)


# ---------------------------------------------------------------- 部署臂
TRAJ = {
    "p": {50: 0.90, 100: 0.60, 200: 0.50},
    "q": {50: 0.80, 100: 0.85, 200: 0.95},
    "r": {50: 0.70, 100: 0.80, 200: 0.90},
    "s": {50: 0.10, 100: 0.20, 200: 0.30},
    "t": {50: 0.75, 100: 0.70, 200: 0.60},
    "g": {50: 0.60, 100: 0.65, 200: 0.70},
}


def test_deploy_arms_three_stage_two_stage_and_costs():
    keys = list(TRAJ)
    rank = {"prior": ["q", "r", "p", "g", "t", "s"], "hybrid": ["q", "p", "r", "g", "t", "s"],
            "tclsc": ["p", "q", "r", "s", "t"]}     # tclsc 只覆盖 shared（无 g）
    row = g39.deploy_arms(keys, TRAJ, rank, budgets=(50, 100, 200), random_seed=0)
    a = row["arms"]
    assert a["oracle"] == pytest.approx(0.95)
    assert a["prior1"] == pytest.approx(0.95) and a["tclsc1"] == pytest.approx(0.50)
    assert a["tclsc3"] == pytest.approx(0.95)
    assert a["SH6-3-3_prior"] == pytest.approx(0.95)
    # 两级（跳过 25%）：先验前 6 全部训到 100 → 留 3（q .85, r .80, t .70）→ 200 取最好 q
    assert a["SH6-3_prior"] == pytest.approx(0.95)
    assert row["cost_iters"]["SH6-3-3_prior"] == 6 * 50 + 3 * 50 + 3 * 100
    assert row["cost_iters"]["SH6-3_prior"] == 6 * 100 + 3 * 100
    assert row["cost_iters"]["catalog_direct"] == 6 * 200
    assert "all@50->top3" in a and "SH6-3-3_random" in a and "SH6-3-1_prior" in a
    assert row["regret"]["prior1"] == pytest.approx(0.0)
    assert row["picks"]["SH6-3-3_prior"] == "q"


# ---------------------------------------------------------------- 工单
def test_s1_worklist_counts_keys_and_checkpoints():
    protos = {"ts2vec_official": {"iters_rule": "ts2vec_size", "batch_size": 8, "lr": 1e-3,
                                  "drop_last": True, "swa": True, "crop_mode": "fixed",
                                  "scaler": "train", "svm_grid": "ts2vec"}}
    sizes = {"Handwriting": (150, 152, 3), "Cricket": (108, 1197, 6)}
    space = CLSSpace.from_yaml()
    rng = np.random.default_rng(0)
    from autocls.labeling import cls_key
    from dataclasses import asdict
    shared = [space.sample_random(rng) for _ in range(3)]
    ggs = space.sample_random(rng)
    nom = [space.sample_random(rng) for _ in range(2)]
    cands = {d: {"shared": [{"cls_key": cls_key(c), "cls": asdict(c)} for c in shared],
                 "ggs": {"cls_key": cls_key(ggs), "cls": asdict(ggs)},
                 "nominated": [{"cls_key": cls_key(c), "cls": asdict(c)} for c in nom]}
             for d in sizes}
    cells = g39.s1_worklist(protos, sizes, cands)
    assert len(cells) == 2 * (3 + 1 + 2)
    assert len({c["key"] for c in cells}) == len(cells)
    hw = [c for c in cells if c["dataset"] == "Handwriting"][0]
    cr = [c for c in cells if c["dataset"] == "Cricket"][0]
    assert hw["iters"] == 200 and hw["checkpoints"] == [50, 100, 200]
    assert cr["iters"] == 600 and cr["checkpoints"] == [150, 300, 600]
    assert all(c["seed"] == 0 and c["stage"] == "s1_catalog" for c in cells)
    assert {c["arm"] for c in cells} == {"shared", "ggs", "nominated"}


def test_finals_worklist_from_picks():
    protos = {"ts2vec_official": {"iters_rule": "ts2vec_size", "batch_size": 8, "lr": 1e-3,
                                  "drop_last": True, "swa": True, "crop_mode": "fixed",
                                  "scaler": "train", "svm_grid": "ts2vec"}}
    sizes = {"Handwriting": (150, 152, 3)}
    space = CLSSpace.from_yaml()
    from autocls.labeling import cls_key
    from dataclasses import asdict
    c1, c2 = space.sample_random(np.random.default_rng(1)), space.sample_random(np.random.default_rng(2))
    key2cls = {cls_key(c1): asdict(c1), cls_key(c2): asdict(c2)}
    picks = {"Handwriting": {"SH6-3-3_prior": cls_key(c1), "prior3": cls_key(c2)}}
    cells = g39.finals_worklist(protos, sizes, picks, key2cls, seeds=(1, 2))
    assert len(cells) == 4 and {c["seed"] for c in cells} == {1, 2}
    assert {c["arm"] for c in cells} == {"final_SH6-3-3_prior", "final_prior3"}
    assert all(c["stage"] == "s2_finals" for c in cells)


def test_finals_worklist_dedups_same_champion():
    protos = {"ts2vec_official": {"iters_rule": "ts2vec_size", "batch_size": 8, "lr": 1e-3,
                                  "drop_last": True, "swa": True, "crop_mode": "fixed",
                                  "scaler": "train", "svm_grid": "ts2vec"}}
    sizes = {"Handwriting": (150, 152, 3)}
    space = CLSSpace.from_yaml()
    from autocls.labeling import cls_key
    from dataclasses import asdict
    c1 = space.sample_random(np.random.default_rng(1))
    picks = {"Handwriting": {"SH6-3-3_prior": cls_key(c1), "prior3": cls_key(c1)}}
    cells = g39.finals_worklist(protos, sizes, picks, {cls_key(c1): asdict(c1)}, seeds=(1, 2))
    assert len(cells) == 2 and cells[0]["arm_names"] == ["final_SH6-3-3_prior", "final_prior3"]


# ---------------------------------------------------------------- 门
def _per_target(sh, p3, paper, hw_max=0.6):
    return {"champion_mean": sh, "prior3_mean": p3, "sh_seed0": sh, "prior3_seed0": p3,
            "paper_ts2vec": paper, "catalog_max": hw_max, "spearman_slice_vs_official": 0.6}


def test_gates_and_verdict():
    per = {"Handwriting": _per_target(0.50, 0.45, 0.515), "UWaveGestureLibrary": _per_target(0.90, 0.88, 0.906),
           "PhonemeSpectra": _per_target(0.23, 0.22, 0.233), "EthanolConcentration": _per_target(0.30, 0.30, 0.308),
           "Cricket": _per_target(0.98, 0.98, 0.972), "HandMovementDirection": _per_target(0.40, 0.42, 0.338)}
    g = g39.run_gates(per, margin=0.03, hw_line=0.485)
    assert g["W1"]["n_parity"] == 6 and g["W1"]["hw_parity"] and g["W1"]["pass"]
    assert g["W2"]["n_ge"] == 5 and g["W2"]["pass"]
    assert g["W3"]["pass"] and g["W4"]["pass"]
    assert g39.verdict_code(g) == "DEPLOY-WIN"
    per["Handwriting"]["champion_mean"] = 0.40
    g = g39.run_gates(per, margin=0.03, hw_line=0.485)
    assert not g["W1"]["pass"] and g39.verdict_code(g) == "DEPLOY-DEAD"


# ---------------------------------------------------------------- 截点评估墙钟上限
def test_eval_with_timeout_passthrough_when_fast():
    assert g39.eval_with_timeout(lambda: 42, 5) == 42


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGALRM"), reason="SIGALRM 仅 Unix（pod）")
def test_eval_with_timeout_raises_on_slow_eval():
    import time as _t
    with pytest.raises(g39.EvalTimeout):
        g39.eval_with_timeout(lambda: _t.sleep(3), 1)


# ---------------------------------------------------------------- WP3b：HAR 外部锚（legacy_full 协议）
def test_har_arrays_from_npz(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "har.npz"
    np.savez(p, X_train=rng.normal(size=(8, 16, 3)).astype(np.float32), y_train=np.arange(8) % 2,
             X_test=rng.normal(size=(4, 16, 3)).astype(np.float32), y_test=np.arange(4) % 2)
    a = g39.har_arrays(p)
    assert a["Xtr"].shape == (8, 16, 3) and a["yte"].shape == (4,)
    assert np.array_equal(a["Xte_split"], a["Xte_train"])       # npz 已按 train 统计量 z-score，两种 scaler 同一份


def test_paper_reference_har_and_uea():
    assert g39.paper_ts2vec("HAR") == pytest.approx(0.930)
    assert g39.paper_reference("HAR")["GGS"] == pytest.approx(0.937)
    assert g39.paper_ts2vec("Handwriting") == pytest.approx(0.515)


def test_checkpoints_for_legacy_full():
    assert g39.checkpoints_for(2320) == (580, 1160, 2320)


def test_sizes_for_har_without_loading():
    assert g39.sizes_for(("HAR",), load=False) == {"HAR": (7352, 128, 9)}


def test_s1_worklist_legacy_protocol_for_har():
    protos = {"legacy_full": {"iters": 2320, "batch_size": 128, "lr": 1e-3, "drop_last": False,
                              "swa": False, "crop_mode": "fixed", "scaler": "split", "svm_grid": "legacy"}}
    space = CLSSpace.from_yaml()
    from autocls.labeling import cls_key
    from dataclasses import asdict
    c = space.sample_random(np.random.default_rng(3))
    cands = {"HAR": {"shared": [{"cls_key": cls_key(c), "cls": asdict(c)}],
                     "ggs": {"cls_key": cls_key(c) + "x", "cls": asdict(c)}, "nominated": []}}
    cells = g39.s1_worklist(protos, {"HAR": (7352, 128, 9)}, cands, protocol="legacy_full")
    assert len(cells) == 2 and cells[0]["iters"] == 2320 and cells[0]["checkpoints"] == [580, 1160, 2320]
    assert cells[0]["protocol"] == "legacy_full" and cells[0]["options"]["swa"] is False


def test_har_gates_and_verdict():
    per = {"HAR": {"champion_mean": 0.936, "prior3_mean": 0.93, "sh_seed0": 0.936, "prior3_seed0": 0.93,
                   "paper_ts2vec": 0.930}}
    g = g39.run_gates_har(per, ggs_anchor=0.935)
    assert g["H1"]["pass"] and g["H2"]["pass"] and g["H3"]["pass"]
    assert g39.verdict_code_har(g) == "HAR-WIN"
    per["HAR"]["sh_seed0"] = 0.92
    assert g39.verdict_code_har(g39.run_gates_har(per, ggs_anchor=0.935)) == "HAR-PARITY"
    per["HAR"]["champion_mean"] = 0.90
    assert g39.verdict_code_har(g39.run_gates_har(per, ggs_anchor=0.935)) == "HAR-DEAD"
