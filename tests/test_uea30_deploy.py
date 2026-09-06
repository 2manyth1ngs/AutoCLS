"""WP10-UEA30：UEA 30 靶外 23 靶目标侧部署（scripts/45_uea30_deploy.py + configs/uea30.yaml）测试——先于实现（铁律 5）。

协议：results/uea30/PROTOCOL.md。覆盖：
- 配置真源：23 个 run 靶、分类 A–E、文献数（与 35:TABLE10 不重叠）、预测带、锚点、尺寸与 iters 规则；
- paper_reference 对未知靶返回 None（不抛）；
- 候选目录：只有 shared-10 + GGS（无提名），先验 LODO / 全 8 源与既有产物逐值一致；
- segment_train：官方 max_train_length 分段语义（加载层，trainer 零改动）；
- 单槽数组缓存 + Xtr_fit；
- OOM 重复剔除；
- 臂：SH10-4-3_prior（主）/ SH20-3_both（副）/ SH6-3-3_prior / SH20-4-3_both 的名单、路径、成本、选择；
- 完整性、门 U1–U3 与读数码、估时锚（硬件标签）、全 30 合并表。
零 GPU；真实产物的金标测试在文件缺失时 skip。
"""
from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from autocls.data.uea import segment_train
from autocls.labeling import cls_key
from autocls.space.cls_space import CLSSpace, twin

g45 = importlib.import_module("scripts.45_uea30_deploy")
g39 = g45._g39
g35 = g45._g35
ROOT = Path(__file__).resolve().parents[1]

_PROTOS = {"ts2vec_official": {"iters_rule": "ts2vec_size", "batch_size": 8, "lr": 1e-3,
                               "drop_last": True, "swa": True, "crop_mode": "fixed",
                               "scaler": "train", "svm_grid": "ts2vec"}}
SIX = ("Handwriting", "UWaveGestureLibrary", "PhonemeSpectra", "EthanolConcentration", "Cricket",
       "HandMovementDirection")


# ---------------------------------------------------------------- 合成夹具
def _cands11(datasets=("RacketSports", "PenDigits"), seed=0):
    """每靶 shared-10 + GGS，无提名（UEA30 目录）。"""
    space = CLSSpace.from_yaml()
    rng = np.random.default_rng(seed)
    shared = [space.sample_random(rng) for _ in range(10)]
    ggs = space.sample_random(rng)
    return {d: {"shared": [{"cls_key": cls_key(c), "cls": asdict(c)} for c in shared],
                "ggs": {"cls_key": cls_key(ggs), "cls": asdict(ggs)},
                "prior": {cls_key(c): float(-i) for i, c in enumerate(shared + [ggs])},
                "prior_order": [cls_key(c) for c in shared + [ggs]],
                "nominated": [], "tclsc_order": [], "fit_on": "all 8 source datasets"}
            for d in datasets}


def _traj22():
    """11 条 legacy + 11 条孪生的合成轨迹（截点 50/100/200；@50 = 终值 − 0.2，@100 = 终值 − 0.1）。"""
    final_off = [0.60, 0.70, 0.65, 0.90, 0.50, 0.55, 0.40, 0.45, 0.30, 0.35, 0.95]
    keys_off = [f"L{i:02d}" for i in range(11)]
    twin_map = {f"L{i:02d}": f"T{i:02d}" for i in range(11)}
    final_on = {f"T{i:02d}": round(v - 0.10, 6) for i, v in enumerate(final_off)}     # 舍入：同分对在各截点并列判定一致
    final_on["T03"], final_on["T01"] = 0.92, 0.80
    traj = {}
    for k, v in list(zip(keys_off, final_off)) + list(final_on.items()):
        traj[k] = {50: round(v - 0.2, 6), 100: round(v - 0.1, 6), 200: round(v, 6)}
    secs = {k: {50: 1.0, 100: 2.0, 200: 4.0} for k in traj}
    return keys_off, twin_map, traj, secs


def _row(dataset, key, accs, *, twin_of=None, seed=0, stage="s1_catalog", error=None, budgets=(50, 100, 200)):
    row = {"dataset": dataset, "cls_key": key, "seed": seed, "stage": stage, "error": error,
           "acc": (accs[-1] if accs else None), "iters": budgets[-1], "checkpoints": list(budgets),
           "checkpoint_results": ([{"iters": b, "acc": a, "elapsed_pretrain_s": 1.0 * i, "eval_seconds": 0.1}
                                   for i, (b, a) in enumerate(zip(budgets, accs))] if accs else None)}
    if twin_of is not None:
        row["twin_of"] = twin_of
    return row


# ---------------------------------------------------------------- 配置真源
@pytest.fixture(scope="module")
def cfg():
    return g45.load_cfg()


def test_cfg_schema_and_sources(cfg):
    run = g45.run_targets(cfg)
    assert len(run) == 23 and len(set(run)) == 23
    assert "InsectWingbeat" not in run and cfg["datasets"]["InsectWingbeat"]["run"] is False
    assert all(d not in run for d in SIX) and "HAR" not in run
    for d in run:
        c = cfg["datasets"][d]
        assert c["class"] in g45.CLASSES, d
        lit = c["literature"]
        assert set(g45.LIT_COLS) <= set(lit) and all(0.0 < float(lit[k]) <= 1.0 for k in g45.LIT_COLS), d
        assert d not in g35.TABLE10, d                                  # 文献数不得双真源
        assert c["anchor_4090"]["pretrain_s"] > 0 and c["anchor_4090"]["eval_s"] > 0, d
        assert 0.0 <= float(c["chance"]) <= 1.0, d
        if c["class"] in ("A", "B", "D"):
            lo, hi = c["band"]
            assert 0.0 <= lo < hi <= 1.0, d
        else:
            assert c.get("band") is None, d
    counts = {k: sum(cfg["datasets"][d]["class"] == k for d in run) for k in g45.CLASSES}
    assert counts == {"A": 7, "B": 3, "C": 3, "D": 5, "E": 5}
    assert cfg["arms"]["main"] == g45.MAIN_ARM == "SH10-4-3_prior"
    assert cfg["arms"]["secondary"] == g45.SECOND_ARM == "SH20-3_both"
    assert cfg["twin"] == {"input_mask": "binomial"}


def test_cfg_reused_sizes_match_existing_truth(cfg):
    sizes = g45.sizes_from_cfg(cfg, list(SIX) + ["HAR"])
    for d in SIX:
        assert sizes[d] == g35.SIZES_SURVEY[d], d
    assert sizes["HAR"] == g39.SIZES_EXTRA["HAR"]
    for d in SIX + ("HAR",):
        assert cfg["datasets"][d]["run"] is False and cfg["datasets"][d].get("reuse")


def test_run_targets_follow_order_field(cfg):
    run = g45.run_targets(cfg)
    orders = [cfg["datasets"][d]["order"] for d in run]
    assert orders == sorted(orders) and len(set(orders)) == len(orders)


def test_iters_rule_from_cfg_sizes(cfg):
    p = _PROTOS["ts2vec_official"]
    for d in g45.run_targets(cfg):
        n, L, C = g45.sizes_from_cfg(cfg, [d])[d]
        assert g35.resolve_iters_from_size(p, n * L * C) == cfg["datasets"][d]["iters"], d
    assert cfg["datasets"]["PenDigits"]["iters"] == 600            # 7494×8×2 = 119904 > 1e5（临界）
    assert cfg["datasets"]["JapaneseVowels"]["iters"] == 200
    assert cfg["datasets"]["EigenWorms"].get("max_train_length") == 3000
    assert sum("max_train_length" in cfg["datasets"][d] for d in g45.run_targets(cfg)) == 1


# ---------------------------------------------------------------- 文献参照
def test_paper_reference_none_for_unknown_and_passthrough(cfg):
    assert g45.paper_reference("NoSuchDataset", cfg) is None
    assert g45.paper_ts2vec("NoSuchDataset", cfg) is None
    assert g45.paper_reference("Handwriting", cfg)["TS2Vec"] == pytest.approx(0.515)
    assert g45.paper_reference("HAR", cfg)["TS2Vec"] == pytest.approx(0.930)
    r = g45.paper_reference("RacketSports", cfg)
    assert r is not None and set(g45.LIT_COLS) <= set(r)
    assert g45.paper_ts2vec("RacketSports", cfg) == pytest.approx(cfg["datasets"]["RacketSports"]["literature"]["TS2Vec"])


def test_default_band_by_class():
    assert g45.default_band("A", 0.90, 0.5) == pytest.approx((0.87, 1.0))
    assert g45.default_band("B", 0.50, 0.3) == pytest.approx((0.40, 0.55))
    assert g45.default_band("B", 0.98, 0.3)[1] == pytest.approx(1.0)
    assert g45.default_band("D", 0.50, 0.52) == pytest.approx((0.49, 0.55))
    assert g45.default_band("A", 0.01, 0.5)[0] == pytest.approx(0.0)
    assert g45.default_band("C", 0.5, 0.3) is None and g45.default_band("E", 0.5, 0.3) is None


def test_noise_sd():
    assert g45.noise_sd(0.5, 100) == pytest.approx(0.05)
    assert g45.noise_sd(0.0, 15) == pytest.approx(0.0)


# ---------------------------------------------------------------- 候选目录（先验金标）
_MX3 = ROOT / "results" / "mx3_pq" / "rprime_cache.jsonl"


@pytest.mark.skipif(not (_MX3.exists() and (ROOT / "results" / "mx4_deploy" / "candidates.json").exists()
                         and (ROOT / "results" / "mx4_deploy_har" / "candidates.json").exists()),
                    reason="缺 mx3_pq 缓存或 WP3/WP3b 候选产物")
def test_build_candidates_prior_golden(cfg, tmp_path):
    out = g45.build_candidates(cfg, tmp_path, ["Handwriting", "SelfRegulationSCP2", "RacketSports", "HAR"])
    wp3 = json.loads((ROOT / "results" / "mx4_deploy" / "candidates.json").read_text(encoding="utf-8"))
    har = json.loads((ROOT / "results" / "mx4_deploy_har" / "candidates.json").read_text(encoding="utf-8"))
    assert out["Handwriting"]["prior"] == wp3["Handwriting"]["prior"]           # LODO，逐值
    assert out["Handwriting"]["prior_order"] == wp3["Handwriting"]["prior_order"]
    assert out["RacketSports"]["prior"] == har["HAR"]["prior"]                  # 非池靶 = 全 8 源
    assert out["HAR"]["prior"] == har["HAR"]["prior"]
    assert "LODO" in out["SelfRegulationSCP2"]["fit_on"] and out["RacketSports"]["fit_on"] == "all 8 source datasets"
    for d in out:
        assert out[d]["nominated"] == [] and out[d]["tclsc_order"] == []
        assert len(out[d]["shared"]) == 10 and out[d]["ggs"]["cls_key"] in out[d]["prior"]
        assert len(out[d]["prior_order"]) == 11
    saved = json.loads((tmp_path / "candidates.json").read_text(encoding="utf-8"))
    assert saved == out


# ---------------------------------------------------------------- 官方 max_train_length 分段（加载层）
def test_segment_train_official_semantics():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 17984, 6)).astype(np.float32)
    S = segment_train(X, 3000)
    assert S.shape == (10, 3597, 6) and S.dtype == np.float32          # 17984 // 3000 = 5 段，每段 ceil = 3597
    np.testing.assert_array_equal(S[0], X[0, :3597])                   # 首段逐位
    np.testing.assert_array_equal(S[4, :3596], X[0, 4 * 3597:])        # 末段 3596 步真实
    assert np.all(S[4, 3596] == 0)                                     # 末段补 1 步零
    np.testing.assert_array_equal(S[5], X[1, :3597])                   # 第二条样本的首段紧随其后
    for L in (3000, 2500, 5999):
        Y = rng.normal(size=(3, L, 2)).astype(np.float32)
        assert segment_train(Y, 3000) is Y                             # sections < 2 → 原样返回同一对象
    Z = rng.normal(size=(1, 6000, 2)).astype(np.float64)
    assert segment_train(Z, 3000).shape == (2, 3000, 2) and segment_train(Z, 3000).dtype == np.float64


# ---------------------------------------------------------------- 单槽数组缓存 + Xtr_fit
def _fake_pair(name, *, cache_dir=None, crop_len=None, scaler="split"):
    L = {"A": 40, "B": 9000}[name]
    rng = np.random.default_rng(len(name))
    Xtr = rng.normal(size=(4, L, 2)).astype(np.float32)
    Xte = rng.normal(size=(3, L, 2)).astype(np.float32)
    return Xtr, np.array([0, 1, 0, 1]), Xte, np.array([0, 1, 1])


def test_uea30_arrays_single_slot_and_xtr_fit(monkeypatch):
    calls = []

    def load(name, **kw):
        calls.append((name, kw.get("scaler"), kw.get("cache_dir")))
        return _fake_pair(name, **kw)

    cfg = {"datasets": {"A": {"run": True}, "B": {"run": True, "max_train_length": 3000}}}
    g45.reset_slot()
    a = g45.uea30_arrays("A", cfg, cache_dir="cd", load=load)
    assert calls == [("A", "train", "cd")]                             # 只走 scaler=train 一路
    assert a["Xte_split"] is a["Xte_train"] and "Xtr_fit" not in a
    assert set(a) >= {"Xtr", "ytr", "Xte_train", "Xte_split", "yte"}
    a2 = g45.uea30_arrays("A", cfg, cache_dir="cd", load=load)
    assert a2 is a and len(calls) == 1                                 # 同靶命中槽
    b = g45.uea30_arrays("B", cfg, cache_dir="cd", load=load)
    assert len(calls) == 2 and b["Xtr_fit"].shape == (12, 3000, 2)     # 9000 // 3000 = 3 段
    assert g45.current_slot_dataset() == "B"
    g45.uea30_arrays("A", cfg, cache_dir="cd", load=load)
    assert len(calls) == 3 and g45.current_slot_dataset() == "A"       # 换靶即释放（B 已不在槽）


def test_make_loader_signature_matches_execute():
    cfg = {"datasets": {"A": {"run": True}}}
    g45.reset_slot()
    loader = g45.make_loader(cfg, cache_dir=None, load=lambda name, **kw: _fake_pair(name, **kw))
    a = loader("A", None)                                              # 39.execute 的调用形态 loader(dataset, har_npz)
    assert a["Xtr"].shape == (4, 40, 2)


# ---------------------------------------------------------------- OOM 重复剔除
def test_drop_repeated_oom(tmp_path, capsys):
    log = tmp_path / "oom_failures.jsonl"
    log.write_text("\n".join(json.dumps({"key": k}) for k in ("k1", "k1", "k2")) + "\n", encoding="utf-8")
    cells = [{"key": "k1", "dataset": "X"}, {"key": "k2", "dataset": "X"}, {"key": "k3", "dataset": "X"}]
    kept = g45.drop_repeated_oom(cells, log, max_attempts=2)
    assert [c["key"] for c in kept] == ["k2", "k3"]
    assert "k1" in capsys.readouterr().out
    assert g45.drop_repeated_oom(cells, tmp_path / "none.jsonl") == cells


# ---------------------------------------------------------------- 臂
def test_uea30_arms_definitions():
    keys_off, twin_map, traj, secs = _traj22()
    prior_order = keys_off                                             # L00 先验第一 … L10 垫底
    row = g45.uea30_arms(keys_off, twin_map, traj, prior_order, budgets=(50, 100, 200), seconds=secs)
    a, c, p = row["arms"], row["cost_iters"], row["picks"]
    assert a["oracle22"] == pytest.approx(0.95) and a["oracle_off"] == pytest.approx(0.95)
    assert a["oracle_on"] == pytest.approx(0.92)
    assert a["prior1"] == pytest.approx(0.60) and a["prior3"] == pytest.approx(0.70)
    assert a["random1"] == pytest.approx(np.mean([traj[k][200] for k in keys_off]))
    assert p["SH6-3-3_prior"] == "L03" and c["SH6-3-3_prior"] == 6 * 50 + 3 * 50 + 3 * 100
    assert p["SH10-4-3_prior"] == "L03" and a["SH10-4-3_prior"] == pytest.approx(0.90)
    assert c["SH10-4-3_prior"] == 10 * 50 + 4 * 50 + 3 * 100
    assert p["SH20-4-3_both"] == "T03" and c["SH20-4-3_both"] == 20 * 50 + 4 * 50 + 3 * 100
    assert p["SH20-3_both"] == "T03" and a["SH20-3_both"] == pytest.approx(0.92)
    assert c["SH20-3_both"] == 20 * 100 + 3 * 100                      # 两级：从 50% 截点起
    assert row["regret"]["SH10-4-3_prior"] == pytest.approx(0.05)
    assert row["n_candidates"] == 22 and row["n_on"] == 11 and row["budgets"] == [50, 100, 200]
    assert row["lists"]["SH10-4-3_prior"] == keys_off[:10]
    assert row["lists"]["SH20-3_both"] == [x for k in keys_off[:10] for x in (k, twin_map[k])]
    assert row["cost_seconds"]["SH20-3_both"] == pytest.approx(20 * 2.0 + 3 * (4.0 - 2.0))
    assert row["is_twin"]["SH20-3_both"] is True and row["is_twin"]["SH10-4-3_prior"] is False


def test_uea30_arms_matches_39_deploy_arms_on_sh6():
    keys_off, twin_map, traj, secs = _traj22()
    row = g45.uea30_arms(keys_off, twin_map, traj, keys_off, budgets=(50, 100, 200), seconds=secs)
    ref = g39.deploy_arms(keys_off, traj, {"prior": keys_off}, budgets=(50, 100, 200), random_seed=0, seconds=secs)
    assert row["arms"]["SH6-3-3_prior"] == ref["arms"]["SH6-3-3_prior"]
    assert row["picks"]["SH6-3-3_prior"] == ref["picks"]["SH6-3-3_prior"]
    assert row["cost_iters"]["SH6-3-3_prior"] == ref["cost_iters"]["SH6-3-3_prior"]
    assert row["arms"]["prior3"] == ref["arms"]["prior3"]


def test_uea30_arms_with_missing_twins_shrinks_lists():
    keys_off, twin_map, traj, secs = _traj22()
    twin_map = {k: v for k, v in twin_map.items() if k != "L05"}
    row = g45.uea30_arms(keys_off, twin_map, traj, keys_off, budgets=(50, 100, 200))
    assert len(row["lists"]["SH20-3_both"]) == 19 and row["n_on"] == 10
    assert row["cost_iters"]["SH20-3_both"] == 19 * 100 + 3 * 100


@pytest.mark.skipif(not ((ROOT / "results" / "mx4_deploy" / "runs.jsonl").exists()
                         and (ROOT / "results" / "mx4_mask" / "runs.jsonl").exists()
                         and (ROOT / "results" / "mx4_mask" / "candidates.json").exists()),
                    reason="缺 WP3 / WP6 产物")
def test_uea30_arms_golden_six_targets():
    """6 靶复算：主臂 / 副臂冠军键与 3 种子均值 == results/mx4_mask/paper_table.md（2026-09-06）。"""
    g40 = importlib.import_module("scripts.40_mask_field_sim")
    gold = {  # (SH10-4-3 pick, mean), (SH20-3 pick, mean, is_twin)
        "Handwriting": (("c92058f2", 0.531), ("c92058f2", 0.531, False)),
        "UWaveGestureLibrary": (("8c4aa283", 0.918), ("c6fc34c5", 0.921, True)),
        "PhonemeSpectra": (("4f7d4135", 0.256), ("a59dc617", 0.251, True)),
        "EthanolConcentration": (("cffb5503", 0.303), ("382ca622", 0.314, True)),
        "Cricket": (("c92058f2", 0.986), ("c92058f2", 0.986, False)),
        "HandMovementDirection": (("9092e36e", 0.450), ("382ca622", 0.459, True)),
    }
    per = g45.reused_six_targets()
    for d, ((k10, m10), (k20, m20, tw)) in gold.items():
        v = per[d]
        assert v["champion_key"].startswith(k10) and round(v["champion_mean"], 3) == pytest.approx(m10), d
        assert v["secondary_key"].startswith(k20) and round(v["secondary_mean"], 3) == pytest.approx(m20), d
        assert v["secondary_is_twin"] is tw, d
        assert v["n_candidates"] >= 20                        # PS 的 20be4f74 在 WP3 评估超时（error 行），legacy 只剩 10
    _ = g40


# ---------------------------------------------------------------- 单靶分析（合成 22 格）
def test_analyze_target_on_synthetic_rows_reports_arms_and_rank_spearman():
    keys_off, twin_map, traj, _ = _traj22()
    cand = {"shared": [{"cls_key": k, "cls": {}} for k in keys_off[:10]], "ggs": {"cls_key": keys_off[10], "cls": {}},
            "prior": {k: float(-i) for i, k in enumerate(keys_off)}, "nominated": []}
    rows = [_row("X", k, [traj[k][50], traj[k][100], traj[k][200]]) for k in keys_off]
    rows += [_row("X", t, [traj[t][50], traj[t][100], traj[t][200]], twin_of=k) for k, t in twin_map.items()]
    s2 = [{"dataset": "X", "cls_key": "L03", "stage": "s2_finals", "acc": a, "error": None} for a in (0.88, 0.92)]
    v = g45.analyze_target("X", rows, cand, s2, klass="E", paper=0.85, band=None, chance=0.2, n_test=100)
    assert v["complete"] is True and v["n_candidates"] == 22 and v["n_on"] == 11
    assert v["champion_key"] == "L03" and v["champion_seed0"] == pytest.approx(0.90)
    assert v["champion_mean"] == pytest.approx(0.90) and v["champion_n"] == 3          # (0.90 + 0.88 + 0.92) / 3
    assert v["secondary_key"] == "T03" and v["secondary_mean"] is None and v["secondary_is_twin"] is True
    assert v["delta_secondary_minus_main"] == pytest.approx(0.92 - 0.90)               # 副臂无 finals → 用 seed 0 差
    assert v["spearman_rank25_vs_100"] == pytest.approx(1.0)                           # 合成轨迹单调 → 截点排名一致
    assert v["noise_sd"] == pytest.approx(g45.noise_sd(0.85, 100))


# ---------------------------------------------------------------- 完整性
def test_completeness_and_error_visibility():
    cand = _cands11(("X",))["X"]
    keys = [e["cls_key"] for e in cand["shared"]] + [cand["ggs"]["cls_key"]]
    rows = [_row("X", k, [0.3, 0.4, 0.5]) for k in keys]
    rows += [_row("X", k + "t", [0.3, 0.4, 0.5], twin_of=k) for k in keys[:-1]]
    rows.append(_row("X", keys[-1] + "t", None, twin_of=keys[-1], error="EvalTimeout: boom"))
    c = g45.completeness(rows, cand)
    assert c["n_legacy_ok"] == 11 and c["n_twin_ok"] == 10 and c["complete"] is False
    assert c["errors"] == [{"cls_key": keys[-1] + "t", "twin_of": keys[-1], "error": "EvalTimeout: boom"}]
    rows[-1] = _row("X", keys[-1] + "t", [0.3, 0.4, 0.5], twin_of=keys[-1])
    assert g45.completeness(rows, cand)["complete"] is True
    rows2 = [r for r in rows if r["cls_key"] != keys[0]] + [_row("X", keys[0], [0.3, 0.4])]   # 缺一个截点
    assert g45.completeness(rows2, cand)["n_legacy_ok"] == 10


# ---------------------------------------------------------------- 门
def _per(klass, champ, paper, *, band=None, second=None, complete=True):
    return {"class": klass, "champion_mean": champ, "secondary_mean": second, "paper_ts2vec": paper,
            "band": band, "complete": complete}


def _gates_cfg():
    return {"gates": {"parity_margin": 0.03, "u1_min": 3, "u2_min_rate": 0.8, "u3_tol": 0.01, "u3_min_rate": 0.6}}


def test_gates_u1_u2_u3_and_verdict():
    per = {f"E{i}": _per("E", c, 0.80, second=c + 0.005) for i, c in enumerate([0.80, 0.78, 0.775, 0.60, 0.50])}
    per.update({f"A{i}": _per("A", 0.95, 0.96, band=(0.93, 1.0), second=0.95) for i in range(4)})
    per["A4"] = _per("A", 0.80, 0.96, band=(0.93, 1.0), second=0.80)
    per["C0"] = _per("C", 0.2, 0.2)
    per["Xinc"] = _per("E", None, 0.5, complete=False)
    g = g45.run_gates_uea30(per, _gates_cfg())
    assert g["U1"]["n_parity"] == 3 and g["U1"]["n"] == 5 and g["U1"]["pass"] is True     # 不完整靶不入分母
    assert g["U2"]["n_in_band"] == 4 and g["U2"]["n"] == 5 and g["U2"]["pass"] is True    # 4/5 = 0.8
    assert g["U3"]["n"] == 10 and g["U3"]["pass"] is True and g["U3"]["mean_delta"] == pytest.approx(0.0025)
    assert g["C"]["datasets"] == ["C0"] and g["incomplete"] == ["Xinc"]
    assert g45.verdict_code_uea30(g) == "UEA30-WIN+TWIN"
    per["A0"]["champion_mean"] = 0.5
    g = g45.run_gates_uea30(per, _gates_cfg())
    assert g["U2"]["pass"] is False and g45.verdict_code_uea30(g) == "UEA30-PARTIAL+TWIN"
    for k in ("E1", "E2"):
        per[k]["champion_mean"] = 0.5
    g = g45.run_gates_uea30(per, _gates_cfg())
    assert g["U1"]["pass"] is False and g45.verdict_code_uea30(g).startswith("UEA30-DEAD")
    for k in per:
        if per[k]["secondary_mean"] is not None:
            per[k]["secondary_mean"] = per[k]["champion_mean"] - 0.05
    g = g45.run_gates_uea30(per, _gates_cfg())
    assert g["U3"]["pass"] is False and g45.verdict_code_uea30(g) == "UEA30-DEAD"


# ---------------------------------------------------------------- 估时（锚点带硬件标签）
def test_dry_run_estimate_uses_anchor_with_hw_tag():
    cfg = {"datasets": {"X": {"run": True, "order": 1, "anchor_4090": {"pretrain_s": 10.0, "eval_s": 1.0}},
                        "Y": {"run": True, "order": 2, "anchor_4090": {"pretrain_s": 100.0, "eval_s": 50.0}}}}
    est = g45.estimate_hours(cfg, ["X", "Y"], launch_s=10.0)
    assert est["hardware"] == "RTX 4090"
    assert est["per_dataset"]["X"]["n_cells"] == 22
    assert est["per_dataset"]["X"]["catalog_h"] == pytest.approx((22 * 20.0 + 66 * 1.0) / 3600)
    assert est["per_dataset"]["X"]["finals_h"] == pytest.approx(4 * (10.0 + 3 * 1.0) / 3600)
    assert est["total_catalog_h"] == pytest.approx(est["per_dataset"]["X"]["catalog_h"] + est["per_dataset"]["Y"]["catalog_h"])
    assert est["conservative_h"] == pytest.approx(1.4 * (est["total_catalog_h"] + est["total_finals_h"]))


# ---------------------------------------------------------------- 全 30 合并表
def test_table30_merge_rows():
    cfg = g45.load_cfg()
    run = g45.run_targets(cfg)
    per_new = {d: {"class": cfg["datasets"][d]["class"], "complete": True, "n_candidates": 22,
                   "champion_key": "k", "champion_mean": 0.5, "champion_seed0": 0.5, "champion_std": 0.01,
                   "secondary_key": "kt", "secondary_mean": 0.51, "secondary_std": 0.01, "secondary_is_twin": True,
                   "arms": {"oracle22": 0.6}, "paper_ts2vec": g45.paper_ts2vec(d, cfg), "band": cfg["datasets"][d].get("band")}
               for d in run}
    per_six = {d: {"class": cfg["datasets"][d]["class"], "complete": True, "n_candidates": 42,
                   "champion_key": "k", "champion_mean": 0.7, "champion_seed0": 0.7, "champion_std": 0.0,
                   "secondary_key": "k", "secondary_mean": 0.7, "secondary_std": 0.0, "secondary_is_twin": False,
                   "arms": {"oracle22": 0.8}, "paper_ts2vec": g45.paper_ts2vec(d, cfg), "band": None}
               for d in SIX}
    har = {"champion_mean": 0.935, "champion_std": 0.001, "paper_ts2vec": 0.930, "arm": "SH6-3-3_prior"}
    rows = g45.table30_rows(cfg, per_new, per_six, har)
    assert len(rows) == 31 and len({r["dataset"] for r in rows}) == 31          # UEA 30 + HAR 外部锚一行
    ins = next(r for r in rows if r["dataset"] == "InsectWingbeat")
    assert ins["run"] is False and ins["main_mean"] is None and "未跑" in ins["note"]
    h = next(r for r in rows if r["dataset"] == "HAR")
    assert h["main_mean"] == pytest.approx(0.935) and h["secondary_mean"] is None and "legacy_full" in h["note"]
    means = g45.table30_means(rows)
    assert set(means) == {"all", "excl_C", "evaluable"}
    assert means["all"]["n"] == 29 and means["excl_C"]["n"] == 26 and means["evaluable"]["n"] == 12   # E5 + 6 + HAR
    assert means["evaluable"]["main"] == pytest.approx((5 * 0.5 + 6 * 0.7 + 0.935) / 12)
    md = g45.table30_markdown(rows, means)
    assert "InsectWingbeat" in md and "29/30" in md and "†" in md
