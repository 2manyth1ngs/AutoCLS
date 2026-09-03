"""全量协议对齐脚本（scripts/35_protocol_alignment.py）纯函数测试。

目标：让本栈在 UEA 官方 train/test 划分（SVM-RBF 下游）上追平已发表模型。
本文件只测零 GPU 的纯函数：协议装载与漂移闸、iters 规则、缓存键、臂的空间合法性、
zero-shot pick 的 Borda 推导（铁律 6：只用以目标数据集为折的排名）、判线阈值、
工单计数与去重、resume 跳过。
"""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pa = importlib.import_module("scripts.35_protocol_alignment")
_m1 = importlib.import_module("scripts.m1_ggs_har")

from autocls.labeling import cls_key  # noqa: E402
from autocls.space.cls_space import CLS, CLSSpace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 常量 / 臂
class TestArms:
    def test_ts2vec_equiv_is_space_legal(self):
        space = CLSSpace.from_yaml()
        assert space.validate(pa.TS2VEC_EQUIV)
        c = pa.TS2VEC_EQUIV
        # TS2Vec 目标：只开 crop、instance+temporal、kernel 2 max、dot、τ=1、InfoNCE
        assert c.crop > 0 and c.jitter == 0 and c.resize == 0 and c.point_mask == 0
        assert c.instance and c.temporal and not c.cross_scale
        assert c.kernel == 2 and c.pool == "max" and c.adj_neighbor is False
        assert c.loss_type == "info_nce" and c.sim_func == "dot" and c.temperature == 1.0
        assert c.norm == "none" and c.emb_jitter == 0 and c.emb_mask == 0

    def test_ggs_t5_outside_space_and_whitelisted(self):
        space = CLSSpace.from_yaml()
        assert not space.validate(pa.GGS_T5)          # kernel 5/avg 在两开关全关时非法
        assert space.validate(_m1.GGS)
        assert "ggs_table5" in pa.OUTSIDE_SPACE_ARMS
        assert pa.GGS_T5.kernel == 5 and pa.GGS_T5.pool == "avg"
        assert pa.GGS_T5.adj_neighbor is False

    def test_targets_and_table10(self):
        assert pa.TARGETS == ("Handwriting", "UWaveGestureLibrary", "PhonemeSpectra",
                              "EthanolConcentration", "Cricket", "HandMovementDirection")
        for d in pa.TARGETS:
            row = pa.TABLE10[d]
            assert set(row) == {"TS2Vec", "T-Loss", "TNC", "TS-TCC", "TST", "DTW"}
        assert pa.TABLE10["Handwriting"]["TS2Vec"] == 0.515
        assert pa.SEEDS == (0, 1, 2)


# ---------------------------------------------------------------- 协议
class TestProtocols:
    def test_load_protocols_from_repo_yaml(self):
        protos = pa.load_protocols(ROOT / "configs" / "train.yaml")
        assert set(protos) >= {"legacy_full", "ts2vec_official"}
        lf, ts = protos["legacy_full"], protos["ts2vec_official"]
        assert lf["iters"] == 2320 and lf["batch_size"] == 128 and lf["lr"] == 1e-3
        assert lf["drop_last"] is False and lf["swa"] is False
        assert lf["crop_mode"] == "fixed" and lf["scaler"] == "split" and lf["svm_grid"] == "legacy"
        assert ts["iters_rule"] == "ts2vec_size" and ts["batch_size"] == 8
        assert ts["drop_last"] is True and ts["swa"] is True
        assert ts["scaler"] == "train" and ts["svm_grid"] == "ts2vec"

    def test_load_protocols_drift_fail_fast(self, tmp_path):
        cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
        cfg["protocols"]["legacy_full"]["iters"] = 2000     # 与 full_iters 漂移
        p = tmp_path / "train.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        with pytest.raises(ValueError):
            pa.load_protocols(p)

    def test_existing_keys_untouched(self):
        cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
        assert cfg["full_iters"] == 2320 and cfg["early_iters"] == 290
        assert cfg["batch_size"] == 128 and cfg["budgets"] == [60, 120, 290]

    @pytest.mark.parametrize("size,expect", [(68_400, 200), (100_000, 200), (113_400, 600),
                                             (7_912_905, 600)])
    def test_resolve_iters_ts2vec_rule(self, size, expect):
        proto = {"iters_rule": "ts2vec_size"}
        X = np.zeros(size, dtype=np.float32)
        assert pa.resolve_iters(proto, X) == expect
        assert pa.ts2vec_default_iters(size) == expect

    def test_resolve_iters_fixed(self):
        assert pa.resolve_iters({"iters": 2320}, np.zeros(10)) == 2320

    def test_resolve_iters_unknown_rule(self):
        with pytest.raises(ValueError):
            pa.resolve_iters({"iters_rule": "epochs"}, np.zeros(10))

    @pytest.mark.parametrize("bs,expect", [(8, 200), (32, 44), (128, 11)])
    def test_equal_epoch_iters(self, bs, expect):
        # HW N=150，官方 200@8 drop_last → 200/18 ≈ 11.1 遍；等遍换算按 N//bs 步/遍
        assert pa.equal_epoch_iters(200, 8, bs, n_train=150, drop_last=True) == expect

    def test_run_options_canonical_and_overrides(self):
        protos = pa.load_protocols(ROOT / "configs" / "train.yaml")
        base = pa.run_options(protos["ts2vec_official"])
        assert base["input_mask"] == "none" and base["repr_dropout"] == 0.0
        assert base["engine"] == "cl_trainer"
        over = pa.run_options(protos["ts2vec_official"], {"input_mask": "binomial"})
        assert over["input_mask"] == "binomial" and over["swa"] is True
        with pytest.raises(ValueError):
            pa.run_options(protos["ts2vec_official"], {"unknown_knob": 1})


# ---------------------------------------------------------------- 缓存键 / resume
class TestCacheKey:
    def test_stable_and_sensitive(self):
        protos = pa.load_protocols(ROOT / "configs" / "train.yaml")
        opts = pa.run_options(protos["ts2vec_official"])
        k0 = pa.cache_key("Handwriting", "ts2vec_official", "abc", 0, opts, 200)
        assert k0 == pa.cache_key("Handwriting", "ts2vec_official", "abc", 0, dict(opts), 200)
        assert len(k0) == 16
        seen = {k0}
        for change in ({"dataset": "Cricket"}, {"protocol": "legacy_full"}, {"key": "abd"},
                       {"seed": 1}, {"iters": 600}, {"arm": "bs_scan_8"}):
            kw = dict(dataset="Handwriting", protocol="ts2vec_official", key="abc",
                      seed=0, iters=200, arm="")
            kw.update(change)
            k = pa.cache_key(kw["dataset"], kw["protocol"], kw["key"], kw["seed"], opts, kw["iters"],
                             arm=kw["arm"])
            assert k not in seen
            seen.add(k)
        for knob, val in (("swa", False), ("drop_last", False), ("crop_mode", "ts2vec"),
                          ("input_mask", "binomial"), ("repr_dropout", 0.1),
                          ("scaler", "split"), ("svm_grid", "legacy"), ("batch_size", 32),
                          ("engine", "ts2vec_wrapper")):
            o = dict(opts)
            o[knob] = val
            k = pa.cache_key("Handwriting", "ts2vec_official", "abc", 0, o, 200)
            assert k not in seen, knob
            seen.add(k)

    def test_load_done_and_resume_skip(self, tmp_path):
        p = tmp_path / "runs.jsonl"
        rows = [{"key": "k1", "acc": 0.5}, {"key": "k2", "acc": None, "error": "boom"}]
        p.write_text("".join(json.dumps(r) + "\n" for r in rows) + "{broken\n", encoding="utf-8")
        done = pa.load_done(p)
        assert set(done) == {"k1", "k2"}          # error 行也算已跑（不是 OOM）
        cells = [{"key": "k1"}, {"key": "k3"}]
        assert [c["key"] for c in pa.pending(cells, done)] == ["k3"]
        assert pa.load_done(tmp_path / "missing.jsonl") == {}


# ---------------------------------------------------------------- zero-shot picks
def _fake_rankings(shared, rand_per_slice):
    """3 切片：main 排名把 shared 键按不同顺序排列并混入随机键。"""
    s = list(shared)
    slices = {
        "D_00": {"main": [s[0], rand_per_slice[0], s[1], s[2]] + s[3:],
                 "cls_only": [s[1], s[0], s[2]] + s[3:] + [rand_per_slice[0]]},
        "D_01": {"main": [s[0], s[2], s[1]] + s[3:] + [rand_per_slice[1]],
                 "cls_only": [s[1], s[2], s[0]] + s[3:] + [rand_per_slice[1]]},
        "D_02": {"main": [rand_per_slice[2], s[2], s[0], s[1]] + s[3:],
                 "cls_only": [s[1], s[0], s[2]] + s[3:] + [rand_per_slice[2]]},
    }
    return {"D": {"290": slices, "60": slices}}


class TestZeroShotPicks:
    def test_borda_over_shared_keys(self):
        shared = [f"s{i:02d}" for i in range(10)]
        rand = ["r0", "r1", "r2"]
        catalog = {"shared": [{"cls_key": k, "cls": {}} for k in shared],
                   "random": {f"D_0{i}": [{"cls_key": rand[i], "cls": {}}] for i in range(3)}}
        picks = pa.zero_shot_picks(_fake_rankings(shared, rand), catalog, "D", budget=290)
        # s00 位次 0,0,1 → 9+9+8=26；s02 位次 2,1,0 → 7+8+9=24；s01 位次 1,2,2 → 8+7+7=22
        assert picks["top1"] == "s00"
        assert picks["top3"] == ["s00", "s02", "s01"]
        assert picks["borda"]["s00"] == 26 and picks["borda"]["s02"] == 24
        assert picks["per_slice_top1"] == ["s00", "s00", "r2"]
        assert picks["cls_only_top1"] == "s01"
        assert picks["n_slices"] == 3 and picks["budget"] == 290

    def test_only_target_fold_is_read(self):
        # 铁律 6：pick 只来自 rankings[<target>]，其他数据集的折不参与
        shared = [f"s{i:02d}" for i in range(10)]
        rk = _fake_rankings(shared, ["r0", "r1", "r2"])
        rk["OTHER"] = {"290": {"OTHER_00": {"main": list(reversed(shared))}}}
        catalog = {"shared": [{"cls_key": k, "cls": {}} for k in shared], "random": {}}
        picks = pa.zero_shot_picks(rk, catalog, "D", budget=290)
        assert picks["top1"] == "s00"

    def test_tie_breaks_by_key(self):
        shared = ["b", "a"]
        rk = {"D": {"290": {"D_00": {"main": ["b", "a"]}, "D_01": {"main": ["a", "b"]}}}}
        catalog = {"shared": [{"cls_key": k, "cls": {}} for k in shared], "random": {}}
        picks = pa.zero_shot_picks(rk, catalog, "D", budget=290)
        assert picks["top1"] == "a" and picks["cls_only_top1"] is None

    def test_missing_dataset_raises(self):
        with pytest.raises(KeyError):
            pa.zero_shot_picks({"D": {}}, {"shared": [], "random": {}}, "E")

    def test_real_rankings_give_shared_keys(self):
        rp = ROOT / "results" / "mx3_pq" / "rankings.json"
        cp = ROOT / "results" / "mx3_pq" / "cls_catalog.json"
        if not (rp.exists() and cp.exists()):
            pytest.skip("mx3_pq 产物缺失")
        rk = json.loads(rp.read_text(encoding="utf-8"))
        cat = json.loads(cp.read_text(encoding="utf-8"))
        shared = {d["cls_key"] for d in cat["shared"]}
        for d in pa.TARGETS:
            picks = pa.zero_shot_picks(rk, cat, d, budget=290)
            assert picks["top1"] in shared and set(picks["top3"]) <= shared
            assert picks["n_slices"] == 6 and len(picks["per_slice_top1"]) == 6


# ---------------------------------------------------------------- 判线
class TestParity:
    @pytest.mark.parametrize("ds,median", [
        ("Handwriting", 0.3685), ("UWaveGestureLibrary", 0.817), ("PhonemeSpectra", 0.2145),
        ("EthanolConcentration", 0.291), ("Cricket", 0.972), ("HandMovementDirection", 0.2835),
    ])
    def test_thresholds(self, ds, median):
        th = pa.parity_thresholds(pa.TABLE10[ds])
        assert th["median"] == pytest.approx(median)
        assert th["line"] == pytest.approx(pa.TABLE10[ds]["TS2Vec"] - pa.PARITY_MARGIN)

    def test_verdict_levels(self):
        row = pa.TABLE10["Handwriting"]        # ts2vec 0.515 → line 0.485，median 0.3685
        assert pa.verdict(0.49, row) == "parity"
        assert pa.verdict(0.40, row) == "median"
        assert pa.verdict(0.30, row) == "below"
        assert pa.verdict(float("nan"), row) == "missing"

    def test_p1_gate(self):
        assert pa.p1_pass([0.49, 0.50, 0.47]) is True
        assert pa.p1_pass([0.49, 0.50, 0.46]) is False     # 单种子 <0.465
        assert pa.p1_pass([0.485, 0.485, 0.484]) is False  # 均值 <0.485
        assert pa.p1_pass([0.49, 0.50]) is False           # 种子不齐


# ---------------------------------------------------------------- 工单
class TestWorklists:
    _sizes = {"Handwriting": (150, 152, 3), "UWaveGestureLibrary": (120, 315, 3),
              "PhonemeSpectra": (3315, 217, 11), "EthanolConcentration": (261, 1751, 3),
              "Cricket": (108, 1197, 6), "HandMovementDirection": (160, 400, 10)}

    def _ctx(self):
        protos = pa.load_protocols(ROOT / "configs" / "train.yaml")
        return protos

    def test_reproduce_worklist_12(self):
        protos = self._ctx()
        cells = pa.reproduce_worklist(protos, self._sizes, datasets=("Handwriting",), seeds=pa.SEEDS)
        assert len(cells) == 12
        grids = {(c["protocol"], c["options"]["swa"]) for c in cells}
        assert grids == {("ts2vec_official", True), ("ts2vec_official", False),
                         ("legacy_full", False), ("legacy_full", True)}
        assert {c["iters"] for c in cells if c["protocol"] == "ts2vec_official"} == {200}
        assert {c["iters"] for c in cells if c["protocol"] == "legacy_full"} == {2320}
        assert all(c["cls_key"] == cls_key(pa.TS2VEC_EQUIV) for c in cells)
        assert len({c["key"] for c in cells}) == 12

    def test_bs_scan_worklist_9(self):
        protos = self._ctx()
        cells = pa.bs_scan_worklist(protos, self._sizes, seeds=pa.SEEDS)
        assert len(cells) == 9
        assert {(c["options"]["batch_size"], c["iters"]) for c in cells} == {(8, 200), (32, 44), (128, 11)}
        assert all(c["stage"] == "bs_scan" and c["dataset"] == "Handwriting" for c in cells)

    def test_eval_audit_worklist(self):
        protos = self._ctx()
        a1 = pa.audit_baseline_worklist(protos, self._sizes, datasets=pa.TARGETS, seeds=pa.SEEDS)
        assert len(a1) == 18 and all(c["iters"] == 0 for c in a1)
        a2 = pa.audit_legacy_worklist(protos, self._sizes, champion_key="f48e5da1d899",
                                      champion_cls=pa.TS2VEC_EQUIV)
        assert len(a2) == 2 and all(c["protocol"] == "legacy_full" and c["seed"] == 0 for c in a2)

    def test_ladder_worklist(self):
        protos = self._ctx()
        cells = pa.ladder_worklist(protos, self._sizes, seeds=pa.SEEDS)
        assert len(cells) == 12
        rungs = {c["arm"] for c in cells}
        assert rungs == {"ladder_crop_ts2vec", "ladder_input_mask", "ladder_repr_dropout", "ladder_all"}
        full = [c for c in cells if c["arm"] == "ladder_all"][0]["options"]
        assert full["crop_mode"] == "ts2vec" and full["input_mask"] == "binomial" and full["repr_dropout"] == 0.1

    def test_target_arms_dedup(self):
        # 去重按 CLS **内容**的规范键（cls_key(cls)），不按调用方给的标签串
        kA, kB, kC = cls_key(pa.TS2VEC_EQUIV), cls_key(_m1.GGS), cls_key(pa.GGS_T5)
        picks = {"top1": kA, "top3": [kA, kB, kC], "cls_only_top1": kB}
        key2cls = {kA: pa.TS2VEC_EQUIV, kB: _m1.GGS, kC: pa.GGS_T5}
        arms = pa.target_arms(picks, key2cls)
        keys = [a["cls_key"] for a in arms]
        assert len(keys) == len(set(keys)) == 3                  # 同一 CLS 只跑一次
        by_key = {a["cls_key"]: a for a in arms}
        assert by_key[kA]["arms"] == ["ts2vec_equiv", "zs_top1", "zs_top3"]
        assert by_key[kB]["arms"] == ["ggs", "zs_top3", "zs_clsonly_top1"]
        assert by_key[kC]["arms"] == ["ggs_table5", "zs_top3"]
        assert by_key[kA]["space_legal"] and by_key[kB]["space_legal"]
        assert by_key[kC]["space_legal"] is False
        # cls_only 与 top1 相同时不产生额外臂名以外的格
        picks2 = {"top1": kA, "top3": [kA], "cls_only_top1": kA}
        arms2 = pa.target_arms(picks2, key2cls)
        assert [a["cls_key"] for a in arms2] == sorted({kA, kB, kC}) or len(arms2) == 3

    def test_targets_worklist_count(self):
        protos = self._ctx()
        kA, kB, kC = cls_key(pa.TS2VEC_EQUIV), cls_key(_m1.GGS), cls_key(pa.GGS_T5)
        picks = {d: {"top1": kA, "top3": [kA, kB, kC], "cls_only_top1": kA} for d in pa.TARGETS}
        key2cls = {kA: pa.TS2VEC_EQUIV, kB: _m1.GGS, kC: pa.GGS_T5}
        cells = pa.targets_worklist(protos, self._sizes, picks, key2cls,
                                    datasets=pa.TARGETS, seeds=pa.SEEDS)
        # 每数据集唯一 CLS = {TS2VEC_EQUIV, GGS, GGS_T5} 3 个 × 3 种子 × 6 数据集
        assert len(cells) == 54
        assert all(c["protocol"] == "ts2vec_official" for c in cells)
        hw = [c for c in cells if c["dataset"] == "Handwriting"]
        assert {c["iters"] for c in hw} == {200}
        assert {c["iters"] for c in cells if c["dataset"] != "Handwriting"} == {600}
        assert {c["space_legal"] for c in cells} == {True, False}
        assert len({c["key"] for c in cells}) == 54

    def test_official_worklist(self):
        protos = self._ctx()
        cells = pa.official_worklist(protos, self._sizes, datasets=pa.TARGETS, seeds=pa.SEEDS)
        assert len(cells) == 18 and all(c["options"]["engine"] == "ts2vec_wrapper" for c in cells)
        assert all(c["cls_key"] == "official_ts2vec" for c in cells)


# ---------------------------------------------------------------- 聚合
class TestAggregate:
    def test_summarize_arm(self):
        rows = [{"dataset": "Handwriting", "arm_names": ["zs_top1", "zs_top3"], "cls_key": "A",
                 "seed": s, "acc": a, "f1": a - 0.01, "svm_cv_score": 0.5}
                for s, a in zip((0, 1, 2), (0.50, 0.48, 0.49))]
        summ = pa.summarize_seeds(rows)
        assert summ["n"] == 3 and summ["mean"] == pytest.approx(0.49)
        assert summ["std"] == pytest.approx(np.std([0.50, 0.48, 0.49]))
        assert summ["accs"] == [0.50, 0.48, 0.49]

    def test_pick_top3_best_of_and_cv_selected(self):
        per_cls = {"A": {"mean": 0.50, "cv_mean": 0.60}, "B": {"mean": 0.55, "cv_mean": 0.55},
                   "C": {"mean": 0.45, "cv_mean": 0.70}}
        best, cv_sel = pa.top3_selection(["A", "B", "C"], per_cls)
        assert best == "B" and cv_sel == "C"
