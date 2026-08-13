"""SWA 噪声探针脚本（scripts/7_swa_noise_probe.py）纯逻辑测试。

期望值全部手工推导（不以被测代码为 oracle）：
- 种子派生必须与 2c 噪声探针逐字同口径（否决条件 2 的前提）；
- P95 取法必须与 2c calibrate 同式（δ 可比性）；
- 任务盲天花板 = 任务内配对多数方向可达 acc（outlet_y 口径）。
"""
import importlib
import json

import pytest

swa_probe = importlib.import_module("scripts.7_swa_noise_probe")
_c2 = importlib.import_module("scripts.2c_generate_labels_workpoint")


class TestSeedsFor:
    def test_rep0_is_canonical(self):
        # rep0 = 2c canonical 口径（labels_workpoint rprime_cache 的键）
        assert swa_probe.seeds_for("t0", "k0", 0) == _c2.canonical_seeds("t0", "k0")

    def test_rep12_are_noise_seeds_with_canonical_split(self):
        from autocls.labeling import run_seed_for

        for rep in (1, 2):
            w, sp = swa_probe.seeds_for("t0", "k0", rep)
            assert w == run_seed_for("t0", f"k0:noise{rep}")  # 2c _noise_loop 同式
            assert sp == run_seed_for("t0", "train_eval_split")

    def test_reps_have_distinct_weight_seeds(self):
        ws = {swa_probe.seeds_for("t0", "k0", rep)[0] for rep in (0, 1, 2)}
        assert len(ws) == 3


class TestBuildCells:
    def test_shape_and_iters(self):
        # 2 数据集 × min(任务数, NOISE_TASKS=2) × 2 CLS × 3 rep = 24
        class T:  # 最小任务桩
            def __init__(self, tid):
                self.task_id = tid

        per_ds = [("A", None, None, [T("a0"), T("a1"), T("a2")]),
                  ("B", None, None, [T("b0"), T("b1")])]
        cells = swa_probe.build_cells(
            per_ds, {"A": 60, "B": 30}, [("c1", "k1"), ("c2", "k2")])
        assert len(cells) == 24
        assert {c[7] for c in cells if c[0] == "A"} == {60}
        assert {c[7] for c in cells if c[0] == "B"} == {30}
        assert {c[3].task_id for c in cells if c[0] == "A"} == {"a0", "a1"}  # 前2任务
        assert {c[6] for c in cells} == {0, 1, 2}


class TestP95:
    def test_matches_2c_index_method(self):
        # 21 个值 0..20：索引 round(0.95×20)=19 → 值 19（取元素不插值，2c 同式）
        assert swa_probe._p95(list(range(21))) == 19
        # 3 个值：round(0.95×2)=2 → 最大值
        assert swa_probe._p95([0.1, 0.3, 0.2]) == 0.3

    def test_single_value(self):
        assert swa_probe._p95([0.5]) == 0.5


class TestLoadCells:
    def test_groups_by_arm_and_last_row_wins(self):
        rows = [
            {"swa": False, "dataset": "A", "task_id": "t", "cls_key": "k",
             "seed": 0, "r_prime": 0.5},
            {"swa": True, "dataset": "A", "task_id": "t", "cls_key": "k",
             "seed": 0, "r_prime": 0.6},
            {"swa": False, "dataset": "A", "task_id": "t", "cls_key": "k",
             "seed": 0, "r_prime": 0.7},   # 同键后行覆盖前行（resume 追加语义）
            {"swa": False, "dataset": "A", "task_id": "t", "cls_key": "k",
             "seed": 1, "r_prime": None},  # null 不进 cell
        ]
        cells = swa_probe.load_cells(rows)
        assert cells[(False, "A", "t", "k")] == {0: 0.7}
        assert cells[(True, "A", "t", "k")] == {0: 0.6}


class TestBlindCeiling:
    def _cells(self):
        # 2 任务 × 2 CLS（rep-mean 直接给单 rep）：
        # 任务 t1：ka=0.9, kb=0.7（ka 胜 0.2）；任务 t2：ka=0.8, kb=0.85（kb 胜 0.05）
        return {
            (True, "A", "t1", "ka"): {0: 0.9}, (True, "A", "t1", "kb"): {0: 0.7},
            (True, "A", "t2", "ka"): {0: 0.8}, (True, "A", "t2", "kb"): {0: 0.85},
        }

    def test_delta0_split_votes_gives_half(self):
        # δ=0：(ka,kb) 两任务方向相反 → 多数方向只对 1/2
        out = swa_probe.blind_ceiling(self._cells(), True, {})
        assert out["n_pairs"] == 2
        assert out["ceiling"] == 0.5

    def test_delta_filters_small_margin_pair(self):
        # δ_A=0.1：t2 的 0.05 差被过滤 → 只剩 t1 一票，任务盲全对
        out = swa_probe.blind_ceiling(self._cells(), True, {"A": 0.1})
        assert out["n_pairs"] == 1
        assert out["ceiling"] == 1.0

    def test_arm_isolation(self):
        # off 臂数据不得混入 on 臂计算
        cells = {(False, "A", "t1", "ka"): {0: 0.9},
                 (False, "A", "t1", "kb"): {0: 0.1}}
        out = swa_probe.blind_ceiling(cells, True, {})
        assert out["n_pairs"] == 0 and out["ceiling"] is None


class TestBitwiseCheck:
    def test_match_and_mismatch_counted(self, tmp_path, monkeypatch):
        (tmp_path / "rprime_cache.jsonl").write_text(
            json.dumps({"task_id": "t", "cls_key": "k", "iters": 60,
                        "r_prime": 0.5}) + "\n", encoding="utf-8")
        (tmp_path / "noise_probe.jsonl").write_text(
            json.dumps({"task_id": "t", "cls_key": "k", "iters": 60, "seed": 1,
                        "r_prime": 0.6}) + "\n", encoding="utf-8")
        monkeypatch.setattr(swa_probe, "WORKPOINT_DIR", tmp_path)
        rows = [
            {"swa": False, "task_id": "t", "cls_key": "k", "iters": 60,
             "seed": 0, "r_prime": 0.5},          # rep0 ↔ rprime_cache 逐位同
            {"swa": False, "task_id": "t", "cls_key": "k", "iters": 60,
             "seed": 1, "r_prime": 0.61},         # rep1 ↔ noise_probe 差 0.01
            {"swa": False, "task_id": "t", "cls_key": "k", "iters": 60,
             "seed": 2, "r_prime": 0.7},          # 无参考
            {"swa": True, "task_id": "t", "cls_key": "k", "iters": 60,
             "seed": 0, "r_prime": 0.9},          # on 臂不参与比对
        ]
        out = swa_probe.bitwise_check(rows, {})
        assert out["n_off"] == 3
        assert out["n_match"] == 1
        assert out["n_missing_ref"] == 1
        assert out["max_abs_diff"] == pytest.approx(0.01)
        assert out["passed"] is False  # 120 条全同才 PASS
