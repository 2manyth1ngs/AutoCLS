"""MX-3-PQ 多截点 R' 网格脚本（scripts/32_pq_generate_rprime.py）测试。

协议：.superpowers/sdd/autocts-e-autocts-autocts-e-autocts-refactored-moler/
interfaces.md「Task 1 → Task 5」「Task 5（脚本 32）」。

覆盖面：行合成（轨迹 → 缓存行）、--resume 的四态判定（done/partial/terminal/
missing）、δ 标定（手算对账 + delta_nd 剔破坏格）、--apply-delta 的**逐行**幂等
回填、--verify 的计数/重复/半条轨迹/设备混杂。

除 smoke 外全部不碰 GPU、不加载数据集（评估器一律 monkeypatch）；smoke 用真实
Handwriting/UniMiB-SHAR 切片但默认仍打桩，置 PQ_SMOKE_REAL=1 才跑真 GPU。
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import pq_common as pq

g32 = importlib.import_module("scripts.32_pq_generate_rprime")

ROOT = Path(__file__).resolve().parents[1]
B2 = (60, 120)          # 两预算的合成网格（单测用，省一半合成行）


# --------------------------------------------------------------- 合成夹具
def make_rec(r_prime, *, iters_list, seconds=1.0, error=None, n_eval=100,
             device="NVIDIA GeForce RTX 4060 Laptop GPU"):
    """evaluate_rprime_trajectory 单截点 record 的合成体（字段照 labeling.py）。"""
    return {"r_prime": r_prime, "acc": r_prime, "f1": r_prime, "svm_c": 1.0,
            "n_train": 200, "n_eval": n_eval, "weight_seed": 7, "split_seed": 11,
            "device_name": device, "swa": False, "seconds": seconds,
            "trajectory_seconds": 3.0, "error": error,
            "checkpoint_iters": list(iters_list)}


def fake_traj(values, *, iters_list, **kw):
    """{iters: rec}：values 是与 iters_list 等长的 R' 序列。"""
    return {b: make_rec(v, iters_list=iters_list, **kw)
            for b, v in zip(iters_list, values)}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def cache_row(slice_id, key, iters, r_prime, *, origin="shared", seed=0,
              error=None, n_eval=100, device="dev-A", iters_list=B2):
    """主缓存/噪声缓存的合成行（只带 32 号脚本真正会读的字段）。"""
    return {"dataset": pq.dataset_of(slice_id), "task_id": slice_id, "tag": "",
            "task_type": "classification", "classes": [0, 1], "cls_key": key,
            "origin": origin, "seed": seed, "iters": iters,
            "checkpoint_iters": list(iters_list),
            "trajectory_id": f"{slice_id}:{key}:7", "r_prime": r_prime,
            "n_eval": n_eval, "device_name": device, "error": error}


@pytest.fixture
def cls_a(make_cls):
    return make_cls()


# --------------------------------------------------------------- 行合成
class TestRowBuilder:
    def test_rows_have_required_keys(self, cls_a):
        task = g32.TaskMeta("Cricket_00", "", "classification", (0, 1))
        rows = g32.build_rows(fake_traj([0.4, 0.5], iters_list=B2), dataset="Cricket",
                              task=task, cls_key="k1", origin="shared", cls_cfg=cls_a,
                              weight_seed=7, seed=0)
        need = {"dataset", "task_id", "tag", "task_type", "classes", "cls_key",
                "origin", "seed", "iters", "checkpoint_iters", "trajectory_id",
                "r_prime", "acc", "f1", "svm_c", "n_train", "n_eval", "weight_seed",
                "split_seed", "device_name", "swa", "seconds", "trajectory_seconds",
                "error", "cls"}
        for row in rows:
            assert need <= set(row), need - set(row)
        assert rows[0]["dataset"] == "Cricket" and rows[0]["task_id"] == "Cricket_00"
        assert rows[0]["swa"] is False and rows[0]["seed"] == 0
        assert isinstance(rows[0]["cls"], dict) and "jitter" in rows[0]["cls"]

    def test_iters_ascending_and_trajectory_id(self, cls_a):
        task = g32.TaskMeta("Cricket_00", "", "classification", (0, 1))
        rows = g32.build_rows(fake_traj([0.4, 0.5], iters_list=B2), dataset="Cricket",
                              task=task, cls_key="k1", origin="shared", cls_cfg=cls_a,
                              weight_seed=7, seed=0)
        assert [r["iters"] for r in rows] == list(B2)
        assert {r["trajectory_id"] for r in rows} == {"Cricket_00:k1:7"}

    def test_origin_and_seed_propagated(self, cls_a):
        task = g32.TaskMeta("Cricket_00", "", "classification", (0, 1))
        rows = g32.build_rows(fake_traj([0.4, 0.5], iters_list=B2), dataset="Cricket",
                              task=task, cls_key="g", origin=g32.ORIGIN_GGS,
                              cls_cfg=cls_a, weight_seed=7, seed=2)
        assert {r["origin"] for r in rows} == {"ggs"}
        assert {r["seed"] for r in rows} == {2}

    def test_nan_rprime_becomes_null(self, cls_a):
        task = g32.TaskMeta("Cricket_00", "", "classification", (0, 1))
        traj = fake_traj([float("nan"), float("nan")], iters_list=B2,
                         error="RuntimeError: boom")
        rows = g32.build_rows(traj, dataset="Cricket", task=task, cls_key="k1",
                              origin="shared", cls_cfg=cls_a, weight_seed=7, seed=0)
        assert [r["r_prime"] for r in rows] == [None, None]
        assert all(r["error"] for r in rows)

    def test_checkpoint_iters_kept_verbatim(self, cls_a):
        """guarded_run_multi 会核对该字段：行必须原样带 rec 自己的那份。"""
        task = g32.TaskMeta("Cricket_00", "", "classification", (0, 1))
        rows = g32.build_rows(fake_traj([0.4, 0.5], iters_list=B2), dataset="Cricket",
                              task=task, cls_key="k1", origin="shared", cls_cfg=cls_a,
                              weight_seed=7, seed=0)
        assert all(r["checkpoint_iters"] == list(B2) for r in rows)


# --------------------------------------------------------------- 四态 / resume
class TestTrajectoryState:
    def _index(self, rows):
        return {(r["task_id"], r["cls_key"], r["iters"]): r for r in rows}

    def test_all_budgets_done(self):
        idx = self._index([cache_row("Cricket_00", "k", b, 0.5) for b in B2])
        assert g32.trajectory_state(idx, "Cricket_00", "k", B2) == g32.DONE

    def test_two_of_three_is_partial(self):
        b3 = (60, 120, 290)
        idx = self._index([cache_row("Cricket_00", "k", b, 0.5, iters_list=b3)
                           for b in (60, 120)])
        assert g32.trajectory_state(idx, "Cricket_00", "k", b3) == g32.PARTIAL

    def test_null_with_error_at_max_budget_is_terminal(self):
        """看门狗隔离只留 max budget 一条 null 行——整条轨迹终结，绝不重试。"""
        b3 = (60, 120, 290)
        idx = self._index([cache_row("Cricket_00", "k", 290, None, iters_list=b3,
                                     error="quarantined")])
        assert g32.trajectory_state(idx, "Cricket_00", "k", b3) == g32.TERMINAL

    def test_null_with_error_anywhere_is_terminal(self):
        idx = self._index([cache_row("Cricket_00", "k", 60, None, error="boom"),
                           cache_row("Cricket_00", "k", 120, None, error="boom")])
        assert g32.trajectory_state(idx, "Cricket_00", "k", B2) == g32.TERMINAL

    def test_no_rows_is_missing(self):
        assert g32.trajectory_state({}, "Cricket_00", "k", B2) == g32.MISSING


class TestResumeSelection:
    def _cells(self, cls_a):
        return [g32.Cell("Cricket_00", "shared", cls_a, "k1"),
                g32.Cell("Cricket_00", "shared", cls_a, "k2"),
                g32.Cell("Cricket_00", "shared", cls_a, "k3"),
                g32.Cell("Cricket_00", "shared", cls_a, "k4")]

    def _index(self, rows):
        return {(r["task_id"], r["cls_key"], r["iters"]): r for r in rows}

    def test_done_skipped_partial_needs_flag(self, cls_a):
        idx = self._index(
            [cache_row("Cricket_00", "k1", b, 0.5) for b in B2]        # done
            + [cache_row("Cricket_00", "k2", 60, 0.5)]                 # partial
            + [cache_row("Cricket_00", "k3", 60, None, error="x")])    # terminal
        cells = self._cells(cls_a)
        todo = g32.pending_cells(cells, idx, B2, resume=True, redo_partial=False)
        assert [c.cls_key for c in todo] == ["k4"]
        todo2 = g32.pending_cells(cells, idx, B2, resume=True, redo_partial=True)
        assert [c.cls_key for c in todo2] == ["k2", "k4"]

    def test_terminal_never_retried_even_with_redo_partial(self, cls_a):
        idx = self._index([cache_row("Cricket_00", "k1", 290, None, error="x",
                                     iters_list=(60, 120, 290))])
        cells = [g32.Cell("Cricket_00", "shared", cls_a, "k1")]
        for redo in (False, True):
            assert g32.pending_cells(cells, idx, (60, 120, 290), resume=True,
                                     redo_partial=redo) == []

    def test_terminal_retried_only_with_explicit_flag(self, cls_a):
        idx = self._index([cache_row("Cricket_00", "k1", 290, None, error="x",
                                     iters_list=(60, 120, 290))])
        cells = [g32.Cell("Cricket_00", "shared", cls_a, "k1")]
        assert len(g32.pending_cells(cells, idx, (60, 120, 290), resume=True,
                                     redo_partial=False,
                                     retry_terminal=True)) == 1

    def test_without_resume_everything_runs(self, cls_a):
        idx = self._index([cache_row("Cricket_00", "k1", b, 0.5) for b in B2])
        cells = [g32.Cell("Cricket_00", "shared", cls_a, "k1")]
        assert len(g32.pending_cells(cells, idx, B2, resume=False,
                                     redo_partial=False)) == 1


class TestNoiseResume:
    """噪声探针的 rep 必须分表判定——同格两个 rep 是两条独立轨迹。"""

    def test_reps_do_not_shadow_each_other(self, cls_a):
        rows = [cache_row("Cricket_00", "k1", b, 0.5, seed=2) for b in B2]
        nidx = g32.index_by_seed(rows)
        cells = [g32.NoiseCell("Cricket_00", "shared", cls_a, "k1", r)
                 for r in (1, 2)]
        todo = g32.pending_cells(cells, nidx, B2, resume=True, redo_partial=False)
        assert [c.rep for c in todo] == [1]     # rep2 已完成，rep1 仍要跑

    def test_partial_rep_needs_redo_flag(self, cls_a):
        nidx = g32.index_by_seed([cache_row("Cricket_00", "k1", 60, 0.5, seed=1)])
        cells = [g32.NoiseCell("Cricket_00", "shared", cls_a, "k1", 1)]
        assert g32.pending_cells(cells, nidx, B2, resume=True,
                                 redo_partial=False) == []
        assert len(g32.pending_cells(cells, nidx, B2, resume=True,
                                     redo_partial=True)) == 1


class TestPrunePartial:
    def test_removes_only_the_named_cell_and_backs_up(self, tmp_path, cls_a):
        path = tmp_path / "rprime_cache.jsonl"
        write_rows(path, [cache_row("Cricket_00", "k1", 60, 0.5),
                          cache_row("Cricket_00", "k2", 60, 0.5),
                          cache_row("Cricket_00", "k2", 120, 0.5)])
        n = g32.prune_partial(path, [g32.Cell("Cricket_00", "shared", cls_a, "k1")],
                              outdir=tmp_path)
        assert n == 1
        left = [json.loads(x) for x in
                path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert {r["cls_key"] for r in left} == {"k2"}
        assert list(tmp_path.glob("rprime_cache.jsonl.bak-*"))

    def test_rep_aware(self, tmp_path, cls_a):
        path = tmp_path / "noise_probe.jsonl"
        write_rows(path, [cache_row("Cricket_00", "k1", 60, 0.5, seed=1),
                          cache_row("Cricket_00", "k1", 60, 0.5, seed=2),
                          cache_row("Cricket_00", "k1", 120, 0.5, seed=2)])
        n = g32.prune_partial(
            path, [g32.NoiseCell("Cricket_00", "shared", cls_a, "k1", 1)],
            outdir=tmp_path)
        assert n == 1
        left = [json.loads(x) for x in
                path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert {r["seed"] for r in left} == {2}


class TestInflightInterlock:
    """CRITICAL 1：并发档的打点文件叫 pq_inflight_<ds>.txt，只认 `inflight*.txt`
    前缀会让 verify 的空闸与 prune 的互锁对它们全部失明。"""

    def test_glob_catches_prefixed_names(self, tmp_path):
        (tmp_path / "pq_inflight_Handwriting.txt").write_text("a", encoding="utf-8")
        (tmp_path / "inflight.txt").write_text("", encoding="utf-8")
        names = {p.name for p in g32.inflight_files(tmp_path)}
        assert names == {"pq_inflight_Handwriting.txt", "inflight.txt"}
        assert [p.name for p in g32.busy_inflight(tmp_path)] == \
            ["pq_inflight_Handwriting.txt"]

    def test_foreign_named_inflight_blocks_prune(self, tmp_path, cls_a):
        path = tmp_path / "rprime_cache.jsonl"
        write_rows(path, [cache_row("Cricket_00", "k1", 60, 0.5)])
        (tmp_path / "pq_inflight_Cricket.txt").write_text("Cricket_01\tk9\t290",
                                                          encoding="utf-8")
        with pytest.raises(g32.InflightBusy) as e:
            g32.prune_partial(path, [g32.Cell("Cricket_00", "shared", cls_a, "k1")],
                              outdir=tmp_path)
        assert "pq_inflight_Cricket.txt" in str(e.value)
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1   # 未改动

    def test_own_inflight_also_blocks(self, tmp_path, cls_a):
        """不再排除"自己那份"：文件层面分不清残迹与在跑，猜错就吞别人的行。"""
        path = tmp_path / "rprime_cache.jsonl"
        write_rows(path, [cache_row("Cricket_00", "k1", 60, 0.5)])
        mine = tmp_path / "inflight.txt"
        mine.write_text("Cricket_00\tk1\t290", encoding="utf-8")
        with pytest.raises(g32.InflightBusy):
            g32.prune_partial(path, [g32.Cell("Cricket_00", "shared", cls_a, "k1")],
                              outdir=tmp_path, extra=mine)
        mine.write_text("", encoding="utf-8")          # 操作方确认后清空
        assert g32.prune_partial(
            path, [g32.Cell("Cricket_00", "shared", cls_a, "k1")],
            outdir=tmp_path, extra=mine) == 1


# --------------------------------------------------------------- plan
class TestPlan:
    def test_full_grid_shape(self):
        plan = g32.build_plan(pq.PQ_DATASETS)
        main = [c for c in plan if c.origin in (pq.ORIGIN_SHARED, pq.ORIGIN_RANDOM)]
        ggs = [c for c in plan if c.origin == g32.ORIGIN_GGS]
        assert len(main) == 45 * 20 and len(ggs) == 45
        assert len({c.slice_id for c in plan}) == 45

    def test_noise_plan_shape(self):
        plan = g32.build_plan(pq.PQ_DATASETS)
        cells = g32.noise_plan(plan, n_slices=2, reps=(1, 2))
        assert len(cells) == 8 * 2 * 10 * 2       # 8 数据集 × 2 切片 × 10 shared × 2 rep
        assert {c.origin for c in cells} == {pq.ORIGIN_SHARED}

    def test_no_key_collision_in_full_grid(self):
        plan = g32.build_plan(pq.PQ_DATASETS)
        keys = [(c.slice_id, c.cls_key) for c in plan]
        assert len(keys) == len(set(keys))     # 撞键会让两个 origin 写同一条缓存键

    def test_limits_for_smoke(self):
        plan = g32.build_plan(("Handwriting", "UniMiB-SHAR"), n_slices=1, n_cls=1)
        assert [c.slice_id for c in plan] == ["Handwriting_00", "Handwriting_00",
                                              "UniMiB-SHAR_00", "UniMiB-SHAR_00"]
        assert [c.origin for c in plan] == ["shared", "ggs", "shared", "ggs"]


# --------------------------------------------------------------- verify
class TestVerify:
    def _plan(self, cls_a):
        return [g32.Cell("Cricket_00", "shared", cls_a, "k1"),
                g32.Cell("Cricket_00", g32.ORIGIN_GGS, cls_a, "g1")]

    def _good_rows(self):
        return ([cache_row("Cricket_00", "k1", b, 0.5) for b in B2]
                + [cache_row("Cricket_00", "g1", b, 0.6, origin="ggs") for b in B2])

    def test_clean_cache_passes(self, tmp_path, cls_a):
        write_rows(tmp_path / "rprime_cache.jsonl", self._good_rows())
        rep = g32.verify(tmp_path, self._plan(cls_a), [], B2)
        assert rep["ok"] and rep["hard_failures"] == []

    def test_missing_cells_fail(self, tmp_path, cls_a):
        write_rows(tmp_path / "rprime_cache.jsonl", self._good_rows()[:2])
        rep = g32.verify(tmp_path, self._plan(cls_a), [], B2)
        assert not rep["ok"] and rep["n_missing"] == 1

    def test_duplicate_keys_fail(self, tmp_path, cls_a):
        rows = self._good_rows()
        write_rows(tmp_path / "rprime_cache.jsonl", rows + rows[:1])
        rep = g32.verify(tmp_path, self._plan(cls_a), [], B2)
        assert not rep["ok"] and rep["n_duplicate"] == 1

    def test_partial_trajectory_fails(self, tmp_path, cls_a):
        write_rows(tmp_path / "rprime_cache.jsonl",
                   self._good_rows()[:1] + self._good_rows()[2:])
        rep = g32.verify(tmp_path, self._plan(cls_a), [], B2)
        assert not rep["ok"] and rep["n_partial"] == 1

    def test_null_without_error_fails(self, tmp_path, cls_a):
        rows = self._good_rows()
        rows[0] = cache_row("Cricket_00", "k1", 60, None)
        write_rows(tmp_path / "rprime_cache.jsonl", rows)
        rep = g32.verify(tmp_path, self._plan(cls_a), [], B2)
        assert not rep["ok"] and rep["n_null_no_error"] == 1

    def _terminal_rows(self):
        rows = [cache_row("Cricket_00", "k1", b, None, error="boom") for b in B2]
        return rows + [cache_row("Cricket_00", "g1", b, 0.6, origin="ggs")
                       for b in B2]

    def test_terminal_fails_by_default(self, tmp_path, cls_a):
        """毒轨迹 = 那几格永远没有 R'。当告警报会让 finish.sh 打印"通过"。"""
        write_rows(tmp_path / "rprime_cache.jsonl", self._terminal_rows())
        rep = g32.verify(tmp_path, self._plan(cls_a), [], B2)
        assert not rep["ok"] and rep["n_terminal"] == 1
        assert rep["terminal_ids"] == ["Cricket_00:k1"]

    def test_terminal_allowed_with_flag(self, tmp_path, cls_a):
        write_rows(tmp_path / "rprime_cache.jsonl", self._terminal_rows())
        rep = g32.verify(tmp_path, self._plan(cls_a), [], B2, allow_terminal=True)
        assert rep["ok"] and rep["n_terminal"] == 1

    def test_noise_terminal_also_gated(self, tmp_path, cls_a):
        """毒噪声格不算"缺行"，却让那格凑不齐重复对 → δ 站在更少样本上。"""
        write_rows(tmp_path / "rprime_cache.jsonl", self._good_rows())
        write_rows(tmp_path / "noise_probe.jsonl",
                   [cache_row("Cricket_00", "k1", b, None, seed=1, error="boom")
                    for b in B2])
        ncells = [g32.NoiseCell("Cricket_00", "shared", cls_a, "k1", 1)]
        rep = g32.verify(tmp_path, self._plan(cls_a), ncells, B2)
        assert not rep["ok"] and rep["n_terminal_noise"] == 1
        assert rep["terminal_ids"] == ["Cricket_00:k1:noise1"]
        assert g32.verify(tmp_path, self._plan(cls_a), ncells, B2,
                          allow_terminal=True)["ok"]

    def test_device_mix_warns_only(self, tmp_path, cls_a):
        rows = self._good_rows()
        rows[0] = cache_row("Cricket_00", "k1", 60, 0.5, device="dev-B")
        write_rows(tmp_path / "rprime_cache.jsonl", rows)
        rep = g32.verify(tmp_path, self._plan(cls_a), [], B2)
        assert rep["ok"] and len(rep["devices"]) == 2

    def test_nonempty_inflight_fails(self, tmp_path, cls_a):
        write_rows(tmp_path / "rprime_cache.jsonl", self._good_rows())
        (tmp_path / "inflight.txt").write_text("Cricket_00\tk1\t120", encoding="utf-8")
        rep = g32.verify(tmp_path, self._plan(cls_a), [], B2)
        assert not rep["ok"] and rep["inflight"]

    def test_noise_counts(self, tmp_path, cls_a):
        write_rows(tmp_path / "rprime_cache.jsonl", self._good_rows())
        write_rows(tmp_path / "noise_probe.jsonl",
                   [cache_row("Cricket_00", "k1", b, 0.5, seed=1) for b in B2])
        ncells = [g32.NoiseCell("Cricket_00", "shared", cls_a, "k1", 1)]
        rep = g32.verify(tmp_path, self._plan(cls_a), ncells, B2)
        assert rep["ok"] and rep["n_noise"] == 2


# --------------------------------------------------------------- δ 标定
def _p95_ref(vals):
    ds = sorted(abs(v) for v in vals)
    return ds[min(len(ds) - 1, int(round(0.95 * (len(ds) - 1))))]


class TestCalibrate:
    """合成网格：1 数据集 × 2 切片 × 3 CLS × (canonical + 2 rep) × 2 预算。"""

    KEYS = ("k1", "k2", "k3")
    SLICES = ("Cricket_00", "Cricket_01")

    def _values(self, slice_id, key, rep, b):
        # 可复现的确定性伪值：rep/预算/键各自贡献不同幅度，便于手算
        base = {"k1": 0.60, "k2": 0.50, "k3": 0.10}[key]
        off = {"Cricket_00": 0.0, "Cricket_01": 0.02}[slice_id]
        return base + off + 0.01 * rep + 0.003 * (b // 60)

    def _write(self, outdir, *, n_eval=100):
        main, noise = [], []
        for s in self.SLICES:
            for k in self.KEYS:
                for b in B2:
                    main.append(cache_row(s, k, b, self._values(s, k, 0, b),
                                          n_eval=n_eval))
                    for rep in (1, 2):
                        noise.append(cache_row(s, k, b, self._values(s, k, rep, b),
                                               seed=rep, n_eval=n_eval))
        write_rows(outdir / "rprime_cache.jsonl", main)
        write_rows(outdir / "noise_probe.jsonl", noise)

    def _hand_diffs(self, b, keys=KEYS):
        out = []
        for s in self.SLICES:
            for k in keys:
                vals = [self._values(s, k, r, b) for r in (0, 1, 2)]
                for i in range(len(vals)):
                    for j in range(i + 1, len(vals)):
                        out.append(abs(vals[i] - vals[j]))
        return out

    def test_delta_matches_hand_computed(self, tmp_path):
        self._write(tmp_path)
        res = g32.calibrate(tmp_path, budgets=B2, reps=(1, 2), bands_of=lambda d: {})
        for b in B2:
            want = pq.calibrate_delta(self._hand_diffs(b), quantum=1 / 100)
            rec = res["Cricket"][str(b)]
            assert rec["delta"] == pytest.approx(want)
            assert rec["n_diffs"] == 18            # 2 切片 × 3 键 × 3 对
            assert rec["p95"] == pytest.approx(_p95_ref(self._hand_diffs(b)))
            assert rec["quantum_floor"] == pytest.approx(2 / 100)

    def test_delta_nd_excludes_strict_cells(self, tmp_path):
        self._write(tmp_path)
        # k3 的 canonical R' ≈0.10/0.12，落在基线带 (0.5 ± 0.05) 之下 = strict 破坏
        bands = {s: (0.5, 0.05) for s in self.SLICES}
        res = g32.calibrate(tmp_path, budgets=B2, reps=(1, 2),
                            bands_of=lambda d: bands)
        for b in B2:
            want_nd = pq.calibrate_delta(self._hand_diffs(b, keys=("k1", "k2")),
                                         quantum=1 / 100)
            assert res["Cricket"][str(b)]["delta_nd"] == pytest.approx(want_nd)
            assert res["Cricket"][str(b)]["n_diffs_nd"] == 12

    def test_incomplete_cells_skipped(self, tmp_path):
        self._write(tmp_path)
        # 再补一个只有 canonical、没有 rep 的格：不该进 diffs（"三点齐全"判据）
        write_rows(tmp_path / "rprime_cache.jsonl",
                   [cache_row("Cricket_00", "k9", b, 0.9) for b in B2])
        res = g32.calibrate(tmp_path, budgets=B2, reps=(1, 2), bands_of=lambda d: {})
        assert res["Cricket"][str(B2[0])]["n_diffs"] == 18

    def test_writes_json_loadable_by_pq_common(self, tmp_path):
        self._write(tmp_path)
        g32.calibrate(tmp_path, budgets=B2, reps=(1, 2), bands_of=lambda d: {})
        table = pq.load_delta_calibration(tmp_path / "delta_calibration.json")
        assert set(table) == {("Cricket", 60), ("Cricket", 120)}
        assert all(v > 0 for v in table.values())

    def test_merges_into_existing_calibration(self, tmp_path):
        """IMPORTANT 2：子集跑不得把别的数据集从 delta_calibration.json 里截掉。"""
        (tmp_path / "delta_calibration.json").write_text(json.dumps(
            {"Handwriting": {"60": {"delta": 0.07}}}), encoding="utf-8")
        self._write(tmp_path)
        res = g32.calibrate(tmp_path, budgets=B2, reps=(1, 2), bands_of=lambda d: {})
        assert set(res) == {"Handwriting", "Cricket"}
        assert res["Handwriting"]["60"]["delta"] == 0.07
        on_disk = json.loads(
            (tmp_path / "delta_calibration.json").read_text(encoding="utf-8"))
        assert set(on_disk) == {"Handwriting", "Cricket"}

    def test_delta_nd_is_null_when_bands_unavailable(self, tmp_path):
        self._write(tmp_path)

        def boom(_d):
            raise ValueError("缺基线")

        res = g32.calibrate(tmp_path, budgets=B2, reps=(1, 2), bands_of=boom)
        for b in B2:
            assert res["Cricket"][str(b)]["delta_nd"] is None
            assert res["Cricket"][str(b)]["n_diffs_nd"] == 0

    def test_delta_nd_is_null_when_a_probe_slice_has_no_band(self, tmp_path):
        self._write(tmp_path)
        partial = {"Cricket_00": (0.5, 0.05)}          # 缺 Cricket_01
        res = g32.calibrate(tmp_path, budgets=B2, reps=(1, 2),
                            bands_of=lambda d: partial)
        assert res["Cricket"][str(B2[0])]["delta_nd"] is None

    def test_no_replicates_is_reported_not_crash(self, tmp_path):
        write_rows(tmp_path / "rprime_cache.jsonl",
                   [cache_row("Cricket_00", "k1", b, 0.5) for b in B2])
        write_rows(tmp_path / "noise_probe.jsonl", [])
        res = g32.calibrate(tmp_path, budgets=B2, reps=(1, 2), bands_of=lambda d: {})
        assert res == {}


# --------------------------------------------------------------- --apply-delta
class TestApplyDelta:
    TABLE = {"Handwriting": {60: 0.075, 120: 0.08, 290: 0.07},
             "UniMiB-SHAR": {60: 0.055, 120: 0.06, 290: 0.05}}

    def _labels_copy(self, tmp_path):
        src = (ROOT / "configs" / "labels.yaml")
        dst = tmp_path / "labels.yaml"
        dst.write_bytes(src.read_bytes())
        return dst

    def test_only_the_block_changes(self, tmp_path):
        path = self._labels_copy(tmp_path)
        before = path.read_text(encoding="utf-8").splitlines()
        g32.apply_delta(path, self.TABLE)
        after = path.read_text(encoding="utf-8").splitlines()
        key = "  delta_per_dataset_budget:"
        i_before = before.index(key + " {}")
        i_after = after.index(key)
        assert before[:i_before] == after[:i_after]
        assert before[i_before + 1:] == after[i_after + 1 + 8:]

    def test_yaml_parses_to_table(self, tmp_path):
        path = self._labels_copy(tmp_path)
        g32.apply_delta(path, self.TABLE)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["pq"]["delta_per_dataset_budget"] == self.TABLE
        # 其余键零漂移
        src = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(
            encoding="utf-8"))
        src["pq"]["delta_per_dataset_budget"] = self.TABLE
        assert cfg == src

    def test_idempotent(self, tmp_path):
        path = self._labels_copy(tmp_path)
        g32.apply_delta(path, self.TABLE)
        once = path.read_bytes()
        g32.apply_delta(path, self.TABLE)
        assert path.read_bytes() == once

    def test_rewrite_replaces_previous_block(self, tmp_path):
        path = self._labels_copy(tmp_path)
        g32.apply_delta(path, self.TABLE)
        g32.apply_delta(path, {"Cricket": {60: 0.09}})
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["pq"]["delta_per_dataset_budget"] == {"Cricket": {60: 0.09}}

    def test_keeps_line_endings(self, tmp_path):
        path = self._labels_copy(tmp_path)
        raw_before = path.read_bytes()
        g32.apply_delta(path, self.TABLE)
        raw_after = path.read_bytes()
        assert (b"\r\n" in raw_before) == (b"\r\n" in raw_after)


# --------------------------------------------------------------- 回填覆盖度闸
class TestDeltaCoverageGate:
    """IMPORTANT 2：labels.yaml 的 δ 表是 33 号的唯一真源，半张表不许回填。"""

    def _full(self):
        return {d: {b: 0.05 for b in pq.BUDGETS} for d in pq.PQ_DATASETS}

    def test_full_table_has_no_gaps(self):
        assert g32.delta_coverage_gaps(self._full()) == []

    def test_missing_dataset_is_a_gap(self):
        t = self._full()
        del t["Cricket"]
        assert any("Cricket" in g for g in g32.delta_coverage_gaps(t))

    def test_missing_budget_is_a_gap(self):
        t = self._full()
        del t["Handwriting"][290]
        assert g32.delta_coverage_gaps(t) == ["Handwriting@[290]"]

    def test_subset_run_is_refused(self):
        assert len(g32.delta_coverage_gaps({"Cricket": {60: 0.05, 120: 0.05,
                                                        290: 0.05}})) == 7


# --------------------------------------------------------------- 原子落盘
class TestAtomicJson:
    """IMPORTANT 3：8 个进程共读共写，落盘窗口里绝不能出现空/半截文件。"""

    def test_uses_tmp_plus_replace_and_leaves_nothing(self, tmp_path, monkeypatch):
        calls = []
        real = os.replace
        monkeypatch.setattr(g32.os, "replace",
                            lambda a, b: (calls.append((str(a), str(b))), real(a, b))[1])
        path = tmp_path / "tasks.json"
        g32.write_json_atomic(path, {"tasks": {"A": 1}})
        assert len(calls) == 1 and calls[0][0].endswith(".tmp")
        assert json.loads(path.read_text(encoding="utf-8")) == {"tasks": {"A": 1}}
        assert list(tmp_path.glob("*.tmp")) == []

    def test_torn_json_raises_instead_of_none(self, tmp_path):
        path = tmp_path / "tasks.json"
        path.write_text('{"tasks": {"A": 1', encoding="utf-8")     # 被杀留下的半行
        with pytest.raises(ValueError):
            g32.read_json(path)

    def test_missing_file_is_none(self, tmp_path):
        assert g32.read_json(tmp_path / "nope.json") is None

    def test_merge_refuses_empty_payload(self):
        with pytest.raises(ValueError):
            g32.merge_payload({"tasks": {"A": 1}}, {})


# --------------------------------------------------------------- 产物合并
class TestMergePayload:
    """pod 上 8 个进程共用一份 tasks.json：子集跑不得把别的数据集抹掉。"""

    def test_tasks_union(self):
        prev = {"pool": ["Cricket"], "tasks": {"Cricket_00": {"a": 1}},
                "eval_ratio": 0.3}
        new = {"pool": ["Handwriting"], "tasks": {"Handwriting_00": {"b": 2}},
               "eval_ratio": 0.3}
        out = g32.merge_payload(prev, new)
        assert out["pool"] == ["Cricket", "Handwriting"]
        assert set(out["tasks"]) == {"Cricket_00", "Handwriting_00"}

    def test_new_wins_on_same_key(self):
        out = g32.merge_payload({"tasks": {"A": 1}}, {"tasks": {"A": 2}})
        assert out["tasks"] == {"A": 2}

    def test_catalog_random_union(self):
        prev = {"shared": [{"cls_key": "s"}], "random": {"Cricket_00": [1]},
                "ggs": {"cls_key": "g"}}
        new = {"shared": [{"cls_key": "s"}], "random": {"Handwriting_00": [2]},
               "ggs": {"cls_key": "g"}}
        out = g32.merge_payload(prev, new)
        assert set(out["random"]) == {"Cricket_00", "Handwriting_00"}

    def test_no_previous_is_identity(self):
        assert g32.merge_payload(None, {"tasks": {"A": 1}}) == {"tasks": {"A": 1}}


# --------------------------------------------------------------- stage1 对账
class TestStage1Consistency:
    def test_missing_reference_is_silent(self, tmp_path):
        res = {"Cricket": {"290": {"p95": 0.05}}}
        assert g32.stage1_consistency(res, path=tmp_path / "nope.jsonl") == []

    def test_ratio_against_real_k3_cache(self):
        if not g32.STAGE1_NOISE.exists():
            pytest.skip("阶段1 k3 缓存未落盘")
        out = g32.stage1_consistency({"Handwriting": {"290": {"p95": 0.05}}})
        assert [r[0] for r in out] == ["Handwriting"]
        _ds, ours, ref, ratio, flagged = out[0]
        assert ours == 0.05 and ref > 0
        assert ratio == pytest.approx(0.05 / ref)
        assert flagged == (not 0.7 <= ratio <= 1.4)

    def test_dataset_absent_from_reference_skipped(self):
        if not g32.STAGE1_NOISE.exists():
            pytest.skip("阶段1 k3 缓存未落盘")
        assert g32.stage1_consistency({"UniMiB-SHAR": {"290": {"p95": 0.05}}}) == []


# --------------------------------------------------------------- 定义漂移闸
class TestDefinitionDrift:
    def test_missing_reference_only_warns(self, tmp_path, capsys):
        bad = g32.check_definition_drift({"Cricket_00": (0, 1)}, "Cricket",
                                         meta_dir=tmp_path, workpoint=tmp_path / "x")
        assert bad == []
        assert "无参照" in capsys.readouterr().out

    def test_mismatch_reported(self, tmp_path):
        meta = tmp_path / "Cricket.json"
        meta.write_text(json.dumps({"tasks": [
            {"task_id": "Cricket_00", "classes": [0, 1]},
            {"task_id": "Cricket_01", "classes": [2, 3]}]}), encoding="utf-8")
        bad = g32.check_definition_drift(
            {"Cricket_00": (0, 1), "Cricket_01": (9, 9)}, "Cricket",
            meta_dir=tmp_path, workpoint=tmp_path / "x")
        assert [b[0] for b in bad] == ["Cricket_01"]

    def test_real_handwriting_matches(self):
        """真值对账：注册表切出的 HW 00/01 classes 必须与 stage1 落盘一致。"""
        meta = ROOT / ("results/stage1_rerun_20260804/results/step0_snr_gate/"
                       "tasks_meta.d")
        if not (meta / "Handwriting.json").exists():
            pytest.skip("stage1 参照未落盘")
        ref = json.loads((meta / "Handwriting.json").read_text(encoding="utf-8"))
        got = {t["task_id"]: tuple(t["classes"]) for t in ref["tasks"]}
        assert g32.check_definition_drift(got, "Handwriting", meta_dir=meta,
                                          workpoint=ROOT / "nope") == []


# --------------------------------------------------------------- smoke
_HAS_DATA = (importlib.util.find_spec("aeon") is not None
             and (ROOT / "data" / "uea_cache").exists())


def _fake_evaluator(task, cls_cfg, X, y, *, weight_seed, split_seed, checkpoints,
                    eval_ratio=0.3, batch_size=128, lr=1e-3, device=None):
    """确定性假评估器：R' 随 weight_seed/预算变化，但不碰 GPU。"""
    rng = np.random.default_rng(weight_seed % (2 ** 31))
    base = 0.4 + 0.3 * rng.random()
    return {int(b): {"r_prime": float(base + 0.01 * i + 0.02 * rng.random()),
                     "acc": float(base), "f1": float(base), "svm_c": 1.0,
                     "n_train": 200, "n_eval": 100, "weight_seed": int(weight_seed),
                     "split_seed": int(split_seed), "device_name": "fake",
                     "swa": False, "seconds": 0.1 * (i + 1), "error": None,
                     "trajectory_seconds": 0.3,
                     "checkpoint_iters": [int(c) for c in checkpoints]}
            for i, b in enumerate(checkpoints)}


@pytest.mark.skipif(not _HAS_DATA, reason="需要 aeon + data/uea_cache")
def test_smoke_end_to_end(tmp_path, monkeypatch):
    if not os.environ.get("PQ_SMOKE_REAL"):
        monkeypatch.setattr(g32, "evaluate_rprime_trajectory", _fake_evaluator)
    rc = g32.main(["--smoke", "--outdir", str(tmp_path)])
    assert rc == 0
    for name in ("tasks.json", "cls_catalog.json", "rprime_cache.jsonl",
                 "noise_probe.jsonl", "delta_calibration.json"):
        assert (tmp_path / name).exists(), name
    rows = [json.loads(x) for x in
            (tmp_path / "rprime_cache.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    assert len(rows) == 2 * 2 * 3          # 2 数据集 × 2 CLS × 3 截点
    assert {r["iters"] for r in rows} == {4, 8, 12}
    assert {r["origin"] for r in rows} == {"shared", "ggs"}
    assert all(math.isfinite(r["r_prime"]) for r in rows)
    cal = json.loads((tmp_path / "delta_calibration.json").read_text(encoding="utf-8"))
    assert set(cal) == {"Handwriting", "UniMiB-SHAR"}
    # 重跑（--resume 幂等）不该新增任何行
    rc2 = g32.main(["--smoke", "--outdir", str(tmp_path)])
    assert rc2 == 0
    rows2 = (tmp_path / "rprime_cache.jsonl").read_text(encoding="utf-8").splitlines()
    assert len([x for x in rows2 if x.strip()]) == len(rows)


@pytest.mark.skipif(not _HAS_DATA, reason="需要 aeon + data/uea_cache")
def test_smoke_partial_recovery(tmp_path, monkeypatch):
    """半条轨迹：--resume 只报不跑（verify 判红），--redo-partial 才剔旧行重跑。"""
    monkeypatch.setattr(g32, "evaluate_rprime_trajectory", _fake_evaluator)
    assert g32.main(["--smoke", "--outdir", str(tmp_path)]) == 0
    cache = tmp_path / "rprime_cache.jsonl"
    rows = [json.loads(x) for x in
            cache.read_text(encoding="utf-8").splitlines() if x.strip()]
    victim = rows[0]
    cache.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows
                             if not (r["task_id"] == victim["task_id"]
                                     and r["cls_key"] == victim["cls_key"]
                                     and r["iters"] == 12)), encoding="utf-8")
    assert g32.main(["--smoke", "--outdir", str(tmp_path)]) == 1      # verify 判红
    left = [x for x in cache.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(left) == 11                                            # 未重跑
    assert g32.main(["--smoke", "--outdir", str(tmp_path), "--redo-partial"]) == 0
    back = [json.loads(x) for x in
            cache.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(back) == 12
    keys = [(r["task_id"], r["cls_key"], r["iters"]) for r in back]
    assert len(keys) == len(set(keys))                                # 零重复键
    assert list(tmp_path.glob("rprime_cache.jsonl.bak-*"))            # 备份留痕


@pytest.mark.skipif(not _HAS_DATA, reason="需要 aeon + data/uea_cache")
def test_smoke_terminal_gate_and_retry(tmp_path, monkeypatch):
    """毒轨迹：--verify 默认判红；--allow-terminal 放行；--retry-terminal 修好。"""
    monkeypatch.setattr(g32, "evaluate_rprime_trajectory", _fake_evaluator)
    assert g32.main(["--smoke", "--outdir", str(tmp_path)]) == 0
    cache = tmp_path / "rprime_cache.jsonl"
    rows = [json.loads(x) for x in
            cache.read_text(encoding="utf-8").splitlines() if x.strip()]
    poisoned = rows[0]["task_id"], rows[0]["cls_key"]
    for r in rows:                                  # 整条轨迹改成 error 行
        if (r["task_id"], r["cls_key"]) == poisoned:
            r["r_prime"], r["error"] = None, "RuntimeError: boom"
    cache.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                     encoding="utf-8")
    assert g32.main(["--smoke", "--outdir", str(tmp_path)]) == 1        # 默认判红
    assert g32.main(["--smoke", "--outdir", str(tmp_path),
                     "--allow-terminal"]) == 0                          # 显式放行
    assert g32.main(["--smoke", "--outdir", str(tmp_path),
                     "--retry-terminal"]) == 0                          # 剔旧行重跑
    back = [json.loads(x) for x in
            cache.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(back) == 12
    assert all(r["r_prime"] is not None for r in back)
    keys = [(r["task_id"], r["cls_key"], r["iters"]) for r in back]
    assert len(keys) == len(set(keys))
    assert list(tmp_path.glob("rprime_cache.jsonl.bak-*"))


@pytest.mark.skipif(not _HAS_DATA, reason="需要 aeon + data/uea_cache")
def test_smoke_refuses_real_grid(tmp_path):
    with pytest.raises(SystemExit) as e:
        g32.main(["--smoke", "--outdir", str(pq.OUT)])
    assert e.value.code == 2
