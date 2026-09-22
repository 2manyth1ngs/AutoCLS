"""WP17-ACTS S2-CORPUS（results/acts/s2/PROTOCOL.md）——先于实现。覆盖：profile s2（源分组 + 每组任务数）、按源切片数、
plan_S（shared × 2 种子 + random 复用 / 新采、已有格不重跑）、缓存种子合并去重、G-P / G-R 判定、读数码、命令行入口。零 GPU、不读真实数据。"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

POS = ["Handwriting", "UniMiB-SHAR", "PhonemeSpectra", "UWaveGestureLibrary", "Cricket", "PTB-XL", "SleepEDF", "PAMAP2",
       "NonInvasiveFetalECGThorax1", "ShapesAll", "LargeKitchenAppliances", "Adiac", "FordA"]
REV = ["SemgHandMovementCh2", "EthanolConcentration", "CinCECGTorso", "SelfRegulationSCP1"]
PROB = ["HandMovementDirection", "SelfRegulationSCP2"]


def _ac():
    return importlib.import_module("scripts.acts_common")


def _g83():
    return importlib.import_module("scripts.83_acts_corpus")


def _fake_cls(i: int):
    from autocls.space.cls_space import CLS
    return CLS(resize=0.0, rescale=0.0, jitter=0.0, point_mask=0.0, freq_mask=0.0, crop=0.5, order=0, emb_jitter=0.0, emb_mask=0.0,
               norm="none", instance=True, temporal=False, cross_scale=False, kernel=None, pool=None, adj_neighbor=None,
               loss_type="info_nce", sim_func="dot", temperature=[0.01, 0.1, 1.0][i % 3])


@pytest.fixture(scope="module")
def cfg():
    return _ac().load_acts_config(profile="s2")


@pytest.fixture(scope="module")
def cfg_pp():
    return _ac().load_acts_config(profile="s1pp")


class TestProfile:
    def test_groups_and_counts(self, cfg, cfg_pp):
        assert cfg["profile"] == "s2" and cfg["slicing"] == cfg_pp["slicing"] and cfg["prune"] == cfg_pp["prune"] and cfg["eval"] == cfg_pp["eval"]
        s = cfg["sources"]
        assert s["positive"] == POS and s["reverse"] == REV and s["probation"] == PROB
        assert s["n_tasks"] == {"positive": 8, "reverse": 16, "probation": 8}
        assert cfg["candidates"]["n_shared"] == 30 and cfg["candidates"]["n_random"] == 20 and cfg["candidates"]["random_reuse_from"] == "s1pp"
        assert cfg["candidates"]["shared_seed"] == cfg_pp["candidates"]["shared_seed"]
        assert cfg["labels"]["shared_seeds"] == [0, 1] and cfg["labels"]["random_seeds"] == [0] and cfg["labels"]["probe0_seeds"] == 3
        assert cfg["seed_caches"] == ["s1pp", "ucradm"]
        assert cfg["probation"] == {"rho_min": 0.5, "min_pass": 5, "guard_min_pass": 5, "headroom_min": 0.10}
        assert cfg["reverse_gate"] == {"rho_min": 0.5, "min_pass": 10, "guard_min_pass": 10, "headroom_min": 0.10}
        assert cfg["gates"]["G-A2p"] == {"real_interaction_share_min": 0.50}
        assert not cfg.get("whole_eval") and cfg["pilot"]["datasets"] == [] and cfg["smoke"]["dataset"] == "Adiac"

    def test_source_task_counts(self, cfg):
        g = _g83()
        counts = g.source_task_counts(cfg)
        assert len(counts) == 19 and sum(counts.values()) == 184
        assert counts["Handwriting"] == 8 and counts["EthanolConcentration"] == 16 and counts["SelfRegulationSCP2"] == 8
        assert list(counts) == POS + REV + PROB          # 顺序 = 正向 → 反向 → 试用（切片 / 计划 / 探针顺序）


class TestPlanS:
    def _meta(self, counts):
        meta = {}
        for s, n in counts.items():
            for r in range(n):
                meta[f"{s}_a{r:02d}"] = {"dataset": s, "n_train": 200, "T": 100, "C": 1, "k": 5, "majority": 0.3, "iters": 200,
                                         "tag": f"c0w{r % 4}n{r % 4}", "time_window": [0, 100], "fold_md5": "x"}
        return meta

    def test_build_plan_s_counts_and_reuse(self, cfg):
        g = _g83()
        counts = {"Handwriting": 8, "EthanolConcentration": 16, "PhonemeSpectra": 8}
        meta = self._meta(counts)
        shared = [(_fake_cls(i), f"key{i:02d}") for i in range(30)]
        reuse = {"PhonemeSpectra_a00": [(_fake_cls(40), "rk40"), (_fake_cls(41), "rk41")]}          # 试点复用（已剪后 2 条）
        rnd = {tid: [(_fake_cls(50 + j), f"rn{tid}{j}") for j in range(20)] for tid in meta if tid not in reuse}
        plan = g.build_plan_s2(cfg, meta, shared=shared, randoms={**reuse, **rnd}, existing_keys=set())
        cells = plan["cells"]
        n_sh = 32 * 30 * 2
        n_rd = 2 + 31 * 20
        assert plan["phase"] == "S" and plan["n_cells"] == n_sh + n_rd and len({c["key"] for c in cells}) == plan["n_cells"]
        assert {c["arm"] for c in cells} == {"shared", "random"} and all(c["eval"] == "cv5" for c in cells)
        assert plan["probe_tasks"] == list(meta) and plan["n_probe"] == 32 * 3
        assert sorted({c["seed"] for c in cells if c["arm"] == "random"}) == [0]
        done = {c["key"] for c in cells[:500]}
        p2 = g.build_plan_s2(cfg, meta, shared=shared, randoms={**reuse, **rnd}, existing_keys=done)
        assert p2["n_existing"] == 500 and p2["n_new"] == plan["n_cells"] - 500 and p2["n_cells"] == plan["n_cells"]

    def test_merge_seed_caches_dedups_by_key(self, tmp_path):
        g = _g83()
        a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"; out = tmp_path / "runs.jsonl"
        a.write_text('{"key": "k1", "acc": 0.5}\n{"key": "k2", "acc": 0.6}\n', encoding="utf-8")
        b.write_text('{"key": "k2", "acc": 0.6}\n{"key": "k3", "acc": 0.7}\n{"key": "k1", "acc": 0.5}\n', encoding="utf-8")
        n = g.merge_seed_caches([a, b], out)
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert n == 3 and [r["key"] for r in rows] == ["k1", "k2", "k3"]
        assert g.merge_seed_caches([a, b], out) == 3                       # 幂等：已有的不再追加

    def test_merge_probe_caches_dedups_by_task_seed(self, tmp_path):
        g = _g83()
        a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"; out = tmp_path / "probe0.jsonl"
        a.write_text('{"task_id": "T1", "seed": 0, "acc": 0.5}\n', encoding="utf-8")
        b.write_text('{"task_id": "T1", "seed": 0, "acc": 0.5}\n{"task_id": "T1", "seed": 1, "acc": 0.4}\n', encoding="utf-8")
        assert g.merge_seed_caches([a, b], out, key=lambda r: (r["task_id"], int(r["seed"]))) == 2


class TestGates:
    def _tasks(self, n, good, seed=0):
        rng = np.random.default_rng(seed); out = {}
        for t in range(n):
            base = np.sort(rng.uniform(0.3, 0.9, size=30))[::-1]
            a0 = base + rng.normal(scale=0.005, size=30)
            a1 = (base + rng.normal(scale=0.005, size=30)) if good else rng.uniform(0.3, 0.9, size=30)
            out[f"S_a{t:02d}"] = (a0.tolist(), a1.tolist())
        return out

    def test_probation_and_reverse_gates(self, cfg):
        g = _g83()
        ok = self._tasks(8, True); bad = self._tasks(8, False)
        maj = {t: 0.2 for t in ok}
        assert g.gate_rows(ok, maj, cfg["probation"])["ok"]
        assert not g.gate_rows(bad, maj, cfg["probation"])["ok"]
        assert not g.gate_rows(ok, {t: 0.85 for t in ok}, cfg["probation"])["ok"]                 # 护栏
        r16 = g.gate_rows(self._tasks(16, True), {f"S_a{t:02d}": 0.2 for t in range(16)}, cfg["reverse_gate"])
        assert r16["ok"] and r16["pass"] == 16 and r16["n"] == 16 and r16["guard_pass"] == 16

    def test_verdict(self, cfg):
        g = _g83()
        assert g.verdict_s2(True, 2) == "S2-CORPUS-GO+PROB2" and g.verdict_s2(True, 0) == "S2-CORPUS-GO+PROB0"
        assert g.verdict_s2(False, 1) == "S2-CORPUS-NOISE+PROB1"

    def test_parser(self):
        g = _g83()
        a = g.build_parser().parse_args(["--plan-s2", "--analyze-s2", "--phase", "S", "--profile", "s2"])
        assert a.plan_s2 and a.analyze_s2 and a.phase == "S"
