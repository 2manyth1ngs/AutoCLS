"""MX-3-PQ 目录排名 + regret + 进化搜索（scripts/34_pq_search.py）测试。

协议：results/mx3_pq/PROTOCOL.md §1 Q6 / §5 判读表（D1/D2/D3 门），
接口：.superpowers/sdd/autocts-e-autocts-autocts-e-autocts-refactored-moler/
interfaces.md「Task 6 → Task 7 saved-model format」「Task 7（脚本 34）」。

覆盖面：
- 臂/regret 定义（oracle、tclsc1/3、clsb1/3、clsonly1、ggs、random1/3、bottom），
  random 臂与解析期望 expected_max_of_k 逐位一致；
- 排名：已知传递性玩具模型的 round-robin 排序 = 效用序；ctx 逐对传进模型；
- 进化驱动：打桩 evolutionary_search 后 evo_picks.json 的实例 × 臂计数；
  --evaluate-evo 的 resume 按缓存键跳过（假评估器，零 GPU）；
- --analyze：D1/D2/D3 门在手工构造输入上的判定 + 分层表形状 + 成本表；
- --smoke：合成 fixture 端到端产出 rankings.json 与 search_analysis.json。

全部不碰 GPU、不加载真实数据集（评估器与数据装载一律 monkeypatch）。
"""
from __future__ import annotations

import dataclasses
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import pq_common as pq

g34 = importlib.import_module("scripts.34_pq_search")
_mx2 = importlib.import_module("scripts.25_mx2_endtoend")

ROOT = Path(__file__).resolve().parents[1]

# smoke 口径与脚本常量对齐（漂了就是协议漂了）
SMOKE_TARGETS = ("Handwriting", "UniMiB-SHAR")
SMOKE_BUDGETS = (4, 8, 12)
assert g34.SMOKE_TARGETS == SMOKE_TARGETS and g34.SMOKE_BUDGETS == SMOKE_BUDGETS


# ---------------------------------------------------------------- 臂 / regret
class TestInstanceArms:
    TRUTH = {"a": 0.9, "b": 0.8, "c": 0.5, "d": 0.2}
    RANKED = {
        "main": ["b", "c", "a", "d"],
        "cls_b": ["c", "a", "b", "d"],
        "cls_only": ["d", "a", "b", "c"],
    }

    def arms(self):
        return g34.instance_arms(self.TRUTH, self.RANKED, ggs_r=0.7, k_top=3)

    def test_oracle_and_bottom(self):
        a = self.arms()
        assert a["oracle"] == pytest.approx(0.9)
        assert a["bottom"] == pytest.approx(0.2)

    def test_main_arm_at_1_and_3(self):
        a = self.arms()
        assert a["tclsc1"] == pytest.approx(0.8)        # ranked[0] = b
        assert a["tclsc3"] == pytest.approx(0.9)        # max(b, c, a)

    def test_clsb_and_clsonly_arms_present(self):
        a = self.arms()
        assert a["clsb1"] == pytest.approx(0.5)         # ranked[0] = c
        assert a["clsb3"] == pytest.approx(0.9)         # max(c, a, b)
        assert a["clsonly1"] == pytest.approx(0.2)      # ranked[0] = d
        assert "clsonly3" not in a                      # 协议只报 @1

    def test_ggs_passthrough(self):
        assert self.arms()["ggs"] == pytest.approx(0.7)

    def test_random_arms_match_expected_max_of_k(self):
        a = self.arms()
        vals = list(self.TRUTH.values())
        assert a["random1"] == pytest.approx(float(np.mean(vals)))
        assert a["random3"] == pytest.approx(_mx2.expected_max_of_k(vals, 3))

    def test_regret_is_oracle_minus_arm(self):
        a = self.arms()
        r = g34.regret_of(a)
        assert r["oracle"] == pytest.approx(0.0)
        assert r["tclsc1"] == pytest.approx(0.9 - 0.8)
        assert r["ggs"] == pytest.approx(0.9 - 0.7)
        assert r["bottom"] == pytest.approx(0.9 - 0.2)

    def test_missing_arm_is_omitted_not_faked(self):
        a = g34.instance_arms(self.TRUTH, {"main": self.RANKED["main"]}, 0.7)
        assert "tclsc1" in a and "clsb1" not in a and "clsonly1" not in a

    def test_main_arm_required(self):
        with pytest.raises((KeyError, ValueError, AssertionError)):
            g34.instance_arms(self.TRUTH, {"cls_b": self.RANKED["cls_b"]}, 0.7)


# ---------------------------------------------------------------- 排名
class _ToyComparator(torch.nn.Module):
    """P(a≻b) = sigmoid(<w, v_a − v_b>)：可传递、与效用序一一对应。

    同时记录每次前向收到的 ctx，用来断言 ctx 真的走到了模型（而不是被吞掉）。
    """

    uses_E = True

    def __init__(self, w: torch.Tensor, ctx_dim: int = 3):
        super().__init__()
        self.register_buffer("w", w)
        self.ctx_dim = ctx_dim
        self.seen_ctx: list[list[float]] = []
        self.seen_e: list[int] = []

    def forward(self, cls_a, cls_b, E, ctx=None):
        n = cls_a.shape[0]
        if self.ctx_dim:
            assert ctx is not None and tuple(ctx.shape) == (n, self.ctx_dim)
            self.seen_ctx.append([float(v) for v in ctx[0]])
        else:
            assert ctx is None
        assert E is not None and E.shape[0] == n
        self.seen_e.append(int(E.shape[0]))
        return torch.sigmoid((cls_a - cls_b) @ self.w)


class TestRanking:
    def setup_method(self):
        self.w = torch.tensor([1.0, 0.0, 0.0])
        # 效用 = 第 0 维：c(0.9) > a(0.5) > d(0.2) > b(0.1)
        self.vecs = torch.tensor([[0.5, 1.0, 0.0], [0.1, 0.0, 1.0],
                                  [0.9, 0.0, 0.0], [0.2, 1.0, 1.0]])
        self.keys = ["a", "b", "c", "d"]

    def test_rank_equals_utility_order(self):
        model = _ToyComparator(self.w)
        ranked, S_list = g34.rank_candidates(
            [model], [torch.zeros(4)], self.vecs, self.keys,
            ctx_row=torch.as_tensor(pq.ctx_vector(120)))
        assert ranked == ["c", "a", "d", "b"]
        assert len(S_list) == 1 and S_list[0].shape == (4, 4)

    def test_ctx_row_reaches_the_model(self):
        model = _ToyComparator(self.w)
        ctx = torch.as_tensor(pq.ctx_vector(290))
        g34.rank_candidates([model], [torch.zeros(4)], self.vecs, self.keys,
                            ctx_row=ctx)
        assert model.seen_ctx, "模型一次 ctx 都没收到"
        for row in model.seen_ctx:
            assert row == pytest.approx([0.0, 0.0, 1.0])

    def test_multi_seed_majority_vote(self):
        """两个同向种子 + 一个反向种子：多数票仍给出效用序。"""
        good = [_ToyComparator(self.w), _ToyComparator(self.w)]
        bad = _ToyComparator(-self.w)
        ranked, _ = g34.rank_candidates(
            good + [bad], [torch.zeros(4)] * 3, self.vecs, self.keys,
            ctx_row=torch.as_tensor(pq.ctx_vector(60)))
        assert ranked == ["c", "a", "d", "b"]

    def test_pair_prob_matrix_diagonal_is_nan(self):
        model = _ToyComparator(self.w)
        P = g34.pair_prob_matrix_ctx(model, self.vecs, torch.zeros(4),
                                     torch.as_tensor(pq.ctx_vector(60)))
        assert np.all(np.isnan(np.diag(P)))
        assert np.isfinite(P[0, 1])

    def test_model_task_inputs_respects_branch(self):
        e, ctx = torch.zeros(4), torch.zeros(3)
        cls_only = _ToyComparator(self.w, ctx_dim=0)
        cls_only.uses_E = False
        assert g34.model_task_inputs(cls_only, e, ctx) == (None, None)
        both = _ToyComparator(self.w, ctx_dim=3)
        E2, ctx2 = g34.model_task_inputs(both, e, ctx)
        assert E2 is e and ctx2 is ctx


# ---------------------------------------------------------------- 模型往返
class TestModelRoundTrip:
    def test_load_pq_model_rebuilds_from_meta(self, tmp_path):
        from autocls.comparator import TCLSC
        from autocls.task_embed import TaskSetEncoder

        se = TaskSetEncoder(dim_input=8, nz=8)
        model = TCLSC(cls_dim=5, set_encoder=se, embedding_dim=4, fz=4,
                      ctx_dim=3, task_branch="E+ctx")
        meta = {"arm": "main", "seed": 0, "fold": "Handwriting",
                "task_branch": "E+ctx", "ctx_dim": 3, "cls_dim": 5,
                "embedding_dim": 4, "fz": 4, "nz": 8, "set_encoder": "settf",
                "e_source": "fold", "settf_windows": 4, "settf_t_max": 8,
                "budgets": list(pq.BUDGETS), "ctx_encoding": "onehot",
                "dim_input": 8}
        path = tmp_path / "main_s0.pt"
        torch.save({"state_dict": model.state_dict(), "meta": meta}, path)

        loaded, meta2 = g34.load_pq_model(path)
        assert meta2["arm"] == "main" and loaded.ctx_dim == 3
        assert loaded.task_branch == "E+ctx"
        for k, v in model.state_dict().items():
            assert torch.allclose(loaded.state_dict()[k], v)

    def test_load_pq_model_cls_only_has_no_task_branch(self, tmp_path):
        from autocls.comparator import TCLSC

        model = TCLSC(cls_dim=5, set_encoder=None, embedding_dim=4, fz=4,
                      ctx_dim=0, task_branch="none")
        meta = {"arm": "cls_only", "seed": 0, "fold": "Handwriting",
                "task_branch": "none", "ctx_dim": 0, "cls_dim": 5,
                "embedding_dim": 4, "fz": 4, "nz": None, "set_encoder": None,
                "e_source": None, "settf_windows": 4, "settf_t_max": 8,
                "budgets": list(pq.BUDGETS), "ctx_encoding": "onehot"}
        path = tmp_path / "cls_only_s0.pt"
        torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
        loaded, _ = g34.load_pq_model(path)
        assert loaded.task_branch == "none" and loaded.ctx_dim == 0
        assert g34.model_task_inputs(loaded, torch.zeros(4), torch.zeros(3)) \
            == (None, None)


# ---------------------------------------------------------------- 候选目录
class TestLoadCatalog:
    def test_falls_back_to_construction_when_file_absent(self, tmp_path):
        from autocls.space.cls_space import CLSSpace

        space = CLSSpace.from_yaml()
        cat, ggs, ggs_key = g34.load_catalog(tmp_path, space,
                                             slices=["Handwriting_00"])
        assert len(cat["Handwriting_00"]) == pq.N_SHARED + pq.N_RANDOM
        assert ggs_key and ggs is not None

    def test_file_and_construction_agree_on_keys(self, tmp_path):
        from autocls.space.cls_space import CLSSpace

        space = CLSSpace.from_yaml()
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=1,
                                budgets=SMOKE_BUDGETS)
        from_file, _c, _k = g34.load_catalog(tmp_path, space)
        built, _c2, _k2 = g34.load_catalog(tmp_path / "nope", space,
                                           slices=["Handwriting_00"])
        assert [k for _o, _c, k in from_file["Handwriting_00"]] \
            == [k for _o, _c, k in built["Handwriting_00"]]

    def test_unknown_schema_falls_back_with_slices(self, tmp_path):
        from autocls.space.cls_space import CLSSpace

        (tmp_path / "cls_catalog.json").write_text(
            json.dumps({"per_slice": {}}), encoding="utf-8")
        cat, _g, _k = g34.load_catalog(tmp_path, CLSSpace.from_yaml(),
                                       slices=["Handwriting_00"])
        assert len(cat["Handwriting_00"]) == pq.N_SHARED + pq.N_RANDOM

    def test_slice_with_shared_only_candidates_survives(self, tmp_path):
        """32 的 smoke 目录 random={} —— 只按 random 建表会让切片整个消失。"""
        from autocls.space.cls_space import CLSSpace

        shared = [{"cls_key": "aa" * 6, "cls": dataclasses.asdict(pq.ggs_cls())}]
        (tmp_path / "cls_catalog.json").write_text(json.dumps(
            {"shared": shared, "random": {},
             "ggs": {"cls_key": "bb" * 6,
                     "cls": dataclasses.asdict(pq.ggs_cls())}}),
            encoding="utf-8")
        cat, _g, ggs_key = g34.load_catalog(tmp_path, CLSSpace.from_yaml(),
                                            slices=["Handwriting_00"])
        assert list(cat) == ["Handwriting_00"]
        assert [k for _o, _c, k in cat["Handwriting_00"]] == ["aa" * 6]
        assert ggs_key == "bb" * 6

    def test_unknown_schema_without_slices_fails_fast(self, tmp_path):
        from autocls.space.cls_space import CLSSpace

        (tmp_path / "cls_catalog.json").write_text(
            json.dumps({"per_slice": {}}), encoding="utf-8")
        with pytest.raises(ValueError):
            g34.load_catalog(tmp_path, CLSSpace.from_yaml())


# ---------------------------------------------------------------- 门 / 分层 / 成本
def _rows(deltas, budgets=(60, 120, 290), datasets=("Handwriting",)):
    """构造 per-instance regret 行：tclsc1 = random1 − δ、clsb1 = random1 − δ/2。"""
    out = []
    for i, d in enumerate(deltas):
        b = budgets[i % len(budgets)]
        ds = datasets[i % len(datasets)]
        out.append({
            "fold": ds, "dataset": ds, "slice": f"{ds}_{i % 6:02d}", "budget": b,
            "regret": {"oracle": 0.0, "tclsc1": 0.30 - d, "tclsc3": 0.20 - d,
                       "clsb1": 0.30 - d / 2, "clsb3": 0.22,
                       "clsonly1": 0.31, "ggs": 0.25,
                       "random1": 0.30, "random3": 0.21, "bottom": 0.5},
            "seconds_top3": 3.0, "seconds_catalog": 20.0,
        })
    return out


class TestGates:
    def test_d1_passes_when_tclsc_beats_random(self):
        rows = _rows([0.02 + 0.001 * i for i in range(126)])
        g = g34.gate(rows, "tclsc1", "random1")
        assert g["n"] == 126 and g["pass"] is True and g["p"] < 0.05

    def test_d1_fails_when_no_difference(self):
        rows = _rows([0.0] * 30)
        g = g34.gate(rows, "tclsc1", "random1")
        assert g["pass"] is False and g["p"] == pytest.approx(1.0)

    def test_d2_uses_clsb_arm(self):
        rows = _rows([0.02 + 0.001 * i for i in range(126)])
        g = g34.gate(rows, "tclsc1", "clsb1")
        assert g["arm_a"] == "tclsc1" and g["arm_b"] == "clsb1"
        assert g["pass"] is True

    def test_gate_missing_arm_reports_reason(self):
        rows = _rows([0.02] * 10)
        for r in rows:
            r["regret"].pop("clsb1")
        g = g34.gate(rows, "tclsc1", "clsb1")
        assert g["p"] is None and g["pass"] is None and g["reason"]


class TestStrata:
    def test_by_budget_and_by_dataset_shape(self):
        rows = _rows([0.02] * 12, budgets=(60, 120, 290),
                     datasets=("Handwriting", "Cricket"))
        by_b = g34.strata_table(rows, "budget")
        by_d = g34.strata_table(rows, "dataset")
        assert sorted(by_b) == ["120", "290", "60"]
        assert sorted(by_d) == ["Cricket", "Handwriting"]
        for tbl in (by_b, by_d):
            for cell in tbl.values():
                assert cell["n"] >= 1
                assert "tclsc1" in cell["mean_regret"]
                assert "D1" in cell and "D2" in cell

    def test_stratum_means_are_plain_means(self):
        rows = _rows([0.0, 0.1], budgets=(60,))
        cell = g34.strata_table(rows, "budget")["60"]
        assert cell["n"] == 2
        assert cell["mean_regret"]["tclsc1"] == pytest.approx((0.30 + 0.20) / 2)


class TestCostTable:
    def test_sums_factor_and_speedup(self):
        rows = _rows([0.0] * 3, budgets=(60, 120, 290))
        cost = g34.cost_table(rows, full_iters=2320)
        assert sorted(cost) == ["120", "290", "60", "pooled"]
        c60 = cost["60"]
        assert c60["zero_shot_top3_s"] == pytest.approx(3.0)
        assert c60["catalog_direct_s"] == pytest.approx(20.0)
        assert c60["full_iters_factor"] == pytest.approx(2320 / 60)
        assert c60["catalog_direct_full_est_s"] == pytest.approx(20.0 * 2320 / 60)
        assert c60["speedup_vs_direct"] == pytest.approx(20.0 / 3.0)
        assert c60["speedup_vs_direct_full"] == pytest.approx(
            20.0 * 2320 / 60 / 3.0)

    def test_pooled_row_sums_all_budgets(self):
        rows = _rows([0.0] * 3, budgets=(60, 120, 290))
        cost = g34.cost_table(rows, full_iters=2320)
        assert cost["pooled"]["zero_shot_top3_s"] == pytest.approx(9.0)
        assert cost["pooled"]["catalog_direct_s"] == pytest.approx(60.0)


# ---------------------------------------------------------------- D3
class TestD3:
    PICKS = {
        t: {str(b): {"main": [{"cls_key": f"m{t}{b}{i}"} for i in range(3)],
                     "cls_b": [{"cls_key": f"c{t}{b}{i}"} for i in range(3)]}
            for b in pq.BUDGETS}
        for t in pq.FOLD_TARGETS
    }

    def test_pass_when_enough_instances_within_margin(self):
        # main 每格都命中（best = oracle − 0.01 ≥ oracle − 0.02）；cls_b 全不中
        def r_of(fold, b, key):
            return 0.99 if key.startswith("m") else 0.50

        rep = g34.d3_report(self.PICKS, r_of, lambda f, b: 1.00,
                            margin=0.02, min_hits=11)
        assert rep["n_instances"] == 21
        assert rep["per_arm"]["main"]["hits"] == 21
        assert rep["per_arm"]["main"]["pass"] is True
        assert rep["per_arm"]["cls_b"]["hits"] == 0
        assert rep["per_arm"]["cls_b"]["pass"] is False
        assert rep["difference"] == 21

    def test_fails_just_below_threshold(self):
        hit_folds = set(list(pq.FOLD_TARGETS)[:3])       # 3 折 × 3 预算 = 9 < 11

        def r_of(fold, b, key):
            return 0.99 if fold in hit_folds else 0.50

        rep = g34.d3_report(self.PICKS, r_of, lambda f, b: 1.00,
                            margin=0.02, min_hits=11)
        assert rep["per_arm"]["main"]["hits"] == 9
        assert rep["per_arm"]["main"]["pass"] is False

    def test_margin_boundary_is_inclusive(self):
        rep = g34.d3_report(self.PICKS, lambda f, b, k: 0.98,
                            lambda f, b: 1.00, margin=0.02, min_hits=1)
        assert rep["per_arm"]["main"]["hits"] == 21

    def test_missing_rprime_counts_as_miss_and_is_listed(self):
        rep = g34.d3_report(self.PICKS, lambda f, b, k: None,
                            lambda f, b: 1.00, margin=0.02, min_hits=11)
        main = rep["per_arm"]["main"]
        assert main["hits"] == 0 and main["n_evaluated"] == 0
        assert main["instances"][0]["missing"], "缺失的 pick 必须逐个列出"


# ---------------------------------------------------------------- 进化驱动
def _stub_evolution(monkeypatch, log):
    """打桩 evolutionary_search：只记调用、返回 3 条确定性条目。"""
    space = None

    def fake(score_pairs, space_, encoder, *, ks, kp, generations, p_cross,
             p_mut, top_k, rng, **kw):
        nonlocal space
        space = space_
        draw = int(rng.integers(10_000))
        log.append({"ks": ks, "kp": kp, "generations": generations,
                    "top_k": top_k, "draw": draw})
        cands = [space_.sample_random(np.random.default_rng(draw + i))
                 for i in range(top_k)]
        import dataclasses

        from autocls.labeling import cls_key
        return [{"cls": dataclasses.asdict(c), "cls_key": cls_key(c),
                 "wins": top_k - i, "generation_found": i}
                for i, c in enumerate(cands)]

    monkeypatch.setattr(g34, "evolutionary_search", fake)
    return log


class TestEvolveDriver:
    def test_picks_file_has_21_instances_times_arms(self, tmp_path, monkeypatch):
        log: list = []
        _stub_evolution(monkeypatch, log)
        g34.write_smoke_fixture(tmp_path, targets=pq.FOLD_TARGETS,
                                n_slices=1, budgets=pq.BUDGETS)
        g34.main(["--evolve", "--outdir", str(tmp_path)])
        doc = json.loads((tmp_path / "evo_picks.json").read_text(encoding="utf-8"))
        assert sorted(doc) == sorted(pq.FOLD_TARGETS)
        n_cells = 0
        for target, per_b in doc.items():
            assert sorted(per_b) == sorted(str(b) for b in pq.BUDGETS)
            for b, per_arm in per_b.items():
                assert sorted(per_arm) == ["cls_b", "main"]
                for arm, picks in per_arm.items():
                    assert len(picks) == 3
                    assert {"cls", "cls_key", "wins", "generation_found"} \
                        <= set(picks[0])
                    n_cells += 1
        assert n_cells == 21 * 2
        assert len(log) == 21 * 2

    def test_rng_is_deterministic_per_fold_and_budget(self, tmp_path, monkeypatch):
        log1: list = []
        _stub_evolution(monkeypatch, log1)
        g34.write_smoke_fixture(tmp_path, targets=pq.FOLD_TARGETS[:2],
                                n_slices=1, budgets=pq.BUDGETS)
        g34.main(["--evolve", "--outdir", str(tmp_path),
                  "--folds", *pq.FOLD_TARGETS[:2]])
        draws1 = [e["draw"] for e in log1]
        log2: list = []
        _stub_evolution(monkeypatch, log2)
        g34.main(["--evolve", "--outdir", str(tmp_path),
                  "--folds", *pq.FOLD_TARGETS[:2]])
        assert [e["draw"] for e in log2] == draws1
        # 不同折/预算的 rng 必须不同（否则 21 格是同一次搜索复制 21 遍）；
        # 同 (折, 预算) 的两个臂**故意共用**随机流——初代采样一致，两臂之差
        # 只来自比较器本身（brief 的 rng 键里没有臂）。
        n_arms = len(g34.EVO_ARMS)          # 调用序 = 折 → 预算 → 臂（确定性）
        per_arm = [draws1[i::n_arms] for i in range(n_arms)]
        for i, draws in enumerate(per_arm):
            assert len(set(draws)) == len(draws), f"臂 {i} 的各格 rng 撞车"
        assert per_arm[0] == per_arm[1]


# ---------------------------------------------------------------- evo 真评估
class _FakeSliced:
    def __init__(self, task_id):
        self.task_id = task_id
        self.pool = "source"
        self.classes = (0, 1)
        self.channels = (0,)
        self.sample_indices = tuple(range(8))
        self.tag = "smoke"
        self.task_type = "classification"


class _FakeDataset:
    def __init__(self, name, slices):
        self.name = name
        self.X = np.zeros((8, 4, 1), dtype=np.float32)
        self.y = np.array([0, 1] * 4)
        self.entry = None
        self.registry = None
        self.slices = slices


def _install_fake_eval(monkeypatch, calls, budgets=pq.BUDGETS):
    def fake_datasets(names):
        return {n: _FakeDataset(n, [_FakeSliced(f"{n}_00")]) for n in names}

    def fake_traj(task, cls_cfg, X, y, *, weight_seed, split_seed, checkpoints,
                  **kw):
        calls.append((task.task_id, weight_seed, tuple(checkpoints)))
        return {int(b): {"r_prime": 0.5 + 0.001 * i, "acc": 0.5, "f1": 0.5,
                         "svm_c": 1.0, "n_train": 6, "n_eval": 2,
                         "weight_seed": weight_seed, "split_seed": split_seed,
                         "device_name": "fake", "swa": False,
                         "seconds": 1.0 + i, "error": None,
                         "checkpoint_iters": [int(x) for x in checkpoints],
                         "trajectory_seconds": 3.0}
                for i, b in enumerate(checkpoints)}

    monkeypatch.setattr(g34, "_load_datasets", fake_datasets)
    monkeypatch.setattr(g34, "EVALUATE_TRAJECTORY", fake_traj)
    return calls


class TestEvaluateEvo:
    def _picks(self, tmp_path, targets, budgets):
        from autocls.labeling import cls_key
        from autocls.space.cls_space import CLSSpace

        space = CLSSpace.from_yaml()
        doc = {}
        for ti, t in enumerate(targets):
            doc[t] = {}
            for b in budgets:
                doc[t][str(b)] = {}
                for ai, arm in enumerate(g34.EVO_ARMS):
                    picks = []
                    for i in range(3):
                        c = space.sample_random(
                            np.random.default_rng([ti, b, ai, i]))
                        import dataclasses
                        picks.append({"cls": dataclasses.asdict(c),
                                      "cls_key": cls_key(c), "wins": 3 - i,
                                      "generation_found": 0})
                    doc[t][str(b)][arm] = picks
        (tmp_path / "evo_picks.json").write_text(
            json.dumps(doc, indent=2), encoding="utf-8")
        return doc

    def test_worklist_dedups_and_records_claims(self, tmp_path):
        doc = self._picks(tmp_path, ["Handwriting"], (60, 120))
        # 人为让 60/main 与 120/cls_b 撞同一条 CLS
        doc["Handwriting"]["120"]["cls_b"][0] = doc["Handwriting"]["60"]["main"][0]
        wl = g34.evo_worklist(doc, budgets=(60, 120))
        keys = [e["cls_key"] for e in wl]
        assert len(keys) == len(set(keys)), "同一条 CLS 只应跑一条轨迹"
        dup = [e for e in wl
               if e["cls_key"] == doc["Handwriting"]["60"]["main"][0]["cls_key"]]
        assert len(dup) == 1
        assert dup[0]["origin"] == "evo_main"          # 先到者定 origin
        assert dup[0]["budget_searched"] == 60
        assert len(dup[0]["claims"]) == 2
        assert {"arm", "budget", "rank"} <= set(dup[0]["claims"][0])
        assert all(e["slice"] == "Handwriting_00" for e in wl)

    def test_evaluate_writes_rows_then_resume_skips(self, tmp_path, monkeypatch):
        self._picks(tmp_path, ["Handwriting"], (60, 120))
        calls: list = []
        _install_fake_eval(monkeypatch, calls)
        g34.main(["--evaluate-evo", "--resume", "--outdir", str(tmp_path)])
        rows = [json.loads(x) for x in
                (tmp_path / "evo_cache.jsonl").read_text(encoding="utf-8")
                .splitlines() if x.strip()]
        n_traj = len(calls)
        assert n_traj > 0
        assert len(rows) == n_traj * len(pq.BUDGETS)
        r = rows[0]
        for f in ("dataset", "task_id", "cls_key", "origin", "iters",
                  "checkpoint_iters", "trajectory_id", "fold",
                  "budget_searched", "cls", "seconds", "r_prime"):
            assert f in r, f
        assert r["origin"] in ("evo_main", "evo_cls_b")
        assert r["fold"] == "Handwriting"
        assert r["checkpoint_iters"] == list(pq.BUDGETS)

        g34.main(["--evaluate-evo", "--resume", "--outdir", str(tmp_path)])
        assert len(calls) == n_traj, "resume 必须按缓存键跳过已跑轨迹"
        rows2 = [x for x in (tmp_path / "evo_cache.jsonl")
                 .read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(rows2) == len(rows)

    def test_reuses_catalog_cache_when_pick_already_evaluated(
            self, tmp_path, monkeypatch):
        doc = self._picks(tmp_path, ["Handwriting"], (60,))
        seeded = doc["Handwriting"]["60"]["main"][0]["cls_key"]
        with open(tmp_path / "rprime_cache.jsonl", "w", encoding="utf-8") as fh:
            for b in pq.BUDGETS:
                fh.write(json.dumps({
                    "dataset": "Handwriting", "task_id": "Handwriting_00",
                    "cls_key": seeded, "origin": "shared", "seed": 0,
                    "iters": b, "checkpoint_iters": list(pq.BUDGETS),
                    "r_prime": 0.8, "seconds": 1.0}) + "\n")
        calls: list = []
        _install_fake_eval(monkeypatch, calls)
        g34.main(["--evaluate-evo", "--resume", "--outdir", str(tmp_path)])
        wl = g34.evo_worklist(doc, budgets=(60,))
        assert len(calls) == len(wl) - 1, "已在主缓存里的 pick 不应重跑"


# ================================================================ 评审修复轮 1
def _one_cls(seed=7):
    from autocls.labeling import cls_key
    from autocls.space.cls_space import CLSSpace
    c = CLSSpace.from_yaml().sample_random(np.random.default_rng(seed))
    return c, cls_key(c)


def _picks_one(tmp_path, cls_and_key, budget=60):
    c, k = cls_and_key
    doc = {"Handwriting": {str(budget): {
        "main": [{"cls": dataclasses.asdict(c), "cls_key": k, "wins": 3,
                  "generation_found": 0}],
        "cls_b": []}}}
    (tmp_path / "evo_picks.json").write_text(json.dumps(doc), encoding="utf-8")
    return doc


class TestTerminalResume:
    """发现 1：看门狗隔离的轨迹只留一行 max-budget 的 null+error，不得反复重跑。"""

    def test_terminal_row_is_not_retried(self, tmp_path, monkeypatch):
        c, k = _one_cls()
        _picks_one(tmp_path, (c, k))
        (tmp_path / "evo_cache.jsonl").write_text(json.dumps({
            "task_id": "Handwriting_00", "cls_key": k, "iters": max(pq.BUDGETS),
            "checkpoint_iters": list(pq.BUDGETS), "r_prime": None,
            "error": "watchdog quarantine", "origin": "evo_main"}) + "\n",
            encoding="utf-8")
        calls = []
        _install_fake_eval(monkeypatch, calls)
        g34.main(["--evaluate-evo", "--resume", "--outdir", str(tmp_path)])
        assert calls == [], "terminal 轨迹被重跑了（会无限循环并追加重复行）"
        rows = [x for x in (tmp_path / "evo_cache.jsonl")
                .read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(rows) == 1, "terminal 轨迹被追加了重复的 max-budget 行"

    def test_partial_trajectory_is_still_retried(self, tmp_path, monkeypatch):
        c, k = _one_cls()
        _picks_one(tmp_path, (c, k))
        (tmp_path / "evo_cache.jsonl").write_text(json.dumps({
            "task_id": "Handwriting_00", "cls_key": k, "iters": pq.BUDGETS[0],
            "checkpoint_iters": list(pq.BUDGETS), "r_prime": 0.5,
            "error": None, "origin": "evo_main"}) + "\n", encoding="utf-8")
        calls = []
        _install_fake_eval(monkeypatch, calls)
        g34.main(["--evaluate-evo", "--resume", "--outdir", str(tmp_path)])
        assert len(calls) == 1, "partial 轨迹应当重跑"

    def test_terminal_in_main_cache_also_skipped(self, tmp_path, monkeypatch):
        c, k = _one_cls()
        _picks_one(tmp_path, (c, k))
        (tmp_path / "rprime_cache.jsonl").write_text(json.dumps({
            "task_id": "Handwriting_00", "cls_key": k, "iters": max(pq.BUDGETS),
            "checkpoint_iters": list(pq.BUDGETS), "r_prime": None,
            "error": "diverged", "origin": "shared"}) + "\n", encoding="utf-8")
        calls = []
        _install_fake_eval(monkeypatch, calls)
        g34.main(["--evaluate-evo", "--resume", "--outdir", str(tmp_path)])
        assert calls == []

    def test_uses_script32_state_machine(self):
        g32 = importlib.import_module("scripts.32_pq_generate_rprime")
        assert g34.trajectory_state is g32.trajectory_state
        assert (g34.STATE_DONE, g34.STATE_TERMINAL) == (g32.DONE, g32.TERMINAL)


class TestIronLaw6AtPointOfUse:
    """发现 2：用折特征/折模型之前必须核对考侧没进过训练侧。"""

    def _meta(self, tmp_path, target, fit_datasets):
        d = tmp_path / "folds" / target
        d.mkdir(parents=True, exist_ok=True)
        (d / "shared_ts2vec.meta.json").write_text(
            json.dumps({"fold": target, "fit_datasets": fit_datasets}),
            encoding="utf-8")

    def test_leaky_fit_pool_raises(self, tmp_path):
        self._meta(tmp_path, "Handwriting", ["Handwriting", "Cricket"])
        with pytest.raises(AssertionError, match="铁律 6"):
            g34.assert_fold_features_compliant(tmp_path, "Handwriting")

    def test_compliant_fit_pool_passes(self, tmp_path):
        self._meta(tmp_path, "Handwriting", sorted(pq.fold_split("Handwriting")[0]))
        meta = g34.assert_fold_features_compliant(tmp_path, "Handwriting")
        assert "Handwriting" not in meta["fit_datasets"]

    def test_missing_meta_warns_only(self, tmp_path, capsys):
        assert g34.assert_fold_features_compliant(tmp_path, "Cricket") is None
        assert "铁律 6" in capsys.readouterr().out

    def test_reuses_script33_assertion(self):
        g33 = importlib.import_module("scripts.33_pq_comparator_loo")
        assert g34.assert_fold_fit_pool is g33.assert_fold_fit_pool

    def test_smoke_fixture_writes_compliant_fold_meta(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=1,
                                budgets=SMOKE_BUDGETS)
        meta = g34.assert_fold_features_compliant(tmp_path, "Handwriting")
        assert meta is not None and "Handwriting" not in meta["fit_datasets"]

    def test_cmd_rank_refuses_leaky_fold(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=1,
                                budgets=SMOKE_BUDGETS)
        self._meta(tmp_path, "Handwriting", ["Handwriting", "Cricket"])
        with pytest.raises(AssertionError, match="铁律 6"):
            g34.main(["--rank", "--outdir", str(tmp_path), "--folds",
                      "Handwriting"])

    def test_model_fold_mismatch_raises(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting", "Cricket"),
                                n_slices=1, budgets=SMOKE_BUDGETS)
        alien = tmp_path / "models" / "Cricket" / "main_s0.pt"
        target = tmp_path / "models" / "Handwriting" / "main_s0.pt"
        target.write_bytes(alien.read_bytes())
        with pytest.raises(ValueError, match="fold"):
            g34.load_pq_model(target, expect_fold="Handwriting")
        m, meta = g34.load_pq_model(target)
        assert meta["fold"] == "Cricket" and m is not None


class TestFeatureFoldContract:
    """协调方增补：meta['feature_fold'] 决定 E 从哪个特征目录来。"""

    def test_key_present_wins_over_fold(self):
        assert g34.feature_fold_of({"fold": "S2", "feature_fold": "all8"}) == "all8"
        assert g34.feature_fold_of(
            {"fold": "Handwriting", "feature_fold": None}) is None

    def test_falls_back_to_fold_only_when_key_absent(self):
        assert g34.feature_fold_of({"fold": "Handwriting"}) == "Handwriting"
        assert g34.feature_fold_of({}, "Cricket") == "Cricket"

    def test_smoke_fixture_carries_feature_fold(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=1,
                                budgets=SMOKE_BUDGETS)
        mdir = tmp_path / "models" / "Handwriting"
        _m, meta = g34.load_pq_model(mdir / "main_s0.pt")
        assert meta["feature_fold"] == "Handwriting"
        _m2, meta2 = g34.load_pq_model(mdir / "cls_b_s0.pt")
        assert meta2["feature_fold"] is None      # 不吃 E 的臂无特征目录

    def test_all8_features_rejected_on_s1_fold(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=1,
                                budgets=SMOKE_BUDGETS)
        path = tmp_path / "models" / "Handwriting" / "main_s0.pt"
        blob = torch.load(path, map_location="cpu", weights_only=True)
        blob["meta"]["feature_fold"] = "all8"
        torch.save(blob, path)
        with pytest.raises(ValueError, match="all8"):
            g34.load_pq_model(path, expect_fold="Handwriting")

    def test_s2_model_allowed_only_when_explicitly_requested(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=1,
                                budgets=SMOKE_BUDGETS)
        path = tmp_path / "models" / "Handwriting" / "main_s0.pt"
        blob = torch.load(path, map_location="cpu", weights_only=True)
        blob["meta"]["fold"] = "S2"
        blob["meta"]["feature_fold"] = "all8"
        torch.save(blob, path)
        with pytest.raises(ValueError, match="fold"):
            g34.load_pq_model(path, expect_fold="Handwriting")
        m, meta = g34.load_pq_model(path, expect_fold="S2")
        assert meta["feature_fold"] == "all8" and m is not None

    def test_features_read_from_feature_fold_dir(self, tmp_path):
        """把特征目录改名成 all8 并让 meta 指过去，S2 臂仍要读到 E。"""
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=1,
                                budgets=SMOKE_BUDGETS)
        src = tmp_path / "folds" / "Handwriting" / "task_features"
        dst = tmp_path / "folds" / "all8" / "task_features"
        dst.mkdir(parents=True)
        for f in src.glob("*.npz"):
            (dst / f.name).write_bytes(f.read_bytes())
        w = g34.load_slice_windows(tmp_path, "all8", "Handwriting_00")
        assert w is not None and w.ndim == 3
        assert g34.load_slice_windows(tmp_path, "nosuch", "Handwriting_00") is None

    def test_target_slices_honours_feature_fold(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=2,
                                budgets=SMOKE_BUDGETS)
        space = __import__("autocls.space.cls_space", fromlist=["CLSSpace"]).CLSSpace
        cat, _g, _k = g34.load_catalog(tmp_path, space.from_yaml())
        assert len(g34.target_slices(tmp_path, "Handwriting", cat)) == 2
        assert g34.target_slices(tmp_path, "Handwriting", cat,
                                 feature_fold="all8") == []


class TestBudgetContract:
    """发现 5：ctx one-hot 的分量顺序由训练时的 budgets 定义，重排必须炸。"""

    def _model(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=1,
                                budgets=SMOKE_BUDGETS)
        return tmp_path / "models" / "Handwriting" / "main_s0.pt"

    def test_reordered_budgets_raise(self, tmp_path):
        path = self._model(tmp_path)
        with pytest.raises(ValueError, match="budgets"):
            g34.load_pq_model(path, expect_budgets=tuple(reversed(SMOKE_BUDGETS)))

    def test_subset_budgets_raise(self, tmp_path):
        path = self._model(tmp_path)
        with pytest.raises(ValueError, match="budgets"):
            g34.load_pq_model(path, expect_budgets=SMOKE_BUDGETS[:2])

    def test_matching_budgets_pass(self, tmp_path):
        path = self._model(tmp_path)
        m, meta = g34.load_pq_model(path, expect_budgets=SMOKE_BUDGETS)
        assert tuple(meta["budgets"]) == SMOKE_BUDGETS and m.ctx_dim == 3

    def test_non_onehot_ctx_encoding_raises(self, tmp_path):
        path = self._model(tmp_path)
        blob = torch.load(path, map_location="cpu", weights_only=True)
        blob["meta"]["ctx_encoding"] = "ordinal"
        torch.save(blob, path)
        with pytest.raises(ValueError, match="ctx_encoding"):
            g34.load_pq_model(path, expect_budgets=SMOKE_BUDGETS)

    def test_cli_reordered_budgets_fail_fast(self, tmp_path):
        self._model(tmp_path)
        with pytest.raises(ValueError, match="budgets"):
            g34.main(["--rank", "--outdir", str(tmp_path), "--folds",
                      "Handwriting", "--budgets", "12", "4", "8"])


class TestSubsetMerge:
    """发现 4：--folds/--arms 子集跑必须并入既有文件，不得抹掉别的折。"""

    def test_deep_merge_keeps_untouched_branches(self):
        prev = {"A": {"60": {"s0": {"main": ["x"]}}}, "B": {"60": {}}}
        new = {"A": {"60": {"s0": {"main": ["y"]}}}}
        out = g34.deep_merge(prev, new)
        assert out["A"]["60"]["s0"]["main"] == ["y"]
        assert out["B"] == {"60": {}}

    def test_rank_subset_preserves_other_folds(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting", "Cricket"),
                                n_slices=1, budgets=SMOKE_BUDGETS)
        g34.main(["--rank", "--outdir", str(tmp_path)])
        doc = json.loads((tmp_path / "rankings.json").read_text(encoding="utf-8"))
        assert sorted(doc) == ["Cricket", "Handwriting"]
        cricket_before = doc["Cricket"]
        g34.main(["--rank", "--outdir", str(tmp_path), "--folds", "Handwriting"])
        doc2 = json.loads((tmp_path / "rankings.json").read_text(encoding="utf-8"))
        assert sorted(doc2) == ["Cricket", "Handwriting"], "子集跑抹掉了别的折"
        assert doc2["Cricket"] == cricket_before

    def test_rank_arm_subset_preserves_other_arms(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=1,
                                budgets=SMOKE_BUDGETS)
        g34.main(["--rank", "--outdir", str(tmp_path)])
        g34.main(["--rank", "--outdir", str(tmp_path), "--arms", "main"])
        doc = json.loads((tmp_path / "rankings.json").read_text(encoding="utf-8"))
        arms = doc["Handwriting"][str(SMOKE_BUDGETS[0])]["Handwriting_00"]
        assert sorted(arms) == sorted(g34.RANK_ARMS), "臂子集跑抹掉了别的臂"

    def test_evolve_subset_preserves_other_folds(self, tmp_path, monkeypatch):
        log = []
        _stub_evolution(monkeypatch, log)
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting", "Cricket"),
                                n_slices=1, budgets=pq.BUDGETS)
        g34.main(["--evolve", "--outdir", str(tmp_path), "--folds", "Cricket"])
        g34.main(["--evolve", "--outdir", str(tmp_path), "--folds", "Handwriting"])
        doc = json.loads((tmp_path / "evo_picks.json").read_text(encoding="utf-8"))
        assert sorted(doc) == ["Cricket", "Handwriting"]

    def test_write_json_merged_is_atomic(self, tmp_path):
        path = tmp_path / "r.json"
        g34.write_json_merged(path, {"A": 1})
        g34.write_json_merged(path, {"B": 2})
        assert json.loads(path.read_text(encoding="utf-8")) == {"A": 1, "B": 2}
        assert not list(tmp_path.glob("*.tmp")), "临时文件没清掉"

    def test_unknown_arm_is_argparse_error(self, tmp_path):
        with pytest.raises(SystemExit):
            g34.main(["--rank", "--outdir", str(tmp_path), "--arms", "nope"])
        with pytest.raises(SystemExit):
            g34.main(["--evolve", "--outdir", str(tmp_path), "--arms", "cls_only"])


class TestProductionGuards:
    """发现 6：合成 fixture 绝不许落进 results/mx3_pq，标记也不许在那里被采信。"""

    def test_write_smoke_fixture_refuses_production(self, tmp_path, monkeypatch):
        prod = tmp_path / "prod"
        prod.mkdir()
        monkeypatch.setattr(pq, "OUT", prod)
        with pytest.raises(SystemExit):
            g34.write_smoke_fixture(prod, targets=("Handwriting",), n_slices=1,
                                    budgets=SMOKE_BUDGETS)
        assert not (prod / g34.FIXTURE_MARKER).exists()
        assert not (prod / "rprime_cache.jsonl").exists()

    def test_real_production_dir_is_refused(self):
        with pytest.raises(SystemExit):
            g34.write_smoke_fixture(pq.OUT)
        assert not (pq.OUT / g34.FIXTURE_MARKER).exists()

    def test_resolve_budgets_refuses_marker_in_production(self, tmp_path,
                                                          monkeypatch):
        prod = tmp_path / "prod"
        prod.mkdir()
        monkeypatch.setattr(pq, "OUT", prod)
        (prod / g34.FIXTURE_MARKER).write_text(
            json.dumps({"budgets": [4, 8, 12]}), encoding="utf-8")
        with pytest.raises(SystemExit):
            g34.resolve_budgets(prod, None)

    def test_resolve_budgets_honours_marker_elsewhere(self, tmp_path):
        (tmp_path / g34.FIXTURE_MARKER).write_text(
            json.dumps({"budgets": [4, 8, 12]}), encoding="utf-8")
        assert g34.resolve_budgets(tmp_path, None) == (4, 8, 12)
        assert g34.resolve_budgets(tmp_path, [60, 120, 290]) == (60, 120, 290)

    def test_no_marker_falls_back_to_protocol_budgets(self, tmp_path):
        assert g34.resolve_budgets(tmp_path, None) == tuple(pq.BUDGETS)


class TestFullGridGuard:
    """发现 3：D1/D2 的分母写死 126，子集上算出的 p 不得印成过/不过。"""

    def test_expected_instances_is_126(self):
        assert g34.expected_instances(pq.BUDGETS) == 126

    def test_incomplete_withdraws_verdict(self):
        g = {"n": 12, "p": 0.001, "pass": True, "reason": None}
        out = g34.mark_incomplete(g, n_expected=126, allow_incomplete=False)
        assert out["incomplete"] is True and out["pass"] is None
        assert out["p"] == 0.001 and "126" in out["reason"]

    def test_allow_incomplete_keeps_verdict(self):
        g = {"n": 12, "p": 0.001, "pass": True, "reason": None}
        out = g34.mark_incomplete(g, n_expected=126, allow_incomplete=True)
        assert out["incomplete"] is True and out["pass"] is True

    def test_full_grid_keeps_verdict(self):
        g = {"n": 126, "p": 0.001, "pass": True, "reason": None}
        out = g34.mark_incomplete(g, n_expected=126, allow_incomplete=False)
        assert out["incomplete"] is False and out["pass"] is True
        assert out["reason"] is None

    def test_smoke_analysis_marks_d1_d2_incomplete(self, tmp_path):
        g34.main(["--smoke", "--outdir", str(tmp_path)])
        an = json.loads((tmp_path / "search_analysis.json")
                        .read_text(encoding="utf-8"))
        for code in ("D1", "D2"):
            assert an["gates"][code]["incomplete"] is True
            assert an["gates"][code]["pass"] is None
            assert an["gates"][code]["n_expected"] == 126
        assert an["gates"]["D3"]["incomplete"] is True
        assert an["d3"]["per_arm"]["main"]["pass"] is None


class TestBuildInstances:
    """发现 7 附带：真值不全的 fail-fast / --allow-incomplete / 垫底四分位计数。"""

    def _fixture(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=1,
                                budgets=SMOKE_BUDGETS)
        g34.main(["--rank", "--outdir", str(tmp_path), "--folds", "Handwriting"])

    def test_missing_truth_fails_fast(self, tmp_path):
        self._fixture(tmp_path)
        rows = [x for x in (tmp_path / "rprime_cache.jsonl")
                .read_text(encoding="utf-8").splitlines() if x.strip()]
        (tmp_path / "rprime_cache.jsonl").write_text(
            "\n".join(rows[3:]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="真值不全"):
            g34.main(["--analyze", "--outdir", str(tmp_path)])

    def test_allow_incomplete_skips_and_records(self, tmp_path):
        self._fixture(tmp_path)
        rows = [x for x in (tmp_path / "rprime_cache.jsonl")
                .read_text(encoding="utf-8").splitlines() if x.strip()]
        keep = [r for r in rows
                if json.loads(r)["iters"] != SMOKE_BUDGETS[0]
                or json.loads(r)["cls_key"] != json.loads(rows[0])["cls_key"]]
        (tmp_path / "rprime_cache.jsonl").write_text(
            "\n".join(keep) + "\n", encoding="utf-8")
        g34.main(["--analyze", "--outdir", str(tmp_path), "--allow-incomplete"])
        an = json.loads((tmp_path / "search_analysis.json")
                        .read_text(encoding="utf-8"))
        assert an["n_skipped"] == 1
        assert an["skipped_instances"][0]["budget"] == SMOKE_BUDGETS[0]
        assert an["n_instances"] == len(SMOKE_BUDGETS) - 1
        assert an["gates"]["D1"]["incomplete"] is True
        assert an["gates"]["D1"]["pass"] is not None     # --allow-incomplete 放行

    def test_seconds_null_does_not_crash(self, tmp_path):
        self._fixture(tmp_path)
        out = []
        for line in (tmp_path / "rprime_cache.jsonl").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["seconds"] = None
                out.append(json.dumps(r))
        (tmp_path / "rprime_cache.jsonl").write_text("\n".join(out) + "\n",
                                                     encoding="utf-8")
        g34.main(["--analyze", "--outdir", str(tmp_path)])
        an = json.loads((tmp_path / "search_analysis.json")
                        .read_text(encoding="utf-8"))
        assert an["cost"]["pooled"]["zero_shot_top3_s"] == 0.0

    def test_avoid_worst_counts_bottom_quartile_picks(self):
        truth = {"k%02d" % i: i / 100 for i in range(20)}
        catalog = {"Handwriting_00": [("shared", None, k) for k in truth]}
        worst, best = "k00", "k19"
        rankings = {"Handwriting": {"60": {"Handwriting_00": {
            "main": [worst] + [k for k in truth if k != worst],
            "cls_b": [best] + [k for k in truth if k != best],
            "cls_only": [worst] + [k for k in truth if k != worst]}}}}
        cache = {("Handwriting_00", k, 60): {"r_prime": v, "seconds": 1.0}
                 for k, v in truth.items()}
        rows, skipped = g34.build_instances(rankings, catalog, "ggs", cache)
        assert skipped == [] and len(rows) == 1
        hits = rows[0]["bottom_quartile_hit"]
        assert hits["tclsc1"] is True and hits["clsonly1"] is True
        assert hits["clsb1"] is False
        assert rows[0]["arms"]["oracle"] == pytest.approx(0.19)
        assert rows[0]["regret"]["tclsc1"] == pytest.approx(0.19)


class TestEProvenance:
    """发现 7：排名用的 E 必须是本折 windows_repr 过**该模型自己的** set_encoder。"""

    def test_encode_tasks_receives_the_npz_windows(self, tmp_path, monkeypatch):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=1,
                                budgets=SMOKE_BUDGETS)
        with np.load(tmp_path / "folds" / "Handwriting" / "task_features"
                     / "Handwriting_00.npz") as z:
            windows = np.asarray(z["windows_repr"], dtype=np.float32)
            e_init = np.asarray(z["E_init"], dtype=np.float32)

        seen = []
        real_loader = g34.load_pq_model

        def recording_loader(path, **kw):
            model, meta = real_loader(path, **kw)
            if g34.uses_E(model):
                inner = model.encode_tasks

                def spy(windows_list):
                    seen.append(np.asarray(windows_list[0].cpu(),
                                           dtype=np.float32))
                    return inner(windows_list)

                model.encode_tasks = spy
            return model, meta

        monkeypatch.setattr(g34, "load_pq_model", recording_loader)
        g34.main(["--rank", "--outdir", str(tmp_path), "--folds", "Handwriting"])
        assert seen, "main 臂一次 encode_tasks 都没调用（E 没走 SetTF？）"
        for got in seen:
            assert got.shape == windows.shape
            np.testing.assert_allclose(got, windows, rtol=0, atol=0)
        assert seen[0].shape != e_init.shape     # E_init 绝不是排名用的 E

    def test_ranking_follows_windows_repr_not_e_init(self, tmp_path):
        g34.write_smoke_fixture(tmp_path, targets=("Handwriting",), n_slices=2,
                                budgets=SMOKE_BUDGETS)
        g34.main(["--rank", "--outdir", str(tmp_path), "--folds", "Handwriting"])
        before = json.loads((tmp_path / "rankings.json")
                            .read_text(encoding="utf-8"))
        feat = tmp_path / "folds" / "Handwriting" / "task_features"
        for npz in sorted(feat.glob("*.npz")):
            with np.load(npz) as z:
                d = {k: z[k] for k in z.files}
            rng = np.random.default_rng(99)
            d["windows_repr"] = (rng.standard_normal(d["windows_repr"].shape) * 5
                                 ).astype(np.float16)
            np.savez_compressed(npz, **d)        # 只动 windows_repr，E_init 不变
        g34.main(["--rank", "--outdir", str(tmp_path), "--folds", "Handwriting"])
        after = json.loads((tmp_path / "rankings.json")
                           .read_text(encoding="utf-8"))

        def arm(doc, a):
            return [doc["Handwriting"][b][s][a]
                    for b in doc["Handwriting"] for s in doc["Handwriting"][b]]

        assert arm(before, "main") != arm(after, "main"), \
            "换掉 windows_repr 后 main 排名没变——E 没真的进比较器"
        assert arm(before, "cls_b") == arm(after, "cls_b")
        assert arm(before, "cls_only") == arm(after, "cls_only")


# ---------------------------------------------------------------- 端到端 smoke
class TestSmoke:
    def test_smoke_produces_rankings_and_analysis(self, tmp_path):
        g34.main(["--smoke", "--outdir", str(tmp_path)])
        rk = json.loads((tmp_path / "rankings.json").read_text(encoding="utf-8"))
        assert sorted(rk) == sorted(SMOKE_TARGETS)
        for target, per_b in rk.items():
            assert sorted(per_b) == sorted(str(b) for b in SMOKE_BUDGETS)
            for b, per_slice in per_b.items():
                assert per_slice, f"{target}@{b} 无实例"
                for s, per_arm in per_slice.items():
                    assert sorted(per_arm) == sorted(g34.RANK_ARMS)
                    for arm, keys in per_arm.items():
                        assert len(keys) == len(set(keys)) == 20

        an = json.loads((tmp_path / "search_analysis.json")
                        .read_text(encoding="utf-8"))
        assert an["n_instances"] == len(SMOKE_TARGETS) * len(SMOKE_BUDGETS) * 2
        assert set(an["gates"]) >= {"D1", "D2"}
        assert "by_budget" in an and "by_dataset" in an
        assert "cost" in an and "avoid_worst" in an
        assert an["mean_regret"]["pooled"]["oracle"] == pytest.approx(0.0)

    def test_smoke_is_idempotent(self, tmp_path):
        g34.main(["--smoke", "--outdir", str(tmp_path)])
        first = (tmp_path / "rankings.json").read_text(encoding="utf-8")
        g34.main(["--smoke", "--outdir", str(tmp_path)])
        assert (tmp_path / "rankings.json").read_text(encoding="utf-8") == first

    def test_smoke_refuses_production_outdir(self):
        with pytest.raises(SystemExit):
            g34.main(["--smoke", "--outdir", str(pq.OUT)])
