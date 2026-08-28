"""共享种子交叉探针（scripts/9_shared_seed_probe.py）纯逻辑测试。

期望值全部手工推导（不以被测代码为 oracle）：
- 种子派生的核心性质：weight_seed 只依赖 (task_id, rep)、与 cls_key 无关——
  这是与生产口径 run_seed_for(task_id, cls_key) 的唯一设计差异，也是本探针
  能做 10×3 完整交叉的前提；
- 二因素 ANOVA 的 SS 分解（2×2 手算已知答案）；
- 三套噪声口径在合成矩阵上的解析行为（纯加性 g-only → 共享臂噪声恰为 0；
  纯交互 ε-only → 配对无降幅）；
- 判据档位与"两口径跨档取更保守档"预注册规则。
"""
import importlib

import numpy as np
import pytest

from autocls.labeling import run_seed_for, shared_seed_for

probe = importlib.import_module("scripts.9_shared_seed_probe")


class TestSharedSeedFor:
    def test_only_depends_on_task_and_rep(self):
        # 函数签名即不含 cls_key；值 = run_seed_for 的 "sharedseed:{rep}" 流
        assert shared_seed_for("t0", 1) == run_seed_for("t0", "sharedseed:1")

    def test_distinct_across_reps_and_tasks(self):
        assert len({shared_seed_for("t0", r) for r in (0, 1, 2)}) == 3
        assert shared_seed_for("t0", 0) != shared_seed_for("t1", 0)

    def test_disjoint_from_canonical_and_noise_streams(self):
        # 与生产 canonical（cls_key 流）和 2c 噪声探针（"{key}:noise{rep}" 流）不相交
        k = "8c4aa283beef"
        for rep in (0, 1, 2):
            s = shared_seed_for("Handwriting_00", rep)
            assert s != run_seed_for("Handwriting_00", k)
            assert s != run_seed_for("Handwriting_00", f"{k}:noise{rep}")


class TestSeedsFor:
    def test_weight_is_shared_split_is_canonical(self):
        w, sp = probe.seeds_for("t0", 2)
        assert w == shared_seed_for("t0", 2)
        assert sp == run_seed_for("t0", "train_eval_split")


class TestValidateRows:
    def _row(self, **kw):
        base = {"task_id": "t0", "cls_key": "k0", "seed": 1, "shared_seed": True,
                "weight_seed": shared_seed_for("t0", 1),
                "split_seed": run_seed_for("t0", "train_eval_split")}
        base.update(kw)
        return base

    def test_good_row_passes(self):
        assert probe.validate_rows([self._row()]) == []

    def test_wrong_weight_seed_caught(self):
        bad = probe.validate_rows([self._row(weight_seed=12345)])
        assert bad == [("t0", "k0", 1)]

    def test_missing_shared_seed_flag_caught(self):
        assert probe.validate_rows([self._row(shared_seed=False)])

    def test_cls_dependent_seed_caught(self):
        # 若实现回退到生产口径（种子含 cls_key），必须被抓出
        assert probe.validate_rows(
            [self._row(weight_seed=run_seed_for("t0", "k0"))])


class TestBuildCells:
    def test_filters_dataset_and_takes_first_two_tasks(self):
        class T:
            def __init__(self, tid):
                self.task_id = tid

        per_ds = [("Handwriting", None, None, [T("h0"), T("h1"), T("h2")]),
                  ("UniMiB-SHAR", None, None, [T("u0"), T("u1")])]
        cells = probe.build_cells(per_ds, {"Handwriting": 60, "UniMiB-SHAR": 30},
                                  [("c1", "k1"), ("c2", "k2")])
        # 只收 Handwriting 前 2 任务：2 任务 × 2 CLS × 3 rep = 12
        assert len(cells) == 12
        assert {c[0] for c in cells} == {"Handwriting"}
        assert {c[3].task_id for c in cells} == {"h0", "h1"}
        assert {c[7] for c in cells} == {60}
        assert {c[6] for c in cells} == {0, 1, 2}


class TestLoadCells:
    def test_last_row_wins_and_null_skipped(self):
        rows = [
            {"task_id": "t", "cls_key": "k", "seed": 0, "r_prime": 0.5},
            {"task_id": "t", "cls_key": "k", "seed": 0, "r_prime": 0.7},  # resume 追加
            {"task_id": "t", "cls_key": "k", "seed": 1, "r_prime": None},
        ]
        cells = probe.load_cells(rows)
        assert cells[("t", "k")] == {0: 0.7}


class TestBuildMatrix:
    def test_orders_by_catalog_and_drops_incomplete(self):
        cells = {("t", "ka"): {0: 0.1, 1: 0.2, 2: 0.3},
                 ("t", "kb"): {0: 0.4, 1: 0.5},              # 缺 rep2 → 剔除
                 ("t", "kc"): {0: 0.7, 1: 0.8, 2: 0.9}}
        m, used, dropped = probe.build_matrix(cells, "t", ["kc", "ka", "kb"])
        assert used == ["kc", "ka"]           # 目录顺序，不是字典序
        assert dropped == ["kb"]
        assert m.tolist() == [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]


class TestAnova2:
    def test_hand_computed_2x2(self):
        # [[0,1],[2,5]]：gm=2，cls_eff=±1.5，seed_eff=±1
        # SS_cls=2×(1.5²×2)=9，SS_seed=2×(1²×2)=4，SS_resid=4×0.25=1（总 14）
        out = probe.anova2(np.array([[0.0, 1.0], [2.0, 5.0]]))
        assert out["ss"]["cls"] == pytest.approx(9.0)
        assert out["ss"]["seed"] == pytest.approx(4.0)
        assert out["ss"]["resid"] == pytest.approx(1.0)
        assert out["ss_share"]["cls"] == pytest.approx(9 / 14)
        # MS_seed=4/1=4，MS_resid=1/1=1 → Var(g)̂=(4−1)/2=1.5，Var(ε)̂=1
        assert out["var_g_hat"] == pytest.approx(1.5)
        assert out["var_eps_hat"] == pytest.approx(1.0)

    def test_pure_additive_has_zero_resid(self):
        # f=[0,0.5] + g=[0,0.1]，无交互 → resid 份额恰为 0
        out = probe.anova2(np.array([[0.0, 0.1], [0.5, 0.6]]))
        assert out["ss"]["resid"] == pytest.approx(0.0, abs=1e-12)


class TestNoiseContrasts:
    ADD = np.array([[0.0, 0.1], [0.5, 0.6]])        # 纯加性：g=[0,0.1]，无 ε
    INTER = np.array([[0.1, -0.1], [-0.1, 0.1]])    # 纯交互：行/列均值全 0

    def test_indep_diffs_are_per_cls_cross_seed(self):
        assert probe.indep_diffs(self.ADD) == pytest.approx([0.1, 0.1])
        assert probe.indep_diffs(self.INTER) == pytest.approx([0.2, 0.2])

    def test_shared_rep_diffs_additive_cancels_exactly(self):
        # Δ_shared 两种子都是 −0.5 → D = 0（g 完全抵消，配对设计的理论最优情形）
        assert probe.shared_rep_diffs(self.ADD) == pytest.approx([0.0])

    def test_shared_rep_diffs_interaction_not_cancelled(self):
        # Δ_shared = [+0.2, −0.2] → |D| = 0.4（交互配对抵消不掉）
        assert probe.shared_rep_diffs(self.INTER) == pytest.approx([0.4])

    def test_counts_with_three_seeds(self):
        m = np.arange(6.0).reshape(2, 3)
        assert len(probe.indep_diffs(m)) == 6      # 2 cls × C(3,2)
        assert len(probe.shared_rep_diffs(m)) == 3  # 1 对 × C(3,2)

    def test_centered_devs_additive(self):
        # 中心 = −0.5：共享偏差恰 0；独立偏差 |±0.1|（g 项残留）
        sh, ind = probe.centered_devs(self.ADD)
        assert sh == pytest.approx([0.0, 0.0])
        assert sorted(ind) == pytest.approx([0.1, 0.1])


class TestP95:
    def test_matches_2c_nearest_rank(self):
        # 21 个值 0..20：索引 round(0.95×20)=19 → 值 19（取元素不插值）
        assert probe._p95(list(range(21))) == 19
        assert probe._p95([0.1, 0.3, 0.2]) == 0.3
        assert probe._p95([0.5]) == 0.5


class TestBandAndDecide:
    def test_band_boundaries(self):
        assert probe.band_of(0.50) == "green"
        assert probe.band_of(0.20) == "mid"
        assert probe.band_of(0.199) == "red"

    def test_decide_takes_conservative_band(self):
        assert probe.decide(0.70, 0.55) == "green"
        assert probe.decide(0.55, 0.45) == "mid"    # 跨档 → 低档
        assert probe.decide(0.30, 0.22) == "mid"
        assert probe.decide(0.25, 0.15) == "red"    # 跨档 → 低档


class TestDerived:
    def test_required_k(self):
        # (0.055/0.033)² = 2.778（outlet_y §5 同式，容 |Δ| p50=0.033）
        assert probe.required_k(0.055) == pytest.approx(2.778, rel=1e-3)

    def test_delta_from_p95_rounding_and_floor(self):
        # ceil(0.052/0.005)=11 → 0.055；量子下限 2×0.02=0.04 支配小 P95
        assert probe.delta_from_p95(0.052, [1 / 155]) == pytest.approx(0.055)
        assert probe.delta_from_p95(0.005, [0.02]) == pytest.approx(0.04)


class TestPairedDirectionStability:
    def test_all_same_sign_is_stable(self):
        m = np.array([[0.9, 0.88, 0.92], [0.5, 0.55, 0.45]])  # Δ 全正
        out = probe.paired_direction_stability(m, delta=0.1)
        assert out == {"n_pairs": 1, "stable_frac": 1.0}

    def test_sign_flip_is_unstable(self):
        m = np.array([[0.9, 0.5, 0.92], [0.5, 0.55, 0.45]])   # Δ=[+.4,−.05,+.47]
        out = probe.paired_direction_stability(m, delta=0.1)
        assert out == {"n_pairs": 1, "stable_frac": 0.0}

    def test_below_delta_excluded(self):
        m = np.array([[0.9, 0.88, 0.92], [0.5, 0.55, 0.45]])
        out = probe.paired_direction_stability(m, delta=0.5)
        assert out == {"n_pairs": 0, "stable_frac": None}
