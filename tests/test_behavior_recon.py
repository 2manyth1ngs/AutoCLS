"""行为条件化侦察（scripts/28_behavior_recon.py）纯逻辑测试。

协议 results/behavior_recon/PROTOCOL.md：三态口径一致性、杀门三档、
四臂正确性与折隔离、行为距离只用观测列、钉源过滤、确定性。
期望值手工推导。
"""
import importlib
import json

import numpy as np
import pytest

br = importlib.import_module("scripts.28_behavior_recon")
fl = importlib.import_module("scripts.15_feasibility_labels")
gp = importlib.import_module("scripts.22_gate2_pilot")


# ---------------------------------------------------------------- 三态口径
class TestTriStateCaliber:
    def test_boundaries_are_inert(self):
        # 协议 §3：destroy 严格 <mean−band、improve 严格 >mean+band
        assert fl.tri_state(0.4, 0.5, 0.1) == "inert"     # 恰在下界
        assert fl.tri_state(0.6, 0.5, 0.1) == "inert"     # 恰在上界
        assert fl.tri_state(0.39999, 0.5, 0.1) == "destroy"
        assert fl.tri_state(0.60001, 0.5, 0.1) == "improve"

    def test_states_order_matches_15(self):
        assert br.STATES == fl.LABELS == ("improve", "inert", "destroy")


# ---------------------------------------------------------------- 第 0 步
class TestCohenKappa:
    def test_perfect(self):
        assert br.cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(1.0)

    def test_manual_value(self):
        # po=0.75; pa=(.5,.5) pb=(.25,.75); pe=.5 → κ=0.5
        assert br.cohen_kappa([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.5)

    def test_constant_agreement(self):
        assert br.cohen_kappa([1, 1], [1, 1]) == pytest.approx(1.0)

    def test_constant_disagreement(self):
        assert br.cohen_kappa([1, 1], [0, 0]) == pytest.approx(0.0)


class TestVerdictStep0:
    def test_bands(self):
        # 杀：M1<0.35 或 M2<0.45
        assert br.verdict_step0(0.34, 0.90) == "kill"
        assert br.verdict_step0(0.90, 0.4499) == "kill"
        # 边界含：0.35/0.45 不杀 → 琥珀
        assert br.verdict_step0(0.35, 0.45) == "amber"
        assert br.verdict_step0(0.49, 0.90) == "amber"
        assert br.verdict_step0(0.50, 0.5999) == "amber"
        # 绿：M1≥0.50 且 M2≥0.60
        assert br.verdict_step0(0.50, 0.60) == "green"
        assert br.verdict_step0(0.90, 0.90) == "green"


# ---------------------------------------------------------------- 观测列抽样
class TestDrawCols:
    def test_deterministic_and_sorted(self):
        c1 = br.draw_cols(10, 5, seed=7)
        c2 = br.draw_cols(10, 5, seed=7)
        assert c1 == c2 == sorted(c1) and len(set(c1)) == 5
        assert all(0 <= c < 10 for c in c1)

    def test_varies_with_seed(self):
        draws = {tuple(br.draw_cols(10, 3, seed=s)) for s in range(8)}
        assert len(draws) > 1


# ---------------------------------------------------------------- 率向量
class TestRateVector:
    def test_two_task_grid(self):
        # 2 任务 × 3 CLS：状态码 → {0,.5,1} 网格
        q = br.rate_vector([np.array([0, 1, 2]), np.array([0, 2, 2])])
        np.testing.assert_allclose(q[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(q[1], [0.0, 0.5, 0.5])
        np.testing.assert_allclose(q[2], [0.0, 0.0, 1.0])


# ---------------------------------------------------------------- 四臂
def _mk_rows(rng, n, n_cls=10):
    Q = rng.dirichlet([1, 1, 1], size=(n, n_cls))
    return [(f"D{i:02d}", Q[i]) for i in range(n)]


class TestArms:
    def test_column_prior_is_mean(self):
        rng = np.random.default_rng(0)
        rows = _mk_rows(rng, 4)
        prior = br.arm_column_prior(np.stack([q for _n, q in rows]))
        np.testing.assert_allclose(prior, np.stack([q for _n, q in rows]).mean(0))

    def test_level_shift_recovers_constant_shift(self):
        # held = prior + 常数态移 → 平移臂误差 ≈0、列先验臂 >0
        prior = np.tile([0.2, 0.5, 0.3], (10, 1))
        held = np.tile([0.4, 0.3, 0.3], (10, 1))
        J = [0, 1, 2]
        unobs = [c for c in range(10) if c not in J]
        pred_b = br.arm_level_shift(prior, held[J], J)
        assert br.masked_brier(pred_b, held, unobs) == pytest.approx(0.0, abs=1e-12)
        assert br.masked_brier(prior, held, unobs) > 0.01

    def test_knn_recovers_planted_triplet(self):
        rng = np.random.default_rng(1)
        truth = np.tile([0.1, 0.2, 0.7], (10, 1))
        rows = [("T1", truth.copy()), ("T2", truth.copy()), ("T3", truth.copy())]
        rows += _mk_rows(rng, 5)          # 远处噪声行
        J = [0, 3, 6]
        pred = br.arm_knn(rows, truth[J], J, k=3)
        unobs = [c for c in range(10) if c not in J]
        assert br.masked_brier(pred, truth, unobs) == pytest.approx(0.0, abs=1e-9)

    def test_knn_tie_break_by_name_deterministic(self):
        q = np.tile([1 / 3, 1 / 3, 1 / 3], (10, 1))
        rows = [("B", q.copy()), ("A", q.copy() * 0 + [[0.5, 0.25, 0.25]] * 10),
                ("C", q.copy())]
        # 距离并列时按名字典序：A 的距离更远（不同行），B/C 并列近
        pred1 = br.arm_knn(rows, q[[0, 1]], [0, 1], k=1)
        pred2 = br.arm_knn(rows, q[[0, 1]], [0, 1], k=1)
        np.testing.assert_allclose(pred1, pred2)

    def test_svd_recovers_rank1_structure(self):
        rng = np.random.default_rng(2)
        mu = np.tile([0.3, 0.4, 0.3], (10, 1)).reshape(-1)
        # 每个 CLS 三元组内和为 0 的方向：行仍在单纯形平面上，clip/归一化为恒等
        v = rng.normal(size=(10, 3))
        v -= v.mean(axis=1, keepdims=True)
        v = v.reshape(-1)
        v /= np.linalg.norm(v)
        rows = np.stack([(mu + s * 0.15 * v) for s in rng.normal(size=24)])
        rows = rows.reshape(24, 10, 3)
        held = (mu + 0.15 * 1.3 * v).reshape(10, 3)
        J = [0, 2, 4, 6, 8]
        pred = br.arm_svd_complete(rows, held[J], J, rank=1, ridge=1e-3)
        unobs = [c for c in range(10) if c not in J]
        err_svd = br.masked_brier(pred, held, unobs)
        err_prior = br.masked_brier(br.arm_column_prior(rows), held, unobs)
        assert err_svd < 1e-3 and err_svd < err_prior / 3


class TestFoldIsolation:
    def test_arms_blind_to_unobserved_held_cells(self):
        # 扰动留出行未观测列 → 四臂预测逐位不变
        rng = np.random.default_rng(3)
        rows = _mk_rows(rng, 8)
        held = rng.dirichlet([1, 1, 1], size=10)
        J = [1, 4, 7]
        p1 = br.predict_dataset_arms(rows, held, J, k=3, rank=2)
        held2 = held.copy()
        for c in range(10):
            if c not in J:
                held2[c] = rng.dirichlet([1, 1, 1])
        p2 = br.predict_dataset_arms(rows, held2, J, k=3, rank=2)
        for arm in p1:
            np.testing.assert_array_equal(p1[arm], p2[arm])

    def test_arm_names(self):
        rng = np.random.default_rng(4)
        rows = _mk_rows(rng, 6)
        p = br.predict_dataset_arms(rows, rows[0][1], [0, 1, 2], k=3, rank=2)
        assert set(p) == {"column_prior", "level_shift", "knn", "svd"}


# ---------------------------------------------------------------- 行为距离 / Brier
class TestBehaviorDistance:
    def test_only_observed_cols(self):
        a = np.array([0, 1, 2, 0, 1])
        b = np.array([0, 2, 2, 1, 0])
        assert br.behavior_distance(a, b, [0, 1, 2]) == pytest.approx(1 / 3)
        # 扰动非观测列 → 距离不变
        b2 = b.copy()
        b2[3] = 2
        assert br.behavior_distance(a, b2, [0, 1, 2]) == \
            br.behavior_distance(a, b, [0, 1, 2])


class TestMaskedBrier:
    def test_manual_value(self):
        pred = np.array([[1.0, 0, 0], [0, 0.5, 0.5], [0, 0, 1.0]])
        target = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 0, 1.0]])
        # 列1: 0+.25+.25=0.5; 列2: 0 → unobs=[1,2] 均值 0.25
        assert br.masked_brier(pred, target, [1, 2]) == pytest.approx(0.25)

    def test_full_mask_onehot_equals_profile_brier(self):
        rng = np.random.default_rng(5)
        pred = rng.dirichlet([1, 1, 1], size=6)
        codes = rng.integers(0, 3, 6)
        onehot = np.eye(3)[codes]
        assert br.masked_brier(pred, onehot, list(range(6))) == \
            pytest.approx(gp.profile_brier(pred, codes))


# ---------------------------------------------------------------- 装配
class TestPinFilter:
    def test_main_drops_crop_datasets(self):
        rows = [{"dataset": "PTB-XL"}, {"dataset": "Handwriting"},
                {"dataset": "NonInvasiveFetalECGThorax1"}]
        out = br.pin_filter(rows, "main")
        assert [r["dataset"] for r in out] == ["Handwriting"]

    def test_crop_keeps_only_crop_datasets(self):
        rows = [{"dataset": "PTB-XL"}, {"dataset": "Handwriting"}]
        out = br.pin_filter(rows, "crop")
        assert [r["dataset"] for r in out] == ["PTB-XL"]


class TestCollectTrainedPerSeed:
    def _write(self, tmp_path, rows):
        p = tmp_path / "cache.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return p

    def test_filters(self, tmp_path):
        bands = {"DS_00": (0.5, 0.1)}
        classes = {"DS_00": [0, 1]}
        shared = {"K1"}
        rows = [
            {"task_id": "DS_00", "cls_key": "K1", "seed": 2, "iters": 290,
             "classes": [0, 1], "r_prime": 0.7, "split_seed": 999},  # 收（split 不查）
            {"task_id": "DS_00", "cls_key": "K1", "seed": 3, "iters": 60,
             "classes": [0, 1], "r_prime": 0.7},                      # 拒 iters
            {"task_id": "DS_00", "cls_key": "KX", "seed": 1, "iters": 290,
             "classes": [0, 1], "r_prime": 0.7},                      # 拒 not_shared
            {"task_id": "DS_00", "cls_key": "K1", "seed": 1, "iters": 290,
             "classes": [0, 9], "r_prime": 0.7},                      # 拒 classes
            {"task_id": "XX_00", "cls_key": "K1", "seed": 1, "iters": 290,
             "classes": [0, 1], "r_prime": 0.7},                      # 拒 no_baseline
        ]
        out, rej = br.collect_trained_per_seed(
            self._write(tmp_path, rows), shared, bands, classes)
        assert set(out) == {("DS_00", "K1", 2)}
        assert rej["iters"] == 1 and rej["not_shared"] == 1
        assert rej["classes"] == 1 and rej["no_baseline"] == 1


# ---------------------------------------------------------------- 面板池
class TestPanelPools:
    def test_um_supplementary_not_primary(self):
        assert br.SUPP_DS == "UniMiB-SHAR"
        assert br.SUPP_DS not in br.PRIMARY_DS
        assert set(br.PRIMARY_DS) == {"CharacterTrajectories",
                                      "UWaveGestureLibrary",
                                      "PhonemeSpectra", "Handwriting"}

    def test_excluded_datasets(self):
        assert "PenDigits" in br.EXCLUDE_DS
        assert br.PIN_CROP == {"PTB-XL", "NonInvasiveFetalECGThorax1"}
