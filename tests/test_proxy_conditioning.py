"""代理条件化（scripts/29_proxy_conditioning.py）纯逻辑测试。

协议 results/proxy_conditioning/PROTOCOL.md：两向残差化、交互份额杀门、
D1x 置换检验与判读带、增强参数口径、逐格代理确定性。期望值手工推导 /
植入结构验证。
"""
import importlib

import numpy as np
import pytest

px = importlib.import_module("scripts.29_proxy_conditioning")
dm = importlib.import_module("scripts.14_destruction_matrix")
gg = importlib.import_module("scripts.17_feasibility_loo_b")


# ---------------------------------------------------------------- 两向残差化
class TestTwoWayResiduals:
    def test_pure_additive_gives_zero(self):
        a = np.array([1.0, 2.0, 5.0])[:, None]
        b = np.array([0.5, -1.0, 3.0, 2.0])[None, :]
        R = px.two_way_residuals(a + b)
        np.testing.assert_allclose(R, 0.0, atol=1e-12)

    def test_row_and_col_sums_zero(self):
        rng = np.random.default_rng(0)
        R = px.two_way_residuals(rng.normal(size=(7, 5)))
        np.testing.assert_allclose(R.sum(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(R.sum(axis=1), 0.0, atol=1e-12)

    def test_invariant_to_main_effects(self):
        rng = np.random.default_rng(1)
        M = rng.normal(size=(6, 4))
        shift = rng.normal(size=(6, 1)) + rng.normal(size=(1, 4))
        np.testing.assert_allclose(px.two_way_residuals(M),
                                   px.two_way_residuals(M + shift), atol=1e-12)

    def test_manual_2x2(self):
        M = np.array([[1.0, 3.0], [5.0, 5.0]])
        # 行均 [2,5] 列均 [3,4] 总均 3.5 → r00 = 1-2-3+3.5 = -0.5
        np.testing.assert_allclose(px.two_way_residuals(M),
                                   [[-0.5, 0.5], [0.5, -0.5]], atol=1e-12)


# ---------------------------------------------------------------- 交互份额
class TestInteractionShare:
    def test_pure_additive_is_zero(self):
        a = np.arange(6.0)[:, None]
        b = np.arange(4.0)[None, :]
        assert px.interaction_share(a + b) == pytest.approx(0.0, abs=1e-12)

    def test_pure_interaction_is_one(self):
        M = np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert px.interaction_share(M) == pytest.approx(1.0, abs=1e-12)

    def test_matches_destruction_matrix_decomposition(self):
        rng = np.random.default_rng(2)
        # dm 口径 p[cls][ds]；本脚本矩阵是 (task, cls) → 传入时转置对齐
        n_cls, n_ds = 5, 8
        vals = rng.uniform(size=(n_cls, n_ds))
        p = {f"c{i}": {f"d{j}": float(vals[i, j]) for j in range(n_ds)}
             for i in range(n_cls)}
        assert px.interaction_share(vals.T) == pytest.approx(
            dm.ss_decomposition(p)["share_inter"], rel=1e-9)

    def test_constant_matrix_is_zero(self):
        assert px.interaction_share(np.full((4, 3), 2.0)) == pytest.approx(0.0)


# ---------------------------------------------------------------- 判读带
class TestVerdicts:
    def test_step0_bands(self):
        assert px.verdict_step0(0.0999) == "kill"
        assert px.verdict_step0(0.10) == "amber"
        assert px.verdict_step0(0.2499) == "amber"
        assert px.verdict_step0(0.25) == "green"

    def test_d1x_bands_green(self):
        assert px.verdict_d1x(0.01, 0.20, amber=False) == "established"
        assert px.verdict_d1x(0.01, 0.10, amber=False) == "edge"
        assert px.verdict_d1x(0.06, 0.30, amber=False) == "not_established"
        assert px.verdict_d1x(0.01, 0.04, amber=False) == "not_established"
        assert px.verdict_d1x(0.05, 0.30, amber=False) == "not_established"  # 0.05 不含

    def test_d1x_amber_rejects_edge(self):
        # 琥珀档不接受边缘行
        assert px.verdict_d1x(0.01, 0.10, amber=True) == "not_established"
        assert px.verdict_d1x(0.01, 0.20, amber=True) == "established"


# ---------------------------------------------------------------- D1x 检验
class TestResidualSpearmanPerm:
    def _planted(self, seed, alpha):
        rng = np.random.default_rng(seed)
        n_t, n_c = 80, 10
        task_x, cls_x = rng.normal(size=(n_t, 1)), rng.normal(size=(1, n_c))
        inter = rng.normal(size=(n_t, n_c))
        X = task_x * 3 + cls_x * 3 + inter
        Y = (rng.normal(size=(n_t, 1)) * 3 + rng.normal(size=(1, n_c)) * 3
             + alpha * inter + (1 - alpha) * rng.normal(size=(n_t, n_c)))
        return X, Y

    def test_detects_planted_interaction(self):
        X, Y = self._planted(0, alpha=0.6)
        rho, p = px.residual_spearman_perm(X, Y, n_perm=2000, seed=1)
        assert rho > 0.3 and p < 0.01

    def test_null_when_only_main_effects_shared(self):
        X, Y = self._planted(3, alpha=0.0)
        rho, p = px.residual_spearman_perm(X, Y, n_perm=2000, seed=1)
        assert p > 0.05

    def test_main_effect_trap_not_counted(self):
        """陷阱：y 的信号全在 CLS 主效应且与 x 的 CLS 主效应同源 → D1x 必须不显著。"""
        rng = np.random.default_rng(5)
        n_t, n_c = 80, 10
        cls_main = rng.normal(size=(1, n_c))
        X = rng.normal(size=(n_t, 1)) + cls_main * 5 + rng.normal(size=(n_t, n_c)) * 0.5
        Y = rng.normal(size=(n_t, 1)) + cls_main * 5 + rng.normal(size=(n_t, n_c)) * 0.5
        from scipy.stats import spearmanr
        pooled = spearmanr(X.ravel(), Y.ravel()).statistic
        rho, p = px.residual_spearman_perm(X, Y, n_perm=2000, seed=1)
        assert pooled > 0.8            # pooled 强相关（陷阱）
        assert p > 0.05                # 但交互残差无信号

    def test_deterministic(self):
        X, Y = self._planted(7, alpha=0.4)
        assert px.residual_spearman_perm(X, Y, n_perm=500, seed=9) == \
            px.residual_spearman_perm(X, Y, n_perm=500, seed=9)

    def test_one_sided_negative_direction_not_significant(self):
        X, Y = self._planted(11, alpha=0.6)
        rho, p = px.residual_spearman_perm(X, -Y, n_perm=2000, seed=1)
        assert rho < 0 and p > 0.5     # 单侧 H1: ρ>0


class TestWithinTaskPermutation:
    def test_permutation_preserves_task_blocks(self):
        rng = np.random.default_rng(0)
        R = rng.normal(size=(6, 5))
        P = px.permute_within_tasks(R, rng)
        # 每行是原行的置换 → 逐行排序后相同
        np.testing.assert_allclose(np.sort(P, axis=1), np.sort(R, axis=1))


# ---------------------------------------------------------------- 增强口径
class TestAugParams:
    def test_six_fields_exactly(self):
        from autocls.space.cls_space import CLSSpace
        cls = CLSSpace.from_yaml().sample_random(np.random.default_rng(0))
        p = px.aug_params(cls)
        assert set(p) == set(px.AUG_FIELDS) and len(p) == 6
        assert all(p[f] == getattr(cls, f) for f in px.AUG_FIELDS)

    def test_matches_cl_trainer_fields(self):
        ct = importlib.import_module("autocls.contrastive.cl_trainer")
        assert tuple(px.AUG_FIELDS) == tuple(ct._AUG_FIELDS)


# ---------------------------------------------------------------- 代理计算
class TestProxyCell:
    def _tiny(self, seed=0, n=48, T=32, C=2, k=3):
        rng = np.random.default_rng(seed)
        y = np.repeat(np.arange(k), n // k)
        X = (rng.normal(size=(n, T, C)) * 0.3
             + y[:, None, None].astype(float))       # 类别可分
        return X.astype(np.float32), y.astype(np.int64)

    def test_deterministic_same_seeds(self):
        X, y = self._tiny()
        from autocls.labeling import load_probe_cls
        fl = importlib.import_module("scripts.15_feasibility_labels")
        cls, key = load_probe_cls(fl.PROBE_CATALOG, 10)[0][0]
        a = px.proxy_cell(X, y, cls, tid="T_00", cls_key=key, n_reps=2, device="cpu")
        b = px.proxy_cell(X, y, cls, tid="T_00", cls_key=key, n_reps=2, device="cpu")
        assert a == b

    def test_returns_expected_keys(self):
        X, y = self._tiny()
        from autocls.labeling import load_probe_cls
        fl = importlib.import_module("scripts.15_feasibility_labels")
        cls, key = load_probe_cls(fl.PROBE_CATALOG, 10)[0][0]
        out = px.proxy_cell(X, y, cls, tid="T_00", cls_key=key, n_reps=2, device="cpu")
        for k in ("fid", "var", "d_g2", "d_g3", "d_g4", "g1_clean", "g1_aug", "dT"):
            assert k in out and np.isfinite(out[k])

    def test_identity_cls_gives_near_zero_fid(self):
        """全零增强 + norm=none → v1 ≡ x（恒等短路），fid 必须 ≈0。"""
        from autocls.space.cls_space import CLS
        X, y = self._tiny()
        ident = CLS(resize=0.0, rescale=0.0, jitter=0.0, point_mask=0.0,
                    freq_mask=0.0, crop=0.0, order=0, emb_jitter=0.0,
                    emb_mask=0.0, norm="none", instance=True, temporal=False,
                    cross_scale=False, kernel=None, pool=None, adj_neighbor=None,
                    loss_type="info_nce", sim_func="cosine", temperature=0.1)
        out = px.proxy_cell(X, y, ident, tid="T_00", cls_key="ident",
                            n_reps=1, device="cpu")
        assert abs(out["fid"]) < 1e-6 and abs(out["dT"]) < 1e-9

    def test_encoder_shared_across_cls_within_task(self):
        """同任务同 rep 的干净几何与 CLS 无关（配对口径）。"""
        X, y = self._tiny()
        from autocls.labeling import load_probe_cls
        fl = importlib.import_module("scripts.15_feasibility_labels")
        pairs = load_probe_cls(fl.PROBE_CATALOG, 10)[0]
        a = px.proxy_cell(X, y, pairs[0][0], tid="T_00", cls_key=pairs[0][1],
                          n_reps=2, device="cpu")
        b = px.proxy_cell(X, y, pairs[1][0], tid="T_00", cls_key=pairs[1][1],
                          n_reps=2, device="cpu")
        assert a["g1_clean"] == pytest.approx(b["g1_clean"])

    def test_geometry_reuses_scripts17(self):
        rng = np.random.default_rng(0)
        V, yy = rng.normal(size=(30, 8)), rng.integers(0, 3, 30)
        np.testing.assert_allclose(px.geom(V, yy),
                                   gg.class_geometry(gg.l2_normalize_rows(V), yy))


# ---------------------------------------------------------------- 面板
class TestRegistryResolution:
    def test_crop_pinned_datasets_use_crop256(self):
        regs = px.load_registries()
        for ds in ("PTB-XL", "NonInvasiveFetalECGThorax1"):
            reg, name = px.resolve_registry(ds, regs)
            assert name == "crop256" and reg.get(ds).crop_len == 256

    def test_source_preferred_and_candidate_fallback(self):
        regs = px.load_registries()
        assert px.resolve_registry("Handwriting", regs)[1] == "source"
        assert px.resolve_registry("UniMiB-SHAR", regs)[1] == "cand"
        assert px.resolve_registry("ChestAccel", regs)[1] == "cand"

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            px.resolve_registry("NoSuchDataset", px.load_registries())

    def test_registries_share_slice_params(self):
        regs = px.load_registries()
        seeds = {r.slice_seed for r in regs.values()}
        samples = {r.n_samples for r in regs.values()}
        assert len(seeds) == 1 and len(samples) == 1


class TestPanel:
    def test_panel_datasets_and_task_counts(self):
        assert px.N_CLS == 10
        assert set(px.PANEL_DS) == {"CharacterTrajectories", "UWaveGestureLibrary",
                                    "PhonemeSpectra", "Handwriting", "UniMiB-SHAR"}
        assert px.N_TASKS_PANEL == 8 and px.N_TASKS_OTHER == 2
