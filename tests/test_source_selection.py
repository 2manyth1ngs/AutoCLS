"""MX-2a E 选源（scripts/19_source_selection.py）纯逻辑测试。

协议 results/mx2a_source_selection/PROTOCOL.md：数据集距离（无泄漏标准化）、
top-k 选择、核权重、随机对照确定性、PairNet 收敛/确定性。
"""
import importlib

import numpy as np
import pytest

ss = importlib.import_module("scripts.19_source_selection")


class TestDatasetDistance:
    def _feats(self):
        rng = np.random.default_rng(0)
        # 4 个训练数据集 × 2 任务 + 1 个留出 × 2 任务，dim 3
        tf = {f"D{i}_{j:02d}": rng.normal(i, 0.1, 3)
              for i in range(4) for j in range(2)}
        tf.update({f"Z_{j:02d}": rng.normal(1.05, 0.1, 3) for j in range(2)})
        return tf

    def test_nearest_and_no_heldout_in_stats(self):
        tf = self._feats()
        train_ds = {f"D{i}": [f"D{i}_00", f"D{i}_01"] for i in range(4)}
        d = ss.dataset_distances(tf, train_ds, ["Z_00", "Z_01"])
        assert set(d) == set(train_ds)
        assert min(d, key=d.get) == "D1"          # Z 造在 D1 附近
        # 泄漏契约：标准化统计量只来自训练任务——把留出特征整体平移，
        # 各训练数据集到留出的距离应同量平移方向一致（序数不因 Z 参与 mu/sd 而改）
        tf2 = {k: (v + 100.0 if k.startswith("Z") else v) for k, v in tf.items()}
        d2 = ss.dataset_distances(tf2, train_ds, ["Z_00", "Z_01"])
        assert all(d2[k] > d[k] for k in d)       # 全部变远，无一被拉近
        # 确定性
        d3 = ss.dataset_distances(tf, train_ds, ["Z_00", "Z_01"])
        assert d == d3

    def test_topk_and_kernel(self):
        d = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
        assert ss.top_k_sources(d, k=2) == ["A", "B"]
        w = ss.kernel_weights(d, sigma=2.0)
        assert w["A"] > w["B"] > w["C"] > w["D"] > 0
        assert w["A"] == pytest.approx(np.exp(-1.0 / 8.0))

    def test_random_subset_deterministic(self):
        pool = [f"D{i}" for i in range(15)]
        s1 = ss.random_sources(pool, k=5, seed=3)
        s2 = ss.random_sources(pool, k=5, seed=3)
        assert s1 == s2 and len(set(s1)) == 5 and set(s1) <= set(pool)
        assert ss.random_sources(pool, k=5, seed=4) != s1


class TestPairNet:
    def _toy(self):
        # y=1 ⇔ a 的第 0 维更大：可分配对任务
        rng = np.random.default_rng(0)
        Xa = rng.normal(size=(120, 4)).astype(np.float32)
        Xb = rng.normal(size=(120, 4)).astype(np.float32)
        y = (Xa[:, 0] > Xb[:, 0]).astype(np.float32)
        return Xa, Xb, y

    def test_deterministic_and_converges(self):
        Xa, Xb, y = self._toy()
        w = np.ones(len(y))
        m1 = ss.train_pairnet(Xa, Xb, y, w, seed=1, epochs=300)
        m2 = ss.train_pairnet(Xa, Xb, y, w, seed=1, epochs=300)
        p1, p2 = ss.pairnet_proba(m1, Xa, Xb), ss.pairnet_proba(m2, Xa, Xb)
        np.testing.assert_array_equal(p1, p2)
        assert ((p1 > 0.5) == y.astype(bool)).mean() > 0.95

    def test_sample_weights_shift_model(self):
        Xa, Xb, y = self._toy()
        w1 = np.ones(len(y))
        w2 = np.linspace(0.1, 3.0, len(y))
        p1 = ss.pairnet_proba(ss.train_pairnet(Xa, Xb, y, w1, seed=1, epochs=50),
                              Xa, Xb)
        p2 = ss.pairnet_proba(ss.train_pairnet(Xa, Xb, y, w2, seed=1, epochs=50),
                              Xa, Xb)
        assert not np.array_equal(p1, p2)


class TestVerdictD:
    def test_bands(self):
        assert ss.verdict_mx2a(0.01, 0.01, 0.5, 0.5) == "established"
        assert ss.verdict_mx2a(0.01, 0.30, 0.5, 0.5) == "fewer_sources_effect"
        assert ss.verdict_mx2a(0.10, 0.5, 0.5, 0.5) == "edge"
        assert ss.verdict_mx2a(0.50, 0.5, 0.01, 0.5) == "edge"    # W2 过触发边缘
        assert ss.verdict_mx2a(0.50, 0.5, 0.5, 0.01) == "edge"    # W1pg 过触发边缘
        assert ss.verdict_mx2a(0.50, 0.5, 0.5, 0.5) == "not_established"


class TestPairsFromLabels:
    def test_direction_and_delta(self):
        # classification：大者优；|Δ|<δ 弃样
        rows = [
            {"dataset": "D", "task_id": "D_00", "cls_key": "a", "r_prime": 0.90},
            {"dataset": "D", "task_id": "D_00", "cls_key": "b", "r_prime": 0.60},
            {"dataset": "D", "task_id": "D_00", "cls_key": "c", "r_prime": 0.895},
        ]
        pairs, stats = ss.pairs_for_task(rows, delta=0.05)
        got = {(p["a"], p["b"]): p["y"] for p in pairs}
        assert got[("a", "b")] == 1 and got[("b", "c")] == 0
        assert ("a", "c") not in got and stats["n_delta_dropped"] == 1
