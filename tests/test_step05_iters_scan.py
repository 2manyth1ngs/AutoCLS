"""Step 0.5 iters 扫描（scripts/12_step05_iters_floor_scan.py）纯逻辑测试。

期望值手工推导：少数票提取（含平票规则）、可及率、2×2 判读四格与
regime 全崩回退（协议 §4）。
"""
import importlib

scan = importlib.import_module("scripts.12_step05_iters_floor_scan")


def _v(tid, a, b, sign, gap):
    return {"task_id": tid, "a": a, "b": b, "sign": sign, "gap": gap}


class TestMinorityGaps:
    def test_hand_computed(self):
        # 对 (a,b)：3 票 [1,1,0] → 多数 1，少数 = sign0 那票（gap 0.03）
        votes = [_v("t1", "a", "b", 1, 0.10), _v("t2", "a", "b", 1, 0.08),
                 _v("t3", "a", "b", 0, 0.03)]
        assert scan.minority_gaps(votes) == [0.03]

    def test_tie_majority_is_one(self):
        # 平票 1:1 → 多数记 sign=1（写死规则），少数 = sign0
        votes = [_v("t1", "a", "b", 1, 0.10), _v("t2", "a", "b", 0, 0.05)]
        assert scan.minority_gaps(votes) == [0.05]

    def test_unanimous_no_minority(self):
        votes = [_v("t1", "a", "b", 1, 0.1), _v("t2", "a", "b", 1, 0.2)]
        assert scan.minority_gaps(votes) == []


class TestReachFraction:
    def test_hand(self):
        frac, n = scan.reach_fraction([0.01, 0.03, 0.05], 0.03)
        assert n == 3 and abs(frac - 2 / 3) < 1e-12

    def test_empty_or_none_delta(self):
        assert scan.reach_fraction([], 0.03) == (None, 0)
        assert scan.reach_fraction([0.1], None) == (None, 1)


class TestScanVerdict:
    def _arm(self, delta, share, reach):
        return {"delta_nd": delta, "nd_share": share, "reach": reach}

    def test_go_iters(self):
        out = scan.scan_verdict({30: self._arm(0.045, 0.9, 0.0),
                                 240: self._arm(0.020, 0.8, 0.15)})
        assert out["cell"] == "GO_iters" and out["best_arm"] == 240

    def test_noise_down_signal_shrinks(self):
        out = scan.scan_verdict({30: self._arm(0.045, 0.9, 0.0),
                                 240: self._arm(0.020, 0.8, 0.05)})
        assert out["cell"] == "noise_down_signal_shrinks"

    def test_floor_rigid(self):
        out = scan.scan_verdict({30: self._arm(0.045, 0.9, 0.0),
                                 240: self._arm(0.040, 0.8, 0.50)})
        assert out["cell"] == "floor_rigid"   # N− 时 S 不参与判定

    def test_regime_collapse_falls_back_to_60(self):
        # 全臂 nd_share < 0.60 → 强制以 60 臂为准
        out = scan.scan_verdict({60: self._arm(0.020, 0.3, 0.20),
                                 240: self._arm(0.015, 0.2, 0.30)})
        assert out["forced_60"] and out["best_arm"] == 60
        assert out["cell"] == "GO_iters"

    def test_undecidable(self):
        out = scan.scan_verdict({60: {"delta_nd": None, "nd_share": 0.0},
                                 240: self._arm(0.015, 0.2, 0.30)})
        assert out["cell"] == "undecidable"

    def test_boundary_exact(self):
        # δ=0.025 恰在 N+ 线上（≤ 判定）；reach=0.10 恰在 S+ 线上（≥ 判定）
        out = scan.scan_verdict({120: self._arm(0.025, 0.7, 0.10)})
        assert out["cell"] == "GO_iters"
