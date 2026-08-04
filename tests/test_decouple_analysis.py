"""解耦实验主分析（PREREGISTRATION.md §5）纯函数单测（铁律 5 先测后写）。

判定/统计核心逐字复用已测代码（gate_v2_eval / validate_degenerate_criterion），
这里只守本脚本新增的胶水规则：

1. 预测判定（§1 写死规则）：NATOPS 不过 → design_failure（阳性对照优先，
   不得把其余 fail 当预测应验）；NATOPS 过且三个全 fail → hit；否则 miss。
2. 子集采样确定性：同 seed 同子集序列（剂量-反应可复现）。
3. Wilson 90% 双侧区间基本性质。
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "decouple_analysis_mod", ROOT / "scripts" / "decouple_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["decouple_analysis_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()

AWR, CT, ER, NA = ("ArticularyWordRecognition", "CharacterTrajectories",
                   "ERing", "NATOPS")


def test_prediction_hit():
    v = M.prediction_verdict({AWR: False, CT: False, ER: False, NA: True})
    assert v["outcome"] == "hit"


def test_prediction_miss_when_extra_pass():
    v = M.prediction_verdict({AWR: True, CT: False, ER: False, NA: True})
    assert v["outcome"] == "miss"
    assert AWR in v["deviations"]


def test_prediction_design_failure_dominates():
    # NATOPS 阳性对照不过 → 整个实验判设计存疑，即使其余三个"如预测"fail
    v = M.prediction_verdict({AWR: False, CT: False, ER: False, NA: False})
    assert v["outcome"] == "design_failure"


def test_subset_sampling_deterministic():
    a = M.sample_subsets(list("abcdefghijklmnopqrst"), k=10, n=50, seed=20260804)
    b = M.sample_subsets(list("abcdefghijklmnopqrst"), k=10, n=50, seed=20260804)
    assert a == b
    assert all(len(s) == 10 and len(set(s)) == 10 for s in a)


def test_wilson_interval_contains_point():
    lo, hi = M.wilson_interval(4, 20, conf=0.90)
    assert 0.0 <= lo < 4 / 20 < hi <= 1.0
