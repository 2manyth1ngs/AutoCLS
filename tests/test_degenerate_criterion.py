"""退化 CLS 判定（PREREGISTRATION.md §3 + VALIDATION.md 修订 v2）的单测。

判据 v2 一句话：**退化 ⟺ 单侧配对 t 检验在 α=0.05 下拒绝 H0: mean(d) ≥ 0**
（训练产生统计显著的**破坏性**效应）。

v1（"不能拒绝无正向效应"）被 §3.1 先行验证证伪：4 个数据集的随机初始化基线
全部贴顶（0.90–0.99），正向增益原则上不可测，v1 把 9/10 条全判了退化。
修订只基于旧数据（VALIDATION.md 留痕），v2 语义下：

1. 显著破坏（负效应）→ 退化；明显正效应 → 常规。
2. **以 0 为中心的噪声 → 常规**（v1 是退化——语义翻转的关键测试）：贴顶数据集上
   "不动"与群体同位，不制造平凡配对，剂量-反应不该计它。
3. 有限格不足 → unknown；配对按 (dataset, task, cls, seed) 四元组严格对齐。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "validate_degen_mod", ROOT / "scripts" / "validate_degenerate_criterion.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["validate_degen_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


def test_destructive_effect_is_degenerate():
    rng = np.random.default_rng(2)
    diffs = list(-0.08 + 0.02 * rng.standard_normal(32))
    v = M.degenerate_verdict(diffs)
    assert v["status"] == "ok" and v["degenerate"] is True
    assert v["p_t"] < 0.05


def test_strong_positive_effect_is_not_degenerate():
    rng = np.random.default_rng(0)
    diffs = list(0.10 + 0.02 * rng.standard_normal(32))
    v = M.degenerate_verdict(diffs)
    assert v["status"] == "ok" and v["degenerate"] is False


def test_zero_centered_noise_is_not_degenerate():
    # v2 语义翻转的关键：贴顶数据集上"不动"= 与群体同位，不制造平凡配对
    rng = np.random.default_rng(1)
    diffs = list(0.0 + 0.02 * rng.standard_normal(32))
    v = M.degenerate_verdict(diffs)
    assert v["status"] == "ok" and v["degenerate"] is False


def test_insufficient_cells_is_unknown():
    v = M.degenerate_verdict([-0.1] * 10, min_cells=24)
    assert v["status"] == "unknown" and v["degenerate"] is None


def test_nan_cells_do_not_count():
    diffs = [-0.1] * 23 + [float("nan")] * 9
    v = M.degenerate_verdict(diffs, min_cells=24)
    assert v["status"] == "unknown"


def test_paired_diffs_align_on_full_key():
    rp = {("D", "t0", "c1", 0): 0.80, ("D", "t0", "c1", 1): 0.70,
          ("D", "t1", "c1", 0): 0.90}
    base = {("D", "t0", "c1", 0): 0.60, ("D", "t0", "c1", 1): 0.65,
            ("D", "t1", "c1", 2): 0.50}          # t1 基线种子错位 → 不得配对
    diffs = M.paired_diffs(rp, base, cls_key="c1")
    assert sorted(np.round(diffs, 6)) == [0.05, 0.20]


def test_paired_diffs_only_selected_cls():
    rp = {("D", "t0", "a", 0): 0.9, ("D", "t0", "b", 0): 0.8}
    base = {("D", "t0", "a", 0): 0.5, ("D", "t0", "b", 0): 0.5}
    assert M.paired_diffs(rp, base, cls_key="a") == [0.4]
