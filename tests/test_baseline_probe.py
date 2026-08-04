"""随机初始化基线探针（autocls/baseline_probe.py + scripts/baseline_probe.py）的测试。

铁律 5 先测后写。被测核心：
- **iters=0 时 CLS 完全不参与**：这是整个探针成立的前提（基线是"任务+划分+初始化"
  的属性，与策略无关）。断言两条截然不同的 CLS 在同种子下给出逐位相同的 R'；
- 种子口径：split 恒为 canonical（与阶段1 canonical R' 同一划分 → 基线是其严格
  零训练对照），weight 随 s 变（5 个随机初始化）；
- 三档/四象限判据（PROTOCOL.md §3 写死的阈值）；
- 统计汇总：per-task 均值/标准差、headroom、chance、realized/best gain 配对。
"""
from __future__ import annotations

import numpy as np
import pytest

from autocls.baseline_probe import (
    BASELINE_ITERS,
    BEST_GAIN_FLOOR,
    CEILING_BASELINE,
    OPEN_BASELINE,
    baseline_seed_specs,
    classify_quadrant,
    classify_tier,
    inert_cls,
    summarize_dataset,
    summarize_task,
)
from autocls.labeling import cls_key, run_seed_for


# ---- 种子口径 ----

def test_split_seed_is_canonical_for_all_seeds():
    """split 恒为阶段1 canonical 划分：基线与 R' 同一 train/eval 切分才可配对相减。"""
    tid = "NATOPS_00"
    canonical_split = run_seed_for(tid, "train_eval_split")
    for s in range(5):
        _, split_seed = baseline_seed_specs(tid, s)
        assert split_seed == canonical_split


def test_weight_seeds_differ_across_seeds_and_tasks():
    w = [baseline_seed_specs("NATOPS_00", s)[0] for s in range(5)]
    assert len(set(w)) == 5, "5 个随机初始化必须互不相同"
    assert baseline_seed_specs("NATOPS_00", 0)[0] != baseline_seed_specs("ERing_00", 0)[0]


def test_baseline_iters_is_zero():
    assert BASELINE_ITERS == 0


# ---- iters=0 → CLS 无关（探针成立的前提，铁律 2 的 max_iters 显式传参路径）----

def test_cls_is_inert_at_zero_iters(make_cls):
    """两条差异极大的 CLS，同 weight/split 种子、iters=0 → R' 必须逐位相同。

    这不是实现细节而是判据前提：若不成立，"基线"就不是任务属性、整个探针无意义。
    """
    from autocls.data.task_slicer import SlicedTask
    from autocls.labeling import evaluate_rprime

    rng = np.random.default_rng(0)
    n, t, c = 48, 64, 3
    y = np.repeat(np.arange(3), n // 3)
    X = (rng.standard_normal((n, t, c)).astype(np.float32)
         + y[:, None, None].astype(np.float32))
    task = SlicedTask(task_id="synthetic_00", pool="synthetic", classes=(0, 1, 2),
                      channels=(0, 1, 2), sample_indices=tuple(range(n)))

    mild = make_cls(jitter=0.3, crop=0.5)
    harsh = make_cls(jitter=1.0, crop=0.2, point_mask=0.9, freq_mask=0.5,
                     resize=0.5, emb_mask=0.5, temperature=1.0)
    assert cls_key(mild) != cls_key(harsh)

    kw = dict(weight_seed=1234, split_seed=5678, iters=BASELINE_ITERS,
              eval_ratio=0.3, batch_size=128, lr=1e-3, device="cpu")
    r_mild = evaluate_rprime(task, mild, X, y, **kw)
    r_harsh = evaluate_rprime(task, harsh, X, y, **kw)

    assert r_mild["error"] is None and r_harsh["error"] is None
    assert r_mild["r_prime"] == r_harsh["r_prime"]
    assert (r_mild["n_train"], r_mild["n_eval"]) == (r_harsh["n_train"], r_harsh["n_eval"])


def test_joblib_backend_does_not_change_rprime(make_cls):
    """threading 后端与默认 loky 逐位同值。

    Windows 上 fit_svm 的 GridSearchCV(n_jobs=-1) 每条要 spawn 一个 loky 进程池
    （实测 8–15s，占单条基线耗时 ~99%），故 runner 默认用 threading。这是纯性能
    开关、不得改动数值——linear_eval 的既有注释断言"n_jobs 不影响取值"，此处把它
    钉成回归测试（否则整份基线与阶段1 的 R' 就不在同一口径上）。
    """
    from joblib import parallel_backend

    from autocls.data.task_slicer import SlicedTask
    from autocls.labeling import evaluate_rprime

    rng = np.random.default_rng(7)
    n, t, c = 60, 48, 2
    y = np.repeat(np.arange(3), n // 3)
    X = (rng.standard_normal((n, t, c)).astype(np.float32)
         + 0.5 * y[:, None, None].astype(np.float32))
    task = SlicedTask(task_id="synthetic_backend_00", pool="synthetic",
                      classes=(0, 1, 2), channels=(0, 1), sample_indices=tuple(range(n)))
    kw = dict(weight_seed=99, split_seed=11, iters=BASELINE_ITERS, device="cpu")

    loky = evaluate_rprime(task, inert_cls(), X, y, **kw)
    with parallel_backend("threading", n_jobs=1):
        threaded = evaluate_rprime(task, inert_cls(), X, y, **kw)

    assert loky["r_prime"] == threaded["r_prime"]
    assert loky["f1"] == threaded["f1"]
    assert loky["svm_c"] == threaded["svm_c"]


def test_inert_cls_is_a_valid_space_member():
    """占位 CLS 必须是 space.yaml 合法取值（虽不参与计算，仍要能被序列化进缓存）。"""
    from autocls.space.cls_space import CLSSpace

    c = inert_cls()
    assert CLSSpace.from_yaml().validate(c)
    assert len(cls_key(c)) == 12


# ---- 三档 / 四象限判据（PROTOCOL.md §3 写死阈值）----

@pytest.mark.parametrize("baseline,tier", [
    (0.99, "ceiling"), (0.95, "ceiling"), (0.9001, "ceiling"),
    (0.90, "mid"), (0.80, "mid"), (0.70, "mid"),
    (0.6999, "open"), (0.51, "open"), (0.10, "open"),
])
def test_classify_tier_thresholds(baseline, tier):
    assert classify_tier(baseline) == tier


def test_tier_thresholds_are_the_prereg_values():
    assert (CEILING_BASELINE, OPEN_BASELINE, BEST_GAIN_FLOOR) == (0.90, 0.70, 0.05)


@pytest.mark.parametrize("baseline,best_gain,quad", [
    (0.98, 0.01, "ceiling"),     # 贴顶：gain 大小无关，headroom 就那么点
    (0.98, 0.30, "ceiling"),
    (0.70, 0.12, "usable"),      # 有空间且用上了
    (0.51, 0.20, "usable"),
    (0.51, 0.004, "inert"),      # 有空间但固定 encoder 用不上（伪离顶）
    (0.75, 0.049, "inert"),      # 增益低于 δ 门槛
])
def test_classify_quadrant(baseline, best_gain, quad):
    assert classify_quadrant(baseline, best_gain) == quad


def test_quadrant_unknown_when_no_trained_reference():
    """无阶段1 配对 R'（如探针骨干未进 stage1 缓存）→ 不臆断，判 unknown。"""
    assert classify_quadrant(0.55, None) == "unknown"
    assert classify_quadrant(0.98, None) == "ceiling"  # 贴顶不依赖 gain，可直接判


# ---- 统计汇总 ----

def test_summarize_task_stats_and_chance():
    rows = [{"r_prime": v} for v in (0.80, 0.84, 0.82, 0.86, 0.78)]
    st = summarize_task(rows, n_classes=4, trained=[0.90, 0.95, 0.88])
    assert st["n_seeds"] == 5
    assert st["baseline_mean"] == pytest.approx(0.82)
    assert st["baseline_std"] == pytest.approx(np.std([0.80, 0.84, 0.82, 0.86, 0.78], ddof=1))
    assert st["baseline_min"] == 0.78 and st["baseline_max"] == 0.86
    assert st["headroom"] == pytest.approx(0.18)
    assert st["chance"] == pytest.approx(0.25)
    assert st["lift_over_chance"] == pytest.approx(0.57)
    assert st["trained_mean"] == pytest.approx(0.91)
    assert st["realized_gain"] == pytest.approx(0.09)
    assert st["best_gain"] == pytest.approx(0.13)   # 0.95 − 0.82


def test_summarize_task_counts_cls_outside_random_init_spread():
    """best_gain 是"10 条取最大"，天然向上偏——补一个零假设诊断计数。

    随机初始化本身就有 σ 的抖动；只有 R' 落在 mean±2σ 之外的 CLS 才算真的
    beat/break 了随机表征。σ 用 ACC 量子 1/n_eval 兜底（有的任务 5 个种子同分，
    σ=0 会让"超出 2σ"退化成"超出均值"）。
    """
    rows = [{"r_prime": v, "n_eval": 100} for v in (0.50, 0.52, 0.51, 0.49, 0.48)]
    # baseline mean=0.50, std≈0.0158 → 2σ≈0.0316 → 上界 0.532、下界 0.468
    st = summarize_task(rows, n_classes=2, trained=[0.60, 0.53, 0.50, 0.40, 0.44])
    assert st["n_trained_above_null"] == 1      # 0.60 > 0.532
    assert st["n_trained_below_null"] == 2      # 0.40, 0.44 < 0.468
    assert st["null_band"] == pytest.approx(2 * st["baseline_std"])


def test_null_band_falls_back_to_acc_quantum_when_seeds_agree():
    rows = [{"r_prime": 0.80, "n_eval": 50} for _ in range(5)]   # σ=0，量子 1/50=0.02
    st = summarize_task(rows, n_classes=2, trained=[0.83, 0.81, 0.75])
    assert st["null_band"] == pytest.approx(0.04)   # 2 × max(0, 1/50)
    assert st["n_trained_above_null"] == 0          # 0.83 未超 0.84
    assert st["n_trained_below_null"] == 1          # 0.75 < 0.76


def test_summarize_task_without_trained_reference():
    st = summarize_task([{"r_prime": 0.5}], n_classes=2, trained=[])
    assert st["trained_mean"] is None
    assert st["realized_gain"] is None and st["best_gain"] is None


def test_summarize_task_ignores_failed_records():
    """R'=None（训练/评估失败）不得混进均值——None 是结论，不是 0 分。"""
    st = summarize_task([{"r_prime": 0.8}, {"r_prime": None}, {"r_prime": 0.9}],
                        n_classes=2, trained=[])
    assert st["n_seeds"] == 2 and st["n_failed"] == 1
    assert st["baseline_mean"] == pytest.approx(0.85)


def test_summarize_dataset_reports_task_level_distribution():
    tasks = {
        "D_00": {"baseline_mean": 0.60, "headroom": 0.40, "best_gain": 0.10,
                 "n_classes": 4, "chance": 0.25},
        "D_01": {"baseline_mean": 0.96, "headroom": 0.04, "best_gain": 0.02,
                 "n_classes": 4, "chance": 0.25},
        "D_02": {"baseline_mean": 0.75, "headroom": 0.25, "best_gain": 0.06,
                 "n_classes": 4, "chance": 0.25},
    }
    ds = summarize_dataset(tasks)
    assert ds["n_tasks"] == 3
    assert ds["baseline_mean"] == pytest.approx(0.77)
    assert (ds["task_min"], ds["task_median"], ds["task_max"]) == (0.60, 0.75, 0.96)
    assert ds["headroom"] == pytest.approx(0.23)
    assert ds["best_gain_max"] == pytest.approx(0.10)
    assert ds["tier"] == "mid"          # 数据集档按 per-task 均值的均值
    assert ds["quadrant"] == "usable"   # best_gain_max 0.10 ≥ 0.05
    assert ds["tier_counts"] == {"ceiling": 1, "mid": 1, "open": 1}


# ---- 铁律 6：目标池绝不进探针 ----

def test_target_pool_excluded_from_probe_scope():
    from autocls.baseline_probe import probe_scope
    from autocls.data.source_pool import ROLE_TARGET, load_registry

    registry = load_registry()
    scope = probe_scope(registry)
    targets = set(registry.names(ROLE_TARGET))
    assert targets and not (set(scope) & targets), "目标池混入探针范围（铁律 6）"
    assert len(scope) == 16, "13 源 + 3 reserve"
