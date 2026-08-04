"""下游评估协议探针（autocls/downstream/protocols.py + autocls/protocol_probe.py）的测试。

铁律 5 先测后写。被测核心：

- **协议 A 不是第二份 fork**：`evaluate_protocol("A", feats...)` 必须与既有
  `evaluate_classification`（= AutoCL 附录 B.1 的实现，也是全部历史 R' 缓存的口径）
  逐位同值。这是整个探针"可比"的前提——若 A 复算不出旧值，B/C 与 A 的差就没意义。
- **特征通路等价于 evaluate_rprime**：`compute_task_features` + 协议 A 的 r_prime
  必须等于 `evaluate_rprime` 的 r_prime（同种子）。dump-then-evaluate 与
  train-and-evaluate 之间不得有静默差异。
- **特征可复现**：同 (task, cls, iters, weight_seed) 两次跑逐位相同 —— 这是
  "既有 R' 无法换协议重算、但可以按种子重跑重现"这一成本结论的前提（PROTOCOL §6.1）。
- few-shot 取样语义（每类 ≤k、类样本不足时取全部、同种子可复现、不同重复不同子集）。
- 判据阈值（PROTOCOL.md §4.3 写死）与判定表（§4.4）的边界行为。
"""
from __future__ import annotations

import numpy as np
import pytest

from autocls.downstream.protocols import (
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOL_B_PRIME,
    PROTOCOL_NAMES,
    FEWSHOT_K,
    FEWSHOT_REPEATS,
    evaluate_protocol,
    fewshot_indices,
)
from autocls.protocol_probe import (
    G1_BASELINE_MAX,
    G2_MIN_ABOVE,
    G3_MIN_ABOVE,
    G3_MIN_BEST_GAIN,
    gate1_verdict,
    gate2_verdict,
    overall_verdict,
)


# --------------------------------------------------------------------------
# 固定的小特征夹具：所有协议消费同一份特征（探针的唯一变量是特征之后的分类器）
# --------------------------------------------------------------------------

@pytest.fixture
def feats():
    """(N, d) 线性可分性中等的三类特征 + 一条 eval 侧。"""
    rng = np.random.default_rng(0)
    d, per = 16, 30
    centers = rng.standard_normal((3, d)).astype(np.float32) * 2.0
    def make(n):
        y = np.repeat(np.arange(3), n)
        X = centers[y] + rng.standard_normal((3 * n, d)).astype(np.float32)
        return X.astype(np.float32), y
    Xtr, ytr = make(per)
    Xev, yev = make(12)
    return Xtr, ytr, Xev, yev


# ---- 协议 A 必须与既有实现逐位同值（不 fork）----

def test_protocol_a_matches_existing_fit_svm(feats):
    """A = linear_eval.fit_svm 的原样复用：同特征同种子 → acc/f1/超参逐位相同。"""
    from sklearn.metrics import accuracy_score, f1_score

    from autocls.downstream.linear_eval import fit_svm

    Xtr, ytr, Xev, yev = feats
    ref_clf = fit_svm(Xtr, ytr, seed=7)
    ref_pred = ref_clf.predict(Xev)

    got = evaluate_protocol(PROTOCOL_A, Xtr, ytr, Xev, yev, seed=7)
    assert got["r_prime"] == accuracy_score(yev, ref_pred)
    assert got["f1"] == f1_score(yev, ref_pred, average="macro")
    assert got["hp"]["C"] == float(ref_clf.C)
    assert got["r_prime"] == got["acc"]


def test_protocol_a_equals_evaluate_classification_end_to_end():
    """dump 特征后跑 A == 直接跑 evaluate_classification（同 encoder、同种子）。

    这条把"表征提取 / 下游评估解耦"钉成回归测试：闸门 1/2 的全部结论都建立在
    "A 那一列复算得出旧协议的值"之上。
    """
    import torch

    from autocls.downstream.linear_eval import encode_features, evaluate_classification
    from autocls.encoders.registry import build_encoder

    rng = np.random.default_rng(3)
    n, t, c = 45, 32, 2
    y = np.repeat(np.arange(3), n // 3)
    X = (rng.standard_normal((n, t, c)).astype(np.float32)
         + y[:, None, None].astype(np.float32))
    Xtr, ytr, Xev, yev = X[:30], y[:30], X[30:], y[30:]

    torch.manual_seed(123)
    enc = build_encoder(input_dims=c)
    ref = evaluate_classification(enc, Xtr, ytr, Xev, yev, device="cpu", seed=5)

    ftr = encode_features(enc, Xtr, device="cpu")
    fev = encode_features(enc, Xev, device="cpu")
    got = evaluate_protocol(PROTOCOL_A, ftr, ytr, fev, yev, seed=5)

    assert got["acc"] == ref["acc"]
    assert got["f1"] == ref["f1"]
    assert got["hp"]["C"] == ref["svm_c"]


def test_compute_task_features_reproduces_evaluate_rprime(make_cls):
    """compute_task_features + 协议 A 的 R' == evaluate_rprime 的 R'（同种子）。

    PROTOCOL §6.1 的成本结论依赖这条：既有缓存换不了协议，但按 weight_seed 重跑
    能重现同一份权重 → 同一份特征。这里用 iters>0 覆盖训练路径（不只 iters=0）。
    """
    from autocls.data.task_slicer import SlicedTask
    from autocls.labeling import evaluate_rprime
    from autocls.protocol_probe import compute_task_features

    rng = np.random.default_rng(11)
    n, t, c = 48, 40, 2
    y = np.repeat(np.arange(3), n // 3)
    X = (rng.standard_normal((n, t, c)).astype(np.float32)
         + 0.8 * y[:, None, None].astype(np.float32))
    task = SlicedTask(task_id="synthetic_feat_00", pool="synthetic", classes=(0, 1, 2),
                      channels=(0, 1), sample_indices=tuple(range(n)))
    cls_cfg = make_cls(jitter=0.2, crop=0.5)
    kw = dict(weight_seed=4242, split_seed=99, iters=3, device="cpu")

    ref = evaluate_rprime(task, cls_cfg, X, y, **kw)
    bundle = compute_task_features(task, cls_cfg, X, y, **kw)
    got = evaluate_protocol(PROTOCOL_A, bundle["train_feats"], bundle["y_train"],
                            bundle["eval_feats"], bundle["y_eval"], seed=kw["weight_seed"])

    assert ref["error"] is None
    assert got["r_prime"] == ref["r_prime"]
    assert (bundle["n_train"], bundle["n_eval"]) == (ref["n_train"], ref["n_eval"])


def test_compute_task_features_is_reproducible(make_cls):
    from autocls.data.task_slicer import SlicedTask
    from autocls.protocol_probe import compute_task_features

    rng = np.random.default_rng(5)
    n, t, c = 36, 32, 2
    y = np.repeat(np.arange(3), n // 3)
    X = rng.standard_normal((n, t, c)).astype(np.float32)
    task = SlicedTask(task_id="synthetic_rep_00", pool="synthetic", classes=(0, 1, 2),
                      channels=(0, 1), sample_indices=tuple(range(n)))
    kw = dict(weight_seed=7, split_seed=13, iters=2, device="cpu")

    a = compute_task_features(task, make_cls(jitter=0.2), X, y, **kw)
    b = compute_task_features(task, make_cls(jitter=0.2), X, y, **kw)
    assert np.array_equal(a["train_feats"], b["train_feats"])
    assert np.array_equal(a["eval_feats"], b["eval_feats"])
    assert np.array_equal(a["y_eval"], b["y_eval"])


# ---- 协议 B / B′：定义与超参口径（PROTOCOL.md §3 写死）----

def test_protocol_b_is_scaled_l2_logistic_regression(feats):
    Xtr, ytr, Xev, yev = feats
    got = evaluate_protocol(PROTOCOL_B, Xtr, ytr, Xev, yev, seed=0)
    assert got["classifier"] == "logreg_l2"
    assert got["scaler"] == "standard"
    assert got["hp"]["C"] in (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2)
    assert 0.0 <= got["r_prime"] <= 1.0


def test_protocol_b_prime_is_linear_svm_without_scaler(feats):
    """B′ 只改"核"这一个旋钮，用来把 B 的效应归因到具体改动上。"""
    Xtr, ytr, Xev, yev = feats
    got = evaluate_protocol(PROTOCOL_B_PRIME, Xtr, ytr, Xev, yev, seed=0)
    assert got["classifier"] == "svm_linear"
    assert got["scaler"] is None
    assert got["hp"]["C"] in (1.0, 10.0, 100.0)


def test_protocols_are_deterministic(feats):
    Xtr, ytr, Xev, yev = feats
    for name in PROTOCOL_NAMES:
        a = evaluate_protocol(name, Xtr, ytr, Xev, yev, seed=1)
        b = evaluate_protocol(name, Xtr, ytr, Xev, yev, seed=1)
        assert a["r_prime"] == b["r_prime"], f"{name} 不确定"


def test_protocol_names_are_the_preregistered_set():
    assert PROTOCOL_NAMES == ("A", "B", "B_prime", "C_k5", "C_k10", "C_k20")
    assert FEWSHOT_K == (5, 10, 20) and FEWSHOT_REPEATS == 5


def test_unknown_protocol_raises(feats):
    Xtr, ytr, Xev, yev = feats
    with pytest.raises(ValueError, match="未登记"):
        evaluate_protocol("D", Xtr, ytr, Xev, yev, seed=0)


# ---- 协议 C：few-shot 取样语义 ----

def test_fewshot_takes_at_most_k_per_class():
    y = np.repeat(np.arange(3), 20)
    idx = fewshot_indices(y, k=5, seed=0)
    counts = np.bincount(y[idx], minlength=3)
    assert list(counts) == [5, 5, 5]


def test_fewshot_takes_all_when_class_smaller_than_k():
    """类样本不足 k 时取全部——不得丢类（丢类会让 eval 侧出现训练时没见过的标签）。"""
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2])
    idx = fewshot_indices(y, k=5, seed=0)
    counts = np.bincount(y[idx], minlength=3)
    assert list(counts) == [2, 5, 3]
    assert set(np.unique(y[idx])) == {0, 1, 2}


def test_fewshot_is_reproducible_and_repeats_differ():
    y = np.repeat(np.arange(4), 25)
    assert np.array_equal(fewshot_indices(y, k=10, seed=42),
                          fewshot_indices(y, k=10, seed=42))
    assert not np.array_equal(fewshot_indices(y, k=10, seed=42),
                              fewshot_indices(y, k=10, seed=43))


def test_protocol_c_reports_per_repeat_spread(feats):
    """C 报 5 次抽样的均值与 std——单次 few-shot 的方差很大，只报点估会误导。"""
    Xtr, ytr, Xev, yev = feats
    got = evaluate_protocol("C_k5", Xtr, ytr, Xev, yev, seed=0)
    assert got["classifier"] == "svm_rbf"        # C 只改样本量，分类器沿用 A
    assert got["k"] == 5
    assert len(got["r_prime_repeats"]) == FEWSHOT_REPEATS
    assert got["r_prime"] == pytest.approx(float(np.mean(got["r_prime_repeats"])))
    assert got["n_train_used"] <= len(ytr)


# ---- 判据（PROTOCOL.md §4.3/§4.4 写死的阈值与判定表）----

def test_thresholds_are_the_preregistered_values():
    assert (G1_BASELINE_MAX, G2_MIN_ABOVE, G3_MIN_ABOVE, G3_MIN_BEST_GAIN) == \
        (0.70, 6, 12, 0.10)


@pytest.mark.parametrize("awr,passed", [
    (0.50, True), (0.6999, True), (0.70, False), (0.85, False), (0.99, False)])
def test_gate1_verdict_is_strict_less_than(awr, passed):
    """G1 = AWR 新基线 < 0.70（严格小于，边界 0.70 判不过）。"""
    assert gate1_verdict(awr)["passed"] is passed


@pytest.mark.parametrize("above,below,passed", [
    (15, 4, True),      # 旧协议 Handwriting 的量级
    (6, 6, True),       # 边界：恰好 6/20 且 ↑ == ↓
    (6, 7, False),      # ↑ < ↓：只是让训练更容易破坏表征
    (5, 0, False),      # 不足 6/20
    (0, 16, False),     # 旧协议 NATOPS 的量级
])
def test_gate2_verdict_needs_both_count_and_direction(above, below, passed):
    got = gate2_verdict({"AWR": (above, below), "NATOPS": (above, below)})
    assert got["passed"] is passed


def test_gate2_requires_every_dataset_to_pass():
    """G2 要 AWR 与 NATOPS **都**满足——任一不过即不过。"""
    got = gate2_verdict({"AWR": (15, 2), "NATOPS": (1, 9)})
    assert got["passed"] is False
    assert got["failed"] == ["NATOPS"]


@pytest.mark.parametrize("above,best_gain,passed", [
    (15, 0.397, True),   # 旧协议实测
    (12, 0.10, True),    # 边界：恰好达标
    (11, 0.40, False),   # 显著性不够
    (14, 0.099, False),  # 幅度不够
])
def test_gate3_control_not_broken(above, best_gain, passed):
    from autocls.protocol_probe import gate3_verdict
    assert gate3_verdict(above, best_gain)["passed"] is passed


@pytest.mark.parametrize("g1,g2,g3,verdict", [
    (True, True, True, "protocol_works"),
    (True, False, True, "false_fix"),
    (True, False, False, "false_fix"),      # G1 过 G2 不过 → 假解决（G3 不改判）
    (False, True, True, "no_effect"),       # G1 不过 → 无效（后两闸不改判）
    (False, False, False, "no_effect"),
    (True, True, False, "broke_control"),
])
def test_overall_verdict_table(g1, g2, g3, verdict):
    assert overall_verdict(g1, g2, g3) == verdict


# ---- 铁律 6 ----

def test_target_pool_never_enters_protocol_probe_scope():
    from autocls.data.source_pool import ROLE_TARGET, load_registry
    from autocls.protocol_probe import GATE2A_DATASETS, GATE2B_DATASETS, gate1_scope

    registry = load_registry()
    targets = set(registry.names(ROLE_TARGET))
    assert targets
    assert not (set(gate1_scope(registry)) & targets), "目标池混入闸门1 范围（铁律 6）"
    assert not (set(GATE2A_DATASETS + GATE2B_DATASETS) & targets)


def test_gate2_split_matches_protocol():
    """MotorImagery 推迟到 2b，且不参与判定（PROTOCOL §4.1）。"""
    from autocls.protocol_probe import GATE2A_DATASETS, GATE2B_DATASETS

    assert GATE2A_DATASETS == ("ArticularyWordRecognition", "NATOPS", "Handwriting")
    assert GATE2B_DATASETS == ("MotorImagery",)
