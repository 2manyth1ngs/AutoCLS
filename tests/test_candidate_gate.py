"""换源池调研的准入判据测试（`results/candidate_survey/PROTOCOL.md` §2/§4；铁律 5 先测后写）。

两组被测：
1. `prescreen`（§2 初筛硬条件 H1/H2/H3）——L≥50、N≥1000、K≥5，K=2–4 判 reserve。
2. `classify_verdict`（§4.2 判据 v2）——把零假设诊断**写进判据**。上一版
   （`classify_quadrant`）只看 `best_gain ≥ δ`，把 MotorImagery 那类"低基线但训不动"
   放行了，诊断是事后才补的。

判据 v2 的阈值用 UEA 16 个**已测**数据集回标（先验数据，非本轮新数据）：必须复现
baseline_probe REPORT §6.1 的"站得住"分类，并把边缘的 SelfRegulationSCP1 判否
（从严——上一轮的教训全部来自把边缘案例放行）。
"""
from __future__ import annotations

import pytest

from autocls.baseline_probe import (
    MIN_N_EVAL,
    MIN_N_SAMPLES,
    MIN_N_CLASSES,
    MIN_SEQ_LEN,
    T3_ABOVE_FRACTION,
    T3_ABOVE_OVER_BELOW,
    classify_verdict,
    prescreen,
    propose_slice,
)


# ---- §5 切片规划：每候选能切多少任务、每任务 n_eval 够不够 ----

def test_min_n_eval_matches_delta_noise_derivation():
    # R' 噪声 ≈ 0.5/√n_eval ≤ 典型 δ 0.05 → n_eval ≥ 100
    assert MIN_N_EVAL == 100


def test_propose_slice_class_range_is_a_subset_band():
    p = propose_slice(n_samples=30000, n_classes=12, n_channels=6)
    lo, hi = p["n_classes_range"]
    assert 2 <= lo < hi <= 12


def test_propose_slice_task_count_capped_by_class_combos():
    # K=5 时 (2,4) 档的组合数有限，planned_tasks 不能超过它
    p = propose_slice(n_samples=30000, n_classes=5, n_channels=3)
    assert p["planned_tasks"] <= p["n_combos"]


def test_propose_slice_reports_n_eval_and_flags_thin_tasks():
    # N=1000、K=26、档约 (10,21) → 任务样本 ≈1000×0.6=600，n_eval≈180 够
    ok = propose_slice(n_samples=1000, n_classes=26, n_channels=3)
    assert ok["n_eval_est"] >= MIN_N_EVAL and ok["n_eval_ok"]
    # N=300 的同形数据集必然不够
    thin = propose_slice(n_samples=300, n_classes=26, n_channels=3)
    assert thin["n_eval_est"] < MIN_N_EVAL and not thin["n_eval_ok"]


def test_propose_slice_caps_n_eval_by_sampling_limit():
    # 任务采样上限 1024 → n_eval 封顶 ≈307，再大的 N 也不会更高
    huge = propose_slice(n_samples=500000, n_classes=10, n_channels=6)
    assert huge["n_eval_est"] <= 308


# ---- §2 初筛硬条件 ----

def test_prescreen_thresholds_match_protocol():
    assert MIN_SEQ_LEN == 50        # H1 固定 encoder 适用边界
    assert MIN_N_SAMPLES == 1000    # H2 由 n_eval≥100 反推
    assert MIN_N_CLASSES == 5       # H3 类别子集是多样性主力


def test_prescreen_accepts_uci_har_profile():
    # 目标画像（历史 M1 记录，非本轮加载）：N=10299 L=128 C=9 K=6
    ok, role, why = prescreen(n_samples=10299, seq_len=128, n_classes=6)
    assert ok and role == "source" and why == []


def test_prescreen_rejects_short_sequences():
    ok, _role, why = prescreen(n_samples=10992, seq_len=8, n_classes=10)   # PenDigits
    assert not ok and any("L" in w for w in why)


def test_prescreen_rejects_small_n():
    ok, _role, why = prescreen(n_samples=575, seq_len=144, n_classes=25)   # AWR
    assert not ok and any("N" in w for w in why)


def test_prescreen_handwriting_passes_at_the_boundary():
    # N=1000 恰在阈值上——H2 由 n_eval≥100 推导，Handwriting 实际任务 n_eval≈127
    ok, role, _why = prescreen(n_samples=1000, seq_len=152, n_classes=26)
    assert ok and role == "source"


def test_prescreen_few_classes_becomes_reserve_not_reject():
    # K=2 的候选（如 Daphnet）：能测基线、不能当源池主力
    ok, role, why = prescreen(n_samples=17090, seq_len=128, n_classes=2)
    assert ok and role == "reserve" and any("K" in w for w in why)


def test_prescreen_reports_all_failed_conditions():
    ok, _role, why = prescreen(n_samples=30, seq_len=8, n_classes=3)
    assert not ok and len(why) >= 2      # N 与 L 都不过，两条都要报


# ---- §4.2 判据 v2 ----

def test_verdict_thresholds_match_protocol():
    assert T3_ABOVE_FRACTION == 0.5
    assert T3_ABOVE_OVER_BELOW == 2.0


def test_verdict_ceiling_short_circuits_before_training():
    # T1 不过：贴顶直接出局，不看增益（省算力，且加难档/换协议两条补救路已被证伪）
    v = classify_verdict(baseline=0.972, best_gain=0.013, n_above=0, n_below=16, n_trained=20)
    assert v == "ceiling"


def test_verdict_ceiling_even_with_large_gain():
    v = classify_verdict(baseline=0.95, best_gain=0.40, n_above=20, n_below=0, n_trained=20)
    assert v == "ceiling"


def test_verdict_unknown_without_training_data():
    v = classify_verdict(baseline=0.55, best_gain=None, n_above=0, n_below=0, n_trained=0)
    assert v == "unknown"


def test_verdict_unknown_when_fewer_rows_than_protocol_requires():
    """测不满 PROTOCOL §4.1 的 20 条就不出判定——这是执行协议，不是改判据。

    真实场景：PTB-XL 在步骤3 首次启动时只跑出 2 条就因 L² 代价被停机。若对 n=2
    出判定，T3 的"超出 ≥ 半数"退化成"超出 ≥1 条"，一条运气好的 CLS 就能判 usable。
    """
    v = classify_verdict(baseline=0.695, best_gain=0.30, n_above=2, n_below=0, n_trained=2)
    assert v == "unknown"


def test_verdict_issues_verdict_at_exactly_the_protocol_row_count():
    v = classify_verdict(baseline=0.555, best_gain=0.397, n_above=15, n_below=4, n_trained=20)
    assert v == "usable"


def test_verdict_inert_when_gain_below_delta():
    # T2 不过：MotorImagery 型——基线≈随机猜测，headroom 名义 0.49 但一分没用上
    v = classify_verdict(baseline=0.511, best_gain=0.049, n_above=4, n_below=1, n_trained=20)
    assert v == "inert"


def test_verdict_inert_when_significance_too_thin():
    # T3 不过：HandMovementDirection——表面增益 +0.169 最大，但只有 1/20 跑出自身噪声带
    v = classify_verdict(baseline=0.443, best_gain=0.169, n_above=1, n_below=0, n_trained=20)
    assert v == "inert"


def test_verdict_inert_when_breakage_outweighs_gains():
    # T3 的第二半：PhonemeSpectra 2↑/7↓——best_gain 由"10 条取最大"的选择偏差撑着
    v = classify_verdict(baseline=0.541, best_gain=0.070, n_above=2, n_below=7, n_trained=20)
    assert v == "inert"


# 用 UEA 16 个已测数据集回标（baseline_probe REPORT §2.2 的实测计数）
_UEA_REGRESSION = [
    #  名称,                   baseline, best_gain, ↑,  ↓, 总, 期望
    ("Handwriting",              0.555,   0.397,   15,  4, 20, "usable"),
    ("UWaveGestureLibrary",      0.879,   0.077,   10,  3, 20, "usable"),
    ("SelfRegulationSCP1",       0.808,   0.092,    5,  2, 20, "inert"),   # 边缘，从严判否
    ("PhonemeSpectra",           0.541,   0.070,    2,  7, 20, "inert"),
    ("Cricket",                  0.876,   0.093,    1,  8, 20, "inert"),
    ("HandMovementDirection",    0.443,   0.169,    1,  0, 20, "inert"),
    ("FingerMovements",          0.557,   0.055,    2,  4, 20, "inert"),
    ("MotorImagery",             0.511,   0.049,    4,  1, 20, "inert"),
    ("SelfRegulationSCP2",       0.547,   0.044,    3,  7, 20, "inert"),
    ("EthanolConcentration",     0.508,   0.033,    0,  8, 20, "inert"),
    ("Heartbeat",                0.720,   0.048,    7,  0, 20, "inert"),
    ("SpokenArabicDigits",       0.912,   0.053,    8,  4, 20, "ceiling"),
    ("ERing",                    0.948,   0.048,    7,  4, 20, "ceiling"),
    ("NATOPS",                   0.972,   0.013,    0, 16, 20, "ceiling"),
    ("CharacterTrajectories",    0.981,   0.016,   10,  4, 20, "ceiling"),
    ("ArticularyWordRecognition", 0.990,  0.014,    0,  6, 20, "ceiling"),
]


@pytest.mark.parametrize("name,base,gain,up,dn,tot,expect", _UEA_REGRESSION)
def test_verdict_reproduces_uea_verdicts(name, base, gain, up, dn, tot, expect):
    assert classify_verdict(baseline=base, best_gain=gain, n_above=up,
                            n_below=dn, n_trained=tot) == expect, name


def test_candidate_registry_loads_npz_source(tmp_path):
    """候选走 `source: npz` 进同一个 load_source_dataset 入口。

    为什么不另开一条加载路径：baseline_probe 的口径一致性靠"同一个入口 + 同一个
    slice_source_tasks"保证（PROTOCOL §1）。候选若走第二份加载逻辑，新旧基线数字
    就不可比，而本轮的全部结论都建立在与 UEA 那 16 个数字可比之上。
    """
    import numpy as np

    from autocls.data.candidates import save_candidate
    from autocls.data.source_pool import load_registry, load_source_dataset

    X = np.random.default_rng(0).normal(size=(20, 128, 3)).astype(np.float32)
    y = np.array([0, 1] * 10, dtype=np.int64)
    save_candidate("Toy", X, y, {}, root=tmp_path)

    reg_path = tmp_path / "candidate_pool.yaml"
    reg_path.write_text(
        "encode_length_floor: 50\n"
        "slice: {seed: 0, n_samples: 1024, min_samples: 32}\n"
        "source_pool:\n"
        "  - {name: Toy, source: npz, domain: HAR, n_channels: 3, n_classes: 2,\n"
        "     l_enc: 128, crop_len: null, n_classes_range: [2, 2], planned_tasks: 2}\n",
        encoding="utf-8",
    )
    registry = load_registry(reg_path)
    X2, y2 = load_source_dataset("Toy", registry, candidate_root=tmp_path)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)


def test_candidate_registry_still_blocks_target_pool(tmp_path):
    """铁律 6 的防护不因新增 npz 源而失效——目标池即便声明成 npz 也必须 fail-fast。"""
    from autocls.data.source_pool import (
        TargetPoolLeakageError,
        load_registry,
        load_source_dataset,
    )

    reg_path = tmp_path / "reg.yaml"
    reg_path.write_text(
        "target_pool:\n"
        "  - {name: UCI-HAR, source: npz, domain: HAR}\n",
        encoding="utf-8",
    )
    with pytest.raises(TargetPoolLeakageError):
        load_source_dataset("UCI-HAR", load_registry(reg_path), candidate_root=tmp_path)


def test_verdict_only_handwriting_and_uwave_are_usable_on_uea():
    """回标的总结论：16 个 UEA 数据集里判 usable 的恰好是那 2 个。

    这条断言把 §4.2 的立场钉死——若将来有人放宽阈值让第三个 UEA 数据集变 usable，
    这个测试会红，逼他重新论证而不是静默改判。
    """
    usable = [n for n, b, g, u, d, t, _e in _UEA_REGRESSION
              if classify_verdict(baseline=b, best_gain=g, n_above=u,
                                  n_below=d, n_trained=t) == "usable"]
    assert usable == ["Handwriting", "UWaveGestureLibrary"]
