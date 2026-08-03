"""单卡阶段1 调度计划（configs/runtime.yaml 是分档/显存/线程的唯一真源）。铁律 5 先测后写。

依据 GATE_5090_REPORT §3 的 1×RTX 5090 实测：
- §3.1 `OMP_NUM_THREADS=1` 是硬性启动条件（默认 192 线程时 0.63×，比 4060 还慢）；
- §3.2 单卡 8 worker（15.2 条/分）> 4 卡各 1（11.6 条/分）——本负载 CPU-bound；
- §3.3 L=1152 单进程峰值 15.8 GB（losses.py:91 的 (n,n) 三元组矩阵），31.4 GiB
  装不下两份 → worker 数必须按 L 分档。

本模块是"分档"的可测核心：worker 分档表、显存峰值模型、并发准入、墙钟估算。
调度用**数据集级并行**（一个数据集一个进程），故分档表在此的语义是
"同一档的数据集最多几个同时在跑"；跨档回填由显存预算把关。
"""
import pytest

from autocls.cloud_plan import (
    admission_order,
    budget_gib,
    can_admit,
    estimate_item_seconds,
    estimate_peak_gib,
    load_runtime,
    simulate_wall_clock,
    stage1_datasets,
    tier_index,
    worker_env,
    workers_for,
)
from autocls.data.source_pool import load_registry


@pytest.fixture(scope="module")
def cfg():
    return load_runtime()


# ---------------------------------------------------------------------------
# 分档表：必须逐条复现 GATE_5090_REPORT §3.3 的表
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("l_enc,expected", [
    (50, 8),     # FingerMovements
    (51, 8),     # NATOPS（实测 8 worker GPU 91%、10.9 GB）
    (65, 8),     # ERing
    (144, 8),    # ArticularyWordRecognition
    (152, 8),    # Handwriting
    (182, 8),    # CharacterTrajectories
    (315, 2),    # UWaveGestureLibrary（保守，未实测）
    (400, 2),    # HandMovementDirection
    (405, 2),    # Heartbeat
    (512, 2),    # MotorImagery
    (896, 1),    # SelfRegulationSCP1
    (1024, 1),   # EthanolConcentration
    (1152, 1),   # SelfRegulationSCP2（实测 2 个必 OOM）
])
def test_worker_tiers_match_gate_report(cfg, l_enc, expected):
    assert workers_for(l_enc, cfg) == expected


def test_workers_for_is_monotone_non_increasing_in_length(cfg):
    lens = [50, 100, 200, 201, 400, 600, 601, 900, 1152, 4096]
    ws = [workers_for(x, cfg) for x in lens]
    assert ws == sorted(ws, reverse=True)


# ---------------------------------------------------------------------------
# 显存峰值模型：锚在实测两点上（启动时预检用它，不是拍脑袋）
# ---------------------------------------------------------------------------


def test_estimate_peak_gib_anchored_on_measurements(cfg):
    # §3.3 实测：L=1152 单进程峰值 15.8 GB
    assert estimate_peak_gib(1152, cfg) == pytest.approx(15.8, abs=0.3)
    # §3.2 实测：NATOPS(L=51) 8 worker 共 10.9 GB
    assert estimate_peak_gib(51, cfg) * 8 == pytest.approx(10.9, abs=1.0)


def test_estimate_peak_gib_increases_with_length(cfg):
    assert estimate_peak_gib(51, cfg) < estimate_peak_gib(512, cfg) < estimate_peak_gib(1152, cfg)


def test_budget_leaves_headroom_under_physical_vram(cfg):
    # 预算必须显著低于 31.4 GiB：模型是两点外推，留头寸给估不准
    assert budget_gib(cfg) < cfg["gpu"]["vram_gib"] * 0.85


def test_tier_table_is_consistent_with_vram_model(cfg):
    # 防漂移：有人改了预算/显存模型却没改分档表（或反之）时，长序列档必须仍然只允许 1 个
    for l_enc in (896, 1024, 1152):
        assert 2 * estimate_peak_gib(l_enc, cfg) > budget_gib(cfg)
        assert workers_for(l_enc, cfg) == 1


# ---------------------------------------------------------------------------
# 并发准入：显存预算 + 同档上限 + 全局进程上限
# ---------------------------------------------------------------------------


def test_two_long_datasets_never_share_the_card(cfg):
    ok, why = can_admit([1152], 1152, cfg)
    assert not ok and why
    ok, _ = can_admit([1152], 1024, cfg)   # 15.8+14.1 远超预算
    assert not ok


def test_short_dataset_backfills_alongside_a_long_one(cfg):
    # 长序列独占整张卡是纯浪费（单 worker GPU 仅 5.7%）——短数据集必须能回填
    ok, why = can_admit([1152], 65, cfg)
    assert ok, why


def test_admission_never_exceeds_budget_or_caps(cfg):
    running: list[int] = []
    for l_enc in (1152, 1024, 896, 512, 405, 400, 315, 182, 152, 144, 65, 50):
        ok, _ = can_admit(running, l_enc, cfg)
        if ok:
            running.append(l_enc)
    assert sum(estimate_peak_gib(x, cfg) for x in running) <= budget_gib(cfg) + 1e-9
    assert len(running) <= cfg["workers"]["max_concurrent"]
    for t in {tier_index(x, cfg) for x in running}:
        same = [x for x in running if tier_index(x, cfg) == t]
        assert len(same) <= workers_for(same[0], cfg)


def test_global_concurrency_cap_bites_even_when_vram_allows(cfg):
    cap = cfg["workers"]["max_concurrent"]
    running = [50] * cap
    assert sum(estimate_peak_gib(x, cfg) for x in running) < budget_gib(cfg)
    ok, why = can_admit(running, 50, cfg)
    assert not ok and why


def test_admission_order_is_longest_first(cfg):
    lens = [65, 1152, 315, 896]
    assert admission_order(lens, cfg) == [1152, 896, 315, 65]


# ---------------------------------------------------------------------------
# 线程环境：不依赖手动 export（默认 192 线程 = 慢 4.2×）
# ---------------------------------------------------------------------------


def test_worker_env_forces_single_thread(cfg):
    env = worker_env(cfg)
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"


# ---------------------------------------------------------------------------
# 墙钟估算：给启动前的 ETA，也是"长杆在哪"的依据
# ---------------------------------------------------------------------------


def test_estimate_item_seconds_anchored_on_measurements(cfg):
    assert estimate_item_seconds(51, cfg) == pytest.approx(12.6, abs=1.0)
    assert estimate_item_seconds(1152, cfg) == pytest.approx(123.6, abs=6.0)


def test_wall_clock_is_at_least_the_serial_long_chain(cfg):
    # 三个长数据集互斥 → 墙钟下界是它们串行之和（这就是阶段1 的长杆）
    jobs = {"SCP2": (1152, 80), "SCP1": (896, 80), "Ethanol": (1024, 80),
            "ERing": (65, 80), "NATOPS": (51, 80)}
    chain = sum(estimate_item_seconds(l, cfg) * n
                for l, n in (jobs["SCP2"], jobs["SCP1"], jobs["Ethanol"]))
    assert simulate_wall_clock(jobs, cfg) >= chain * 0.99


def test_wall_clock_beats_running_everything_serially(cfg):
    jobs = {"SCP2": (1152, 80), "ERing": (65, 80), "NATOPS": (51, 80),
            "AWR": (144, 80), "Handwriting": (152, 80)}
    serial = sum(estimate_item_seconds(l, cfg) * n for l, n in jobs.values())
    assert simulate_wall_clock(jobs, cfg) < serial


# ---------------------------------------------------------------------------
# 阶段1 跑哪些数据集（注册表唯一真源）
# ---------------------------------------------------------------------------


def test_stage1_datasets_are_untested_sources_plus_reserve():
    reg = load_registry()
    names = stage1_datasets(reg)
    assert set(names) == ({e.name for e in reg.source if not e.tested}
                          | {e.name for e in reg.reserve})
    assert not (set(names) & reg.names("target")), "目标池绝不入阶段1（铁律 6）"
    assert names == sorted(names, key=lambda n: -reg.get(n).l_enc)
