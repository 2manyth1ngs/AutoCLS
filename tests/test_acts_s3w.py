"""WP17-ACTS S3-W（results/acts/s3w/PROTOCOL.md）——先于实现。覆盖：保守项的基准率 / 配对 / 刷新日程、
扩充池 regret 与高估、α 选择、读数码、命令行，以及 --cons 0 与 S3-R′ 逐位相同、--cons > 0 能在 CPU 上走通。"""
from __future__ import annotations

import importlib
import types

import numpy as np


def _r():
    return importlib.import_module("scripts.105_acts_s3r")


def _w():
    return importlib.import_module("scripts.109_acts_s3w")


def test_base_rates():
    r = _r()
    L = np.array([0.9, 0.5, 0.2, 0.6, 0.4, 0.5, 0.1])          # n_sh = 2：下标 2..6 是 random
    use = np.ones(7, bool); use[6] = False                   # 验证留出不进基准率
    t = r.base_rates(L, use, n_sh=2)
    # c = 0（0.9）：random {0.2, 0.6, 0.4, 0.5} 无一更高 → 0
    assert t[0] == 0.0
    # c = 1（0.5）：0.6 更高、0.5 平局 → (1 + 0.5) / 4
    assert abs(t[1] - 1.5 / 4) < 1e-12
    # c = 3（0.6，本身是 random）：不和自己比 → {0.2, 0.4, 0.5} 无一更高 → 0
    assert t[3] == 0.0
    assert np.isnan(r.base_rates(L, np.r_[True, True, np.zeros(5, bool)], n_sh=2)).all()     # 没有 random → nan


def test_cons_pairs():
    r = _r()
    a, b, y, adv_first = r.cons_pairs(n_adv=2, refs=np.array([4, 7]), t=np.array([0.0] * 4 + [0.25, 0, 0, 0.6]), offset=10)
    # 对抗配置在扩展下标 10、11；每对 (x, c) 与 (c, x) 各一次
    assert len(a) == 8
    for x in (10, 11):
        for c, tc in ((4, 0.25), (7, 0.6)):
            i = np.where((a == x) & (b == c))[0]; j = np.where((a == c) & (b == x))[0]
            assert len(i) == 1 and len(j) == 1 and abs(y[i[0]] - tc) < 1e-12 and abs(y[j[0]] - (1 - tc)) < 1e-12
            assert adv_first[i[0]] and not adv_first[j[0]]


def test_cons_schedule():
    r = _r()
    tr = list(range(10))
    assert r.cons_schedule(tr, 0, 3, seed=1) == tr                       # epoch 0 全部
    seen = set()
    for ep in (1, 2, 3):
        s = r.cons_schedule(tr, ep, 3, seed=1)
        assert 3 <= len(s) <= 4 and set(s) <= set(tr)
        seen |= set(s)
    assert seen == set(tr)                                                # 三个 epoch 轮换覆盖全部
    assert r.cons_schedule(tr, 4, 3, seed=1) == r.cons_schedule(tr, 1, 3, seed=1)   # 周期 3


def test_regret_ext():
    w = _w()
    s0 = np.array([0.5, 0.9, 0.7, 0.8])
    P = np.full((4, 4), 0.5)
    P[0, 1:] = 0.9; P[1:, 0] = 0.1                                         # 模型最看好 0 号（真值最差）
    P[2, 3] = 0.8; P[3, 2] = 0.2
    P[3, 1] = 0.8; P[1, 3] = 0.2
    # 胜场 0:3、2:1、3:1、1:0 → 循环赛前 3 = {0, 2, 3}，其中真值最好 0.8 → regret 0.1
    assert abs(w.regret_ext(P, ["a", "b", "c", "d"], s0, k=3) - 0.1) < 1e-12
    assert w.regret_ext(P, ["a", "b", "c", "d"], s0, k=4) == 0.0


def test_over_rate():
    w = _w()
    Psh = np.array([[0.9, 0.9, 0.9], [0.2, 0.2, 0.2]])                   # 2 条扩充配置 × 3 条 shared
    over = w.over_rate(Psh, v_aug=np.array([0.5, 0.95]), end0_sh=np.array([0.4, 0.6, 0.9]))
    # 预测胜率 (0.9 + 0.2) / 2 = 0.55；真值胜率 (1/3 + 1) / 2 = 2/3 → 高估 −0.1167
    assert abs(over - (0.55 - 2 / 3)) < 1e-12


def test_dedupe_aug():
    w = _w()
    ents = [("t", "k1", [0], 0.5), ("t", "k1", [0], 0.6), ("t", "k2", [1], 0.4), ("t", "lab", [2], 0.9), ("u", "k1", [0], 0.3)]
    out = w.dedupe_aug(ents, labeled={"t": {"lab"}, "u": set()})
    assert [(e[0], e[1], e[3]) for e in out] == [("t", "k1", 0.5), ("t", "k2", 0.4), ("u", "k1", 0.3)]   # 先到先留、剔已标


def test_select_alpha():
    w = _w()
    assert w.select_alpha({0.3: 0.020, 1.0: 0.012, 3.0: 0.011}) == 1.0         # 0.012 与最小值差 ≤ 0.002 → 取较小
    assert w.select_alpha({0.3: 0.020, 1.0: 0.016, 3.0: 0.011}) == 3.0
    assert w.select_alpha({0.3: 0.010, 1.0: 0.010, 3.0: 0.010}) == 0.3


def test_verdict():
    w = _w()
    assert w.verdict_s3w(0.650, 0.630, mech=True) == "S3W-BEATS-BOTH+E+MECH"
    assert w.verdict_s3w(0.630, 0.625, mech=False) == "S3W-BEATS-TS2VEC"
    assert w.verdict_s3w(0.610, 0.590, mech=False) == "S3W-UP+E"
    assert w.verdict_s3w(0.600, 0.600, mech=True) == "S3W-FLAT+MECH"
    assert w.verdict_s3w(0.580, 0.600, mech=False) == "S3W-DOWN"


def test_cli():
    r, w = _r(), _w()
    a = r.build_parser().parse_args(["--train", "--arm", "E", "--fold", "Cricket", "--cons", "1", "--cons-ks", "500"])
    assert a.cons == 1.0 and a.cons_ks == 500 and a.cons_gens == 3 and a.cons_top == 5 and a.cons_ref == 16 and a.cons_period == 3
    assert r.build_parser().parse_args(["--train"]).cons == 0.0
    b = w.build_parser().parse_args(["--search", "--arm", "CONS", "--dataset", "Cricket"])
    assert b.search and b.arm == "CONS"
    assert set(w.ARMS_W) == {"CONS", "CONS_B"}


def _train(tmp_path, cons, arm="E"):
    r = _r()
    ns = types.SimpleNamespace(outdir=str(tmp_path / f"c{cons}_{arm}"), device="cpu", resume=False, synthetic=True, max_epochs=1, max_tasks_per_src=1,
                               fold="Cricket", seed=0, arm=arm, stop="acc", gate=0.0, e_mode="settf", no_shared=False, raw_dir=None,
                               cons=cons, cons_ks=60, cons_gens=1, cons_top=3, cons_ref=4, cons_period=3)
    r.cmd_train(ns)
    return np.load(tmp_path / f"c{cons}_{arm}" / "train" / f"{arm}_s0_Cricket.npz")


def test_cons_zero_is_s3rp(tmp_path):
    r = _r()
    z0 = _train(tmp_path, 0.0)
    ns = types.SimpleNamespace(outdir=str(tmp_path / "plain"), device="cpu", resume=False, synthetic=True, max_epochs=1, max_tasks_per_src=1,
                               fold="Cricket", seed=0, arm="E", stop="acc", gate=0.0, e_mode="settf", no_shared=False, raw_dir=None)
    r.cmd_train(ns)                                                            # 旧调用（无 cons 字段）
    z1 = np.load(tmp_path / "plain" / "train" / "E_s0_Cricket.npz")
    h0 = [{k: v for k, v in h.items() if k != "sec"} for h in __import__("json").loads(str(z0["hist"]))]
    h1 = [{k: v for k, v in h.items() if k != "sec"} for h in __import__("json").loads(str(z1["hist"]))]
    assert h0 == h1 and float(z0["cons"]) == 0.0


def test_cons_trains(tmp_path):
    z = _train(tmp_path, 1.0)
    ch = __import__("json").loads(str(z["cons_hist"]))
    assert float(z["cons"]) == 1.0 and len(ch) == 1
    assert ch[0]["n_tasks_refreshed"] > 0 and 0.0 <= ch[0]["pred_adv"] <= 1.0 and 0.0 <= ch[0]["base_rate"] <= 1.0
    zb = _train(tmp_path, 1.0, arm="B")
    assert float(zb["cons"]) == 1.0
