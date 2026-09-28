"""WP17-ACTS S3-Y（results/acts/s3y/PROTOCOL.md）——先于实现。覆盖：反对称头的单配置打分、顶部排序权重、
目标 TRAIN 分层 K 折、候选并集、按分数进化、CV 选择、读数码、命令行、--head field --loss lambda 能训练且默认路径不变。"""
from __future__ import annotations

import importlib
import json
import types

import numpy as np
import torch


def _r():
    return importlib.import_module("scripts.105_acts_s3r")


def _y():
    return importlib.import_module("scripts.111_acts_s3y")


def test_tclsc_score_matches_forward():
    from autocls.comparator.tclsc_model import TCLSC
    from autocls.task_embed import TaskSetEncoder
    torch.manual_seed(0)
    a, b = torch.randn(7, 12), torch.randn(7, 12)
    E = torch.randn(7, 16)
    for head in ("field", "diff"):
        m = TCLSC(12, TaskSetEncoder(dim_input=8, nz=16), task_branch="E+ctx", head=head).eval()
        with torch.no_grad():
            p = m(a, b, E); s = torch.sigmoid(m.score(a, E) - m.score(b, E))
        assert torch.allclose(p, s, atol=1e-6)
        m0 = TCLSC(12, None, task_branch="none", head=head).eval()
        with torch.no_grad():
            assert torch.allclose(m0(a, b, None), torch.sigmoid(m0.score(a) - m0.score(b)), atol=1e-6)
    m = TCLSC(12, TaskSetEncoder(dim_input=8, nz=16), task_branch="E+ctx", head="concat")
    try:
        m.score(a, E); assert False, "concat 头没有单配置分数"
    except ValueError:
        pass


def test_top_weights():
    r = _r()
    L = np.array([0.9, 0.8, 0.5, 0.4, 0.1, 0.05])             # 后两条失败
    fail = np.array([False, False, False, False, True, True])
    use = np.ones(6, bool)
    a = np.array([0, 2, 0, 1]); b = np.array([1, 3, 4, 2])
    lam = r.top_weights(L, fail, use, a, b)
    assert lam[1] == 0.0                                        # 两条都在可用中位以下 → 0
    assert lam[0] > 0 and lam[2] > lam[0]                       # 顶部 vs 失败 > 顶部两条之间（增益差更大）
    w = np.array([1.0, 0.5, 0.25, 1.0])
    w2 = r.apply_top(w, lam)
    assert abs(w2.sum() - w.sum()) < 1e-12 and w2[1] == 0.0      # 总和归一回原值
    assert np.array_equal(r.apply_top(w, np.zeros(4)), w)        # 全零 → 原权重


def test_whole_arrays_tcv():
    ac = importlib.import_module("scripts.acts_common")
    rng = np.random.default_rng(0)
    y = np.r_[np.repeat([0, 1, 2], [9, 12, 6]), np.repeat([0, 1, 2], 4)]      # 前 27 = 官方 TRAIN
    X = rng.normal(size=(len(y), 10, 2)).astype(np.float32)
    ev_all = []
    for f in range(3):
        a = ac.whole_arrays_tcv(X, y, n_train=27, k=3, fold=f, seed=5)
        assert len(set(a["train_idx"]) & set(a["eval_idx"])) == 0
        assert sorted(set(a["train_idx"]) | set(a["eval_idx"])) == list(range(27))           # TEST 不读
        assert all((y[a["eval_idx"]] == c).sum() >= 1 for c in (0, 1, 2))                  # 分层
        assert np.allclose(a["Xtr"].reshape(-1, 2).mean(0), 0, atol=1e-5)
        ev_all += a["eval_idx"].tolist()
    assert sorted(ev_all) == list(range(27))                                                # 3 折划分
    b = ac.whole_arrays_tcv(X, y, n_train=27, k=3, fold=1, seed=5)
    assert np.array_equal(b["eval_idx"], ac.whole_arrays_tcv(X, y, n_train=27, k=3, fold=1, seed=5)["eval_idx"])


def test_union_and_select():
    y = _y()
    u = y.union_candidates([["a", "b"], ["b", "c"], ["d", "a"]])
    assert u == ["a", "b", "c", "d"]
    cv = {"a": [0.8, 0.7, 0.9], "b": [0.9, 0.8, 0.7], "c": [0.5, 0.5, 0.5]}
    ens = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert y.select_cv(["a", "b", "c"], cv, ens) == "b"             # a、b 均值相同 → 集成分数高者
    assert y.select_cv(["a", "c"], {"a": [0.8, 0.8, 0.8], "c": [0.9, 0.9, 0.9]}, ens) == "c"


def test_zscore_and_evolve():
    y = _y()
    s = np.array([1.0, 2.0, 3.0, 4.0])
    mu, sd = y.norm_stats(s)
    assert abs(mu - 2.5) < 1e-12 and abs(sd - np.std(s)) < 1e-12
    ac = importlib.import_module("scripts.acts_common")
    from autocls.space.cls_encoder import CLSEncoder
    cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg); enc = CLSEncoder(space, mode="binned")
    cands = ac._sample_catalog(space, cfg, np.random.default_rng(0), 40, set())
    wv = np.random.default_rng(1).normal(size=enc.dim)
    fit = lambda V: np.asarray(V, float) @ wv                                          # noqa: E731
    out = y.score_evolve(fit, [c for c, _k in cands], space, enc, cfg, np.random.default_rng(2), kp=10, gens=3)
    f = fit(np.stack([enc.encode(c) for c in out]))
    assert len(out) == 10 and np.all(np.diff(f) <= 1e-12)                                # 按适应度降序
    assert f[0] >= fit(np.stack([enc.encode(c) for c, _k in cands])).max() - 1e-12       # 不差于初始最好


def test_verdict():
    y = _y()
    assert y.verdict_s3y(0.650, 0.630) == "S3Y-BEATS-BOTH+E"
    assert y.verdict_s3y(0.630, 0.625) == "S3Y-BEATS-TS2VEC"
    assert y.verdict_s3y(0.610, 0.590) == "S3Y-UP+E"
    assert y.verdict_s3y(0.600, 0.600) == "S3Y-FLAT"
    assert y.verdict_s3y(0.580, 0.600) == "S3Y-DOWN"


def test_cli():
    r, y = _r(), _y()
    a = r.build_parser().parse_args(["--train", "--arm", "E", "--fold", "Cricket", "--head", "field", "--loss", "lambda"])
    assert a.head == "field" and a.loss == "lambda"
    d = r.build_parser().parse_args(["--train"])
    assert d.head == "concat" and d.loss == "bce"
    b = y.build_parser().parse_args(["--search", "--arm", "NEW", "--dataset", "Cricket"])
    assert b.search and b.arm == "NEW" and set(y.ARMS_Y) == {"NEW", "NEW_B"}


def _train(tmp_path, arm, **kw):
    r = _r()
    ns = types.SimpleNamespace(outdir=str(tmp_path / f"{arm}_{kw.get('head', 'x')}"), device="cpu", resume=False, synthetic=True, max_epochs=1,
                               max_tasks_per_src=1, fold="Cricket", seed=0, arm=arm, stop="acc", gate=0.0, e_mode="settf", no_shared=False,
                               raw_dir=None, **kw)
    r.cmd_train(ns)
    return np.load(tmp_path / f"{arm}_{kw.get('head', 'x')}" / "train" / f"{arm}_s0_Cricket.npz")


def test_train_field_lambda(tmp_path):
    for arm in ("E", "B"):
        z = _train(tmp_path, arm, head="field", loss="lambda")
        assert str(z["head"]) == "field" and str(z["loss"]) == "lambda" and len(json.loads(str(z["hist"]))) == 1


def test_default_path_unchanged(tmp_path):
    z0 = _train(tmp_path, "E", head="concat", loss="bce")
    z1 = _train(tmp_path, "E")
    h = lambda z: [{k: v for k, v in e.items() if k != "sec"} for e in json.loads(str(z["hist"]))]   # noqa: E731
    assert h(z0) == h(z1)
