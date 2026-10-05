"""WP17-ACTS AL（results/acts/al/PROTOCOL.md）——先于实现。InfoTS 式学增强（aug_learn）：变换、混合、训练器路径、缓存键兼容。"""
from __future__ import annotations

import hashlib
import importlib
import json

import numpy as np
import pytest
import torch


def _la():
    return importlib.import_module("autocls.contrastive.learned_aug")


def test_transforms_shapes():
    la = _la()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 60, 3, generator=g)
    assert la.TRANSFORM_NAMES == ("subsequence", "cutout", "jitter", "scaling", "time_warp", "window_slice", "window_warp")
    for name in la.TRANSFORM_NAMES:
        y = la.TRANSFORMS[name](x, g)
        assert y.shape == x.shape and torch.isfinite(y).all(), name
    sub = la.TRANSFORMS["subsequence"](x, g)
    zero_t = (sub == 0).all(dim=2).all(dim=0)               # 整批同一窗外置零
    assert zero_t.any() or torch.equal(sub, x)


def test_autoaug_mixture():
    la = _la()
    m = la.AutoAug(seed=0)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 60, 3, generator=g)
    w = m.sample_weights(g)
    assert w.shape == (7,) and abs(float(w.sum()) - 1) < 1e-6
    assert torch.equal(m.view(x, g, p1=0.0), x)              # 不增强的 batch → 原样
    assert m.view(x, g, p1=1.0).shape == x.shape


def _enc():
    from autocls.encoders.registry import build_encoder
    torch.manual_seed(0)
    return build_encoder(input_dims=3)


def _cls():
    m1 = importlib.import_module("scripts.35_protocol_alignment")
    return m1.TS2VEC_EQUIV


def test_trainer_off_path_bit_identical():
    from autocls.contrastive.cl_trainer import pretrain_encoder
    X = np.random.default_rng(0).normal(size=(12, 60, 3)).astype(np.float32)
    e1, e2 = _enc(), _enc()
    pretrain_encoder(e1, _cls(), X, max_iters=5, batch_size=4, device="cpu", seed=3)
    pretrain_encoder(e2, _cls(), X, max_iters=5, batch_size=4, device="cpu", seed=3, aug_learn="none")
    for a, b in zip(e1.state_dict().values(), e2.state_dict().values()):
        assert torch.equal(a, b)


def test_trainer_sup_requires_labels_and_updates_meta():
    from autocls.contrastive.cl_trainer import pretrain_encoder
    X = np.random.default_rng(0).normal(size=(12, 60, 3)).astype(np.float32); y = np.array([0, 1, 2] * 4)
    with pytest.raises(ValueError):
        pretrain_encoder(_enc(), _cls(), X, max_iters=7, batch_size=4, device="cpu", seed=3, aug_learn="sup")
    info: dict = {}
    losses = pretrain_encoder(_enc(), _cls(), X, max_iters=7, batch_size=4, device="cpu", seed=3, aug_learn="sup", y_train=y, aug_info=info)
    assert len(losses) >= 2 and info["meta_steps"] >= 1                     # 第 2 遍开始前做过元学习步
    assert info["weights"].shape == (7,) and not np.allclose(info["weights"], info["weights_init"])


def test_cache_key_backward_compatible():
    m = importlib.import_module("scripts.35_protocol_alignment")
    protos = m.load_protocols(); p = protos["ts2vec_official"]
    opts = m.run_options(p)
    old_keys = [k for k in m.OPTION_KEYS if k != "aug_learn"]
    payload = json.dumps({"dataset": "D", "protocol": "ts2vec_official", "cls_key": "abc", "seed": 0, "iters": 200, "arm": "a",
                          "options": {k: opts[k] for k in old_keys}}, sort_keys=True)
    assert m.cache_key("D", "ts2vec_official", "abc", 0, opts, 200, arm="a") == hashlib.sha1(payload.encode()).hexdigest()[:16]
    on = m.run_options(p, {"aug_learn": "sup"})
    assert "aug_learn" in m.OPTION_KEYS and m.cache_key("D", "ts2vec_official", "abc", 0, on, 200, arm="a") != m.cache_key("D", "ts2vec_official", "abc", 0, opts, 200, arm="a")


def test_probe_verdict_and_jobs():
    a = importlib.import_module("scripts.120_acts_al")
    assert a.verdict_al(0.025) == "AL-HEADROOM" and a.verdict_al(0.015) == "AL-WEAK" and a.verdict_al(0.004) == "AL-FLAT"
    assert len(a.job_list()) == 11 and a.REF_KEY
