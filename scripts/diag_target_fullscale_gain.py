"""目标池诊断②：UCI-HAR 全量尺度的预训练学习空间（跨受试者 val，test 不碰）。

背景（出路 A 标定 2026-08-12 再判 no_workpoint 后）：任务级切片（1024 样本、
3-6 类子集）即便跨受试者仍基线 0.90-0.99 贴顶，任务级标定不可用是真结论。
但 M1 anchor（GGS 全量 7352 × 2320 iters → test 0.9372）暗示**全量尺度**学习
空间可能存在。本脚本在同一跨受试者 val（留出组 {8,17,21,25}，随机初始化基线
0.891）上测 GGS 全量预训练两档（10ep=580 / 40ep=2320 iters）的 val acc——
增益显著则出路 B（2320 全口径 sweep）可行，否则 UCI-HAR 全尺度也无学习空间。

encoder 走 ensure_pretrained 落盘 encoders/（与主脚本共享：GGS@2320 即将来的
桥接运行，B 路线可逐键复用）。输出 subject_protocol/fullscale_gain_diag.json。
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import yaml  # noqa: E402
from joblib import parallel_backend  # noqa: E402

from autocls.downstream.linear_eval import encode_features, fit_svm  # noqa: E402
from autocls.labeling import cls_key, device_label  # noqa: E402

_tz = importlib.import_module("scripts.5_target_zero_shot")

ARMS_ITERS = (580, 2320)   # 10ep / 40ep（M1 口径）@ 全量 spe=58
SEED = 0
OUT = ROOT / "results" / "target_zero_shot" / "subject_protocol" / \
    "fullscale_gain_diag.json"


def main() -> None:
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    X_train, y_train = _tz.load_target_train()
    subjects = _tz.load_train_subjects(len(y_train))
    fit_idx, val_idx = _tz.subject_val_split(subjects)
    ggs_k = cls_key(_tz._m1.GGS)
    rows = []
    # threading 后端：fit_svm 的 GridSearchCV loky spawn 教训（勿删）
    with parallel_backend("threading", n_jobs=1):
        for iters in ARMS_ITERS:
            t0 = time.time()
            enc, cached = _tz.ensure_pretrained(
                ROOT / "results" / "target_zero_shot", _tz._m1.GGS, ggs_k, SEED,
                iters, X_train, train_cfg, "cuda")
            f_fit = encode_features(enc, X_train[fit_idx])
            f_val = encode_features(enc, X_train[val_idx])
            clf = fit_svm(f_fit, y_train[fit_idx], seed=SEED)
            acc = float(np.mean(clf.predict(f_val) == y_train[val_idx]))
            rows.append({"cls_key": ggs_k, "iters": iters, "seed": SEED,
                         "val_acc": acc, "encoder_cached": cached,
                         "seconds": round(time.time() - t0, 1),
                         "device_name": device_label("cuda")})
            print(f"GGS@{iters}: val_acc={acc:.4f} ({rows[-1]['seconds']}s)", flush=True)
    out = {
        "val_holdout_subjects": list(_tz.VAL_HOLDOUT_SUBJECTS),
        "random_init_val_ref": 0.8911,   # subject_leak_diag 3 种子均值
        "arms": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
