"""目标池诊断③（排查一）：UCI-HAR 官方 test 上的随机初始化基线——终判的直接验证。

背景（subject_protocol/REPORT.md §5-4 留给用户拍板的 1 次 test 评估，2026-08-12
用户授权）：终判"UCI-HAR 不可评"的关键推论是"M1 GGS@2320 test 0.9372 ≈ 随机
特征水平，test 与 val 之差是受试者组难度差异而非预训练收益"——该推论此前只有
val 证据链（随机 0.891 / GGS@2320 0.887），test 侧从未直测。

本脚本：随机初始化 encoder（torch.manual_seed(seed) + build_encoder，M1 同构建
路径、零预训练）× 3 种子 → evaluate_classification（M1 逐行同款：全量 train 7352
拟合 SVM-RBF、官方 test 2947 评估）。

判读（用户预注册，先于数据）：
- gain = GGS@2320 anchor − 随机 test 均值；主锚 rerun_4060=0.9348（同硬件，
  估时锚点带硬件标签的教训），副锚 original=0.9372。
- gain < 0.03（随机 ≈0.93）→ 终判坐实（预训练无净收益），Epilepsy/FD 预检照跑；
- gain ≥ 0.03（随机 ≈0.85 档）→ 评估协议有问题，先修协议、预检推迟。

test 触碰记账：本次为 UCI-HAR 累计第 1 次（此前 0；--final 从未跑）。
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from joblib import parallel_backend  # noqa: E402

from autocls.downstream import evaluate_classification  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import device_label  # noqa: E402

_tz = importlib.import_module("scripts.5_target_zero_shot")

SEEDS = (0, 1, 2)
GAIN_BOUNDARY = 0.03    # 预注册分支边界（用户 2026-08-12 拍板）
OUT = ROOT / "results" / "target_zero_shot" / "subject_protocol" / \
    "test_random_baseline.json"


def main() -> None:
    X_train, y_train = _tz.load_target_train()
    X_test, y_test = _tz.load_target_test()   # test 触碰：本诊断唯一消费点
    assert len(y_test) == 2947, f"test 规模异常: {len(y_test)}"
    rows = []
    # threading 后端：fit_svm 的 GridSearchCV loky spawn 教训（勿删）
    with parallel_backend("threading", n_jobs=1):
        for seed in SEEDS:
            t0 = time.time()
            torch.manual_seed(seed)               # M1 同款：种子只喂权重初始化
            encoder = build_encoder(input_dims=X_train.shape[-1])
            res = evaluate_classification(encoder, X_train, y_train, X_test, y_test)
            rows.append({"seed": seed, "test_acc": res["acc"], "f1": res["f1"],
                         "svm_c": res["svm_c"],
                         "seconds": round(time.time() - t0, 1),
                         "device_name": device_label("cuda")})
            print(f"seed={seed}: test_acc={res['acc']:.4f} "
                  f"({rows[-1]['seconds']}s)", flush=True)
    accs = [r["test_acc"] for r in rows]
    mean = float(np.mean(accs))
    anchors = _tz.M1_GGS_2320
    gains = {k: round(v - mean, 4) for k, v in anchors.items()}
    verdict = ("protocol_ok_verdict_confirmed"
               if gains["rerun_4060"] < GAIN_BOUNDARY else "protocol_suspect")
    out = {
        "purpose": "UCI-HAR test 随机初始化基线（排查一；用户授权的第 1 次 test 触碰）",
        "seeds": list(SEEDS),
        "test_accs": [round(a, 4) for a in accs],
        "mean": round(mean, 4),
        "std": round(float(np.std(accs)), 4),
        "m1_ggs_2320_anchors": anchors,
        "pretrain_gain_vs_random": gains,
        "gain_boundary": GAIN_BOUNDARY,
        "verdict": verdict,
        "val_refs": {"random_init_val": 0.8911, "ggs2320_val": 0.8874},
        "runs": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"mean={mean:.4f} gains={gains} → {verdict}", flush=True)
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
