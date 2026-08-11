"""目标池诊断：UCI-HAR train 内切分的受试者泄漏 vs 跨受试者真实 headroom。

背景（2026-08-11 目标池 zero-shot 评估步骤1 发现）：8 个目标任务的随机初始化
基线均值 0.968（5/8 任务 >0.99）、全量随机初始化 val 基线 0.961——train 内
任意分层切分全部贴顶。机制假设：UCI HAR 官方窗口 50% 重叠 + train 内切分
同受试者混杂 → eval 侧几乎每个窗口在 train 侧有同受试者近邻（孪生窗），
SVM 用随机特征做近邻匹配即可贴顶；官方 test 是跨受试者划分（GGS@2320 才
0.9372），学习空间在 test 口径上仍存在。

本脚本只用官方 train 侧数据（测试集纪律不破）：
1. 逐任务：canonical eval 侧样本的受试者是否也出现在 train 侧（重叠率）；
2. 跨受试者随机初始化基线：按受试者留出 ~20% 做 val，随机初始化 encoder +
   SVM 的 val acc ×3 种子——与 0.961（受试者混杂口径）对比，差值即泄漏抬升。

输出 results/target_zero_shot/subject_leak_diag.json。
"""
from __future__ import annotations

import json
import sys
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

from autocls.data.datasets import load_har  # noqa: E402
from autocls.data.source_pool import load_registry  # noqa: E402
from autocls.data.task_slicer import materialize_task  # noqa: E402
from autocls.downstream.linear_eval import encode_features, fit_svm  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import run_seed_for, split_task_train_eval  # noqa: E402

_tz = __import__("importlib").import_module("scripts.5_target_zero_shot")

SUBJECT_TXT = ROOT / "data" / "raw" / "uci_har" / "inner" / "UCI HAR Dataset" / \
    "train" / "subject_train.txt"
N_SEEDS = 3
HOLDOUT_SUBJECT_FRAC = 0.2
OUT = ROOT / "results" / "target_zero_shot" / "subject_leak_diag.json"


def main() -> None:
    X_train, y_train = _tz.load_target_train()
    subjects = np.loadtxt(SUBJECT_TXT, dtype=np.int64)
    assert len(subjects) == len(y_train), \
        f"subject_train.txt {len(subjects)} 行 ≠ train {len(y_train)}（顺序对齐前提破坏）"

    reg = load_registry()
    tasks = _tz.target_tasks(reg, y_train)

    # 1) 逐任务受试者重叠率（canonical 切分 = evaluate_rprime 同种子）
    per_task = {}
    for t in tasks:
        _Xt, yt = materialize_task(t, X_train, y_train)
        tr_idx, ev_idx = split_task_train_eval(
            yt, t.task_type, eval_ratio=0.3,
            seed=run_seed_for(t.task_id, "train_eval_split"))
        gidx = np.array(t.sample_indices)
        s_tr = set(subjects[gidx[tr_idx]].tolist())
        s_ev = subjects[gidx[ev_idx]]
        overlap = float(np.mean([s in s_tr for s in s_ev]))
        per_task[t.task_id] = {"n_eval": int(len(ev_idx)),
                               "eval_subject_in_train_frac": overlap,
                               "n_subjects_train_side": len(s_tr)}
        print(f"{t.task_id}: eval 侧受试者与 train 侧重叠 {overlap:.1%} "
              f"(train 侧 {len(s_tr)} 人)", flush=True)

    # 2) 跨受试者随机初始化基线（只动 train；受试者级留出 ~20%）
    uniq = np.unique(subjects)
    rng = np.random.default_rng(0)
    n_hold = max(1, int(round(len(uniq) * HOLDOUT_SUBJECT_FRAC)))
    hold_subj = set(rng.choice(uniq, size=n_hold, replace=False).tolist())
    val_mask = np.isin(subjects, list(hold_subj))
    fit_idx, val_idx = np.where(~val_mask)[0], np.where(val_mask)[0]
    print(f"跨受试者切分: 留出受试者 {sorted(hold_subj)} → "
          f"fit {len(fit_idx)} / val {len(val_idx)}", flush=True)

    accs = []
    # threading 后端：fit_svm 的 GridSearchCV loky spawn 教训（勿删）
    with parallel_backend("threading", n_jobs=1):
        for s in range(N_SEEDS):
            torch.manual_seed(s)
            enc = build_encoder(input_dims=X_train.shape[-1])
            f_fit = encode_features(enc, X_train[fit_idx])
            f_val = encode_features(enc, X_train[val_idx])
            clf = fit_svm(f_fit, y_train[fit_idx], seed=s)
            acc = float(np.mean(clf.predict(f_val) == y_train[val_idx]))
            accs.append(acc)
            print(f"[subject-disjoint base s{s}] val_acc={acc:.4f}", flush=True)

    out = {
        "per_task_subject_overlap": per_task,
        "subject_disjoint_random_init_val": {
            "held_out_subjects": sorted(int(s) for s in hold_subj),
            "n_fit": int(len(fit_idx)), "n_val": int(len(val_idx)),
            "seeds": accs, "mean": float(np.mean(accs)),
        },
        "subject_mixed_random_init_val_ref":
            "results/target_zero_shot/prep_summary.json: full_random_init_val",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {OUT}", flush=True)


if __name__ == "__main__":
    main()
