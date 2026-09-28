"""S3-V 事后描述（不改门）：① 各臂前 3 提名的 TEST 上限 / 平均（种子 0）与 3 种子 sd；
② 与 S3-C-D 同 6 集已知部署数的对照；③ 排序层：各臂留出 56 任务 regret@3 与可用对准确率（S3-R′ 口径）。"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
v = importlib.import_module("scripts.108_acts_s3v"); r105 = importlib.import_module("scripts.105_acts_s3r")
OUT = ROOT / "results/acts/s3v"
rows = v._rows(OUT); sel = json.loads((OUT / "select.json").read_text(encoding="utf-8"))
L = ["# S3-V 事后描述", "", "## ① 前 3 提名的 TEST（种子 0，完整 TRAIN）", "",
     "| 臂 | 选中（3 种子均值） | 前 3 最好 | 前 3 平均 | 前 3 最差 | 选中 3 种子 sd 均值 |", "|---|---:|---:|---:|---:|---:|"]
o = {}
for a in v.ARMS:
    best, avg, worst, pick, sds = [], [], [], [], []
    per = {}
    for d in v.DATASETS:
        ks = sel[a][d]["ranked"]; t = v.off_tid(d)
        te = [rows[(t, k, 0)] for k in ks]
        k = sel[a][d]["pick"]; ps = [rows[(t, k, s)] for s in v.SEEDS]
        best.append(max(te)); avg.append(np.mean(te)); worst.append(min(te)); pick.append(np.mean(ps)); sds.append(np.std(ps, ddof=1))
        per[d] = {"top3_test_s0": te, "pick": k, "pick_seeds": ps, "val": sel[a][d]["val"]}
    o[a] = {"per": per, "pick": float(np.mean(pick)), "best3": float(np.mean(best)), "avg3": float(np.mean(avg)), "worst3": float(np.mean(worst)),
            "sd": float(np.mean(sds))}
    L.append(f"| {a} | {np.mean(pick):.4f} | {np.mean(best):.4f} | {np.mean(avg):.4f} | {np.mean(worst):.4f} | {np.mean(sds):.3f} |")
L += ["", "逐数据集前 3（种子 0）TEST：", ""]
for a in v.ARMS:
    L.append(f"- {a}：" + "；".join(f"{d[:6]} " + "/".join(f"{x:.3f}" for x in o[a]['per'][d]['top3_test_s0']) for d in v.DATASETS))
# ③ 排序层
dirs = {"FULL": (ROOT / "results/acts/s3rp", "E"), "WO_SETTF": (OUT / "abl_mean", "E"), "WO_TS2VEC": (OUT / "abl_raw", "E"),
        "WO_SHARED": (OUT / "abl_noshared", "E"), "BLIND": (ROOT / "results/acts/s3rp", "B")}
L += ["", "## ③ 排序层（留出 56 任务，S3-R′ 口径）", "", "| 臂 | regret@3 | 可用对准确率 | 好半区准确率 | 最优 epoch 中位 |", "|---|---:|---:|---:|---:|"]
o["rank"] = {}
for a, (dd, arm) in dirs.items():
    H = r105.heldout_scores(dd, arm)
    eps = []
    for f in v.DATASETS:
        for s in v.SEEDS:
            with np.load(dd / "train" / f"{arm}_s{s}_{f}.npz") as z:
                eps.append(int(z["best_epoch"]))
    r3 = float(np.mean([h["r3"] for h in H.values()])); acc = float(np.nanmean([h["acc"] for h in H.values()])); at = float(np.nanmean([h["acc_top"] for h in H.values()]))
    o["rank"][a] = {"r3": r3, "acc": acc, "acc_top": at, "best_epoch_median": float(np.median(eps)), "per_task_r3": {int(i): h["r3"] for i, h in H.items()}}
    L.append(f"| {a} | {r3:.4f} | {acc:.3f} | {at:.3f} | {np.median(eps):.1f} |")
(OUT / "posthoc.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
(OUT / "posthoc.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
