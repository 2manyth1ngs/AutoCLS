"""S3-K 事后（看过 tables.md 之后写）：E 预测坐标的幅度——â 与真实 a 的标准差之比（留出任务），以及"按训练侧方差放大 â"后的池内 Spearman。"""
import importlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
g98 = importlib.import_module("scripts.98_acts_s3k")
g89 = importlib.import_module("scripts.89_acts_s3c")
meta, arr = g89.load_data(ROOT / "results/acts/s3c"); tasks = meta["tasks"]; groups = np.array(meta["groups"])
rows = []
for s in g89.FOLDS:
    with np.load(ROOT / "results/acts/s3e/axis_targets.npz") as z:
        at = np.asarray(z[f"aux/{s}"], float)[:, :3]
    ah = np.mean([np.load(ROOT / f"results/acts/s3j/train/I_s{sd}_{s}.npz")["Ahat"] for sd in range(5)], axis=0)
    te, tr = groups == s, groups != s
    rows.append({"fold": s, "sd_true_te": at[te].std(0).round(2).tolist(), "sd_hat_te": ah[te].std(0).round(3).tolist(),
                 "sd_hat_tr": ah[tr].std(0).round(3).tolist(), "ratio_tr": (ah[tr].std(0) / at[tr].std(0)).round(3).tolist()})
out = {"per_fold": rows, "mean_ratio_tr": np.mean([r["ratio_tr"] for r in rows], axis=0).round(3).tolist()}
(ROOT / "results/acts/s3k/posthoc_scale.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
