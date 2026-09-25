"""S3-E 事后（看过 tables.md 之后）：★ 臂 field-on-L4 逐种子的留出 regret@3 / 换源 Δ，以及集成方式（胜场均值 vs P 均值）对照——查集成排序变差的原因。"""
import importlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
g89 = importlib.import_module("scripts.89_acts_s3c")
from autocls.search.rank import symmetrize, wins_from_scores
OUT = ROOT / "results/acts/s3e"; meta, arr = g89.load_data(ROOT / "results/acts/s3c")
tasks = meta["tasks"]; groups = np.array(meta["groups"]); N = g89.N_SH; e0, e1 = arr["end0"], arr["end1"]
star = json.loads((OUT / "select.json").read_text())["star"]
held = [i for i, g in enumerate(groups) if g in g89.FOLDS]
def wins(P):
    P = np.array(P, float); np.fill_diagonal(P, 0.5); return wins_from_scores(symmetrize(P)).astype(float)
def reg(score, i):
    keys = meta["keys"][tasks[i]]; o = sorted(range(N), key=lambda j: (-score[j], keys[j]))[:3]
    return e1[i, int(np.argmax(e0[i]))] - e1[i, o[int(np.argmax(e0[i, o]))]]
Z = {sd: {s: np.load(OUT / "train" / f"{star}_s{sd}_{s}.npz") for s in g89.FOLDS} for sd in range(5)}
res = {}
for sd in range(5):
    r = [reg(wins(Z[sd][groups[i]][f"P/{tasks[i]}"][:N, :N]), i) for i in held]
    x = [np.mean([reg(wins(Z[sd][groups[i]][f"X/{tasks[i]}"][k]), i) for k in range(16)]) for i in held]
    hist = json.loads(str(Z[sd][g89.FOLDS[0]]["hist"]))
    res[f"seed{sd}"] = {"regret3": float(np.mean(r)), "cross": float(np.mean(np.array(x) - np.array(r))),
                        "best_epoch_HW": int(np.argmin([h["val_loss"] for h in hist]))}
rw = [reg(np.mean([wins(Z[sd][groups[i]][f"P/{tasks[i]}"][:N, :N]) for sd in range(5)], 0), i) for i in held]
rp = [reg(wins(np.mean([Z[sd][groups[i]][f"P/{tasks[i]}"][:N, :N] for sd in range(5)], 0)), i) for i in held]
res["ens_mean_wins"] = float(np.mean(rw)); res["ens_mean_P"] = float(np.mean(rp))
print(json.dumps(res, indent=1)); (OUT / "posthoc_seeds.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
