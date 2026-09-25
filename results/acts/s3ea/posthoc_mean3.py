"""S3-EA 事后（看过 tables.md 之后写）：选择噪声稳健性——每臂 3 条提名的两种子均值取平均（不经 seed 0 选 1），与主读数同 24 任务配对。"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
g83, g89, g92 = (importlib.import_module(f"scripts.{m}") for m in ("83_acts_corpus", "89_acts_s3c", "92_acts_s3ea"))
S2, S3C, S3E, OUT = ROOT / "results/acts/s2", ROOT / "results/acts/s3c", ROOT / "results/acts/s3e", ROOT / "results/acts/s3ea"
rows = dict(g83._g35.load_done(S2 / "runs.jsonl")); plans = [json.loads((S2 / "plan_S.json").read_text(encoding="utf-8"))]
for d in (S3C, S3E, OUT):
    rows.update(g83._g35.load_done(d / "runs/runs.jsonl")); plans.append(json.loads((d / "runs/plan_X.json").read_text(encoding="utf-8")))
val = {}
for pl in plans:
    for c in pl["cells"]:
        r = rows.get(c["key"])
        if r and not r.get("error") and r.get("checkpoint_results"):
            val[(c["task_id"], c["cls_key"], int(c["seed"]))] = float(r["checkpoint_results"][-1]["acc"])
st = json.loads((S3C / "per_task.json").read_text(encoding="utf-8"))["search"]; tasks = list(st)
sg = np.array([st[t]["source"] for t in tasks])
arms = ("E-mean", "E-lcb", "B-mean", "B-lcb", "B-lcb-div", "X-lcb")
m3 = {a: [] for a in arms}
for t in tasks:
    s = st[t]["source"]
    e = json.loads((S3E / "search" / f"{s}.json").read_text(encoding="utf-8"))[t]; b = json.loads((OUT / "search" / f"{s}.json").read_text(encoding="utf-8"))[t]
    noms = {"E-mean": e["mean"], "E-lcb": e["lcb"], **{a: b[a] for a in g92.SEARCH_ARMS}}
    for a in arms:
        m3[a].append(float(np.mean([(val[(t, x["cls_key"], 0)] + val[(t, x["cls_key"], 1)]) / 2 for x in noms[a]])))
out = {a: float(np.mean(v)) for a, v in m3.items()}


def cmp(a, b):
    d = np.array(m3[a]) - np.array(m3[b])
    return {"d": float(d.mean()), "p": float(wilcoxon(d).pvalue), "npos": g89.n_sources_positive(d, sg)}
out["E-lcb−B-lcb"] = cmp("E-lcb", "B-lcb"); out["E-lcb−X-lcb"] = cmp("E-lcb", "X-lcb"); out["B-lcb−B-mean"] = cmp("B-lcb", "B-mean")
out["per_source"] = {a: {s: float(np.mean([m3[a][j] for j in range(len(tasks)) if sg[j] == s])) for s in g89.FOLDS} for a in arms}
(OUT / "posthoc_mean3.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "per_source"}, ensure_ascii=False, indent=1))
for a in arms:
    print(a, {s: round(v, 3) for s, v in out["per_source"][a].items()})
