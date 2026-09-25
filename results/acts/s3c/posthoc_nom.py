"""S3-C 事后（看过 tables.md 之后）：每臂提名 3 条的平均真值（seed 0 / 1 均值）vs 随机 3 条 vs 目录任务盲前 3 条 vs 目录中位；提名来自初始种群还是进化代数。"""
import importlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
g83 = importlib.import_module("scripts.83_acts_corpus"); g85 = importlib.import_module("scripts.85_acts_s3ap")
from scipy.stats import rankdata
OUT = ROOT / "results/acts/s3c"; S2 = ROOT / "results/acts/s2"
meta = json.loads((OUT / "data.json").read_text(encoding="utf-8")); tasks = meta["tasks"]; groups = np.array(meta["groups"])
D = g85.load_all()["pool"]; end0, end1 = D["end0"], D["end1"]
rows = dict(g83._g35.load_done(S2 / "runs.jsonl")); rows.update(g83._g35.load_done(OUT / "runs" / "runs.jsonl"))
val = {}
for c in json.loads((S2 / "plan_S.json").read_text(encoding="utf-8"))["cells"] + json.loads((OUT / "runs" / "plan_X.json").read_text(encoding="utf-8"))["cells"]:
    r = rows.get(c["key"])
    if r and not r.get("error") and r.get("checkpoint_results"):
        val[(c["task_id"], c["cls_key"], int(c["seed"]))] = float(r["checkpoint_results"][-1]["acc"])
out = {a: [] for a in ("A", "B", "C", "D", "R", "cat3", "cat_median")}
gens = {a: [] for a in ("A", "B", "C", "D")}
for s in meta["folds"]:
    for t in meta["search"][s]:
        i = tasks.index(t)
        for a in ("A", "B", "C", "D"):
            e = json.loads((OUT / "search" / f"{a}_{s}.json").read_text(encoding="utf-8"))[t]
            out[a].append(np.mean([(val[(t, x["cls_key"], 0)] + val[(t, x["cls_key"], 1)]) / 2 for x in e["nominated"]]))
            gens[a] += [x["generation_found"] for x in e["nominated"]]
            if a == "A":
                out["R"].append(np.mean([(val[(t, x["cls_key"], 0)] + val[(t, x["cls_key"], 1)]) / 2 for x in e["random"]]))
        trn = np.where(groups != groups[i])[0]
        mr = np.array([rankdata(-end0[j]) for j in trn]).mean(0); o = np.argsort(mr, kind="stable")[:3]
        out["cat3"].append(float(np.mean((end0[i, o] + end1[i, o]) / 2))); out["cat_median"].append(float(np.median((end0[i] + end1[i]) / 2)))
res = {a: float(np.mean(v)) for a, v in out.items()}
res["gen_found_frac_evolved"] = {a: float(np.mean(np.array(v) > 0)) for a, v in gens.items()}
print(json.dumps(res, indent=1)); (OUT / "posthoc_nom.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
