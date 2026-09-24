"""S3-A 事后（锁后、看过 tables.md 之后）：同 160 任务 / 17 源 LOSO 上的 SH6-3-3 / SH10-4-3（S2 _headroom_four 同口径，先验 = 训练源平均秩），供成本对照。"""
import importlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
g84 = importlib.import_module("scripts.84_acts_s3a"); g83 = importlib.import_module("scripts.83_acts_corpus"); ac = importlib.import_module("scripts.acts_common")
s2 = ROOT / "results/acts/s2"
cfg = ac.load_acts_config(profile="s2")
plan = json.loads((s2 / "plan_S.json").read_text(encoding="utf-8")); cat = json.loads((s2 / "catalog.json").read_text(encoding="utf-8"))
an = json.loads((s2 / "analysis_S.json").read_text(encoding="utf-8")); rows = g83._g35.load_done(s2 / "runs.jsonl")
meta = plan["tasks_meta"]; keys = g83.cat_keys(cat); raw = g83._raw_seed_frames(plan, rows, cat, cfg)
tasks = g84.pool_tasks(plan["probe_tasks"], meta, an["pool_after_gates"], an["reverse_demoted_to_8"])
M = g83._fill_pairs({"end_0": np.array([raw[t][0] for t in tasks]), "end_1": np.array([raw[t][1] for t in tasks])})
for nm, idx in (("q25_0", 0), ("q50_0", 1)):
    M[nm] = g83._fill_single(np.array([[g83._ckpt(rows, plan, t, k, 0, idx) for k in keys] for t in tasks]), M["end_0"])
groups = np.array([meta[t]["dataset"] for t in tasks])
H = g83._headroom_four(M, groups)
out = {k: H[k] for k in ("prior1", "top3", "sh633", "sh1043")}
per = {k: np.array(H["per_task"][k]) for k in ("top3", "sh633", "sh1043")}
res = json.loads((ROOT / "results/acts/s3a/analysis.json").read_text(encoding="utf-8"))
out["ci_sh633"] = g84.cluster_bootstrap_ci(per["sh633"], groups); out["ci_sh1043"] = g84.cluster_bootstrap_ci(per["sh1043"], groups)
print(json.dumps(out, indent=1))
(ROOT / "results/acts/s3a/posthoc_sh.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
