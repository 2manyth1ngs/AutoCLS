"""S3-C 事后（用户要求看逐折结果）：6 个 UEA 折源逐源的排序与搜索读数，另存逐任务数组 per_task.json。
口径与 89 --analyze 完全相同（同一 rr_order / regret_at_k / pick_final / 参照路径），只是按源分组报。"""
import importlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from scipy.stats import rankdata, spearmanr
g83 = importlib.import_module("scripts.83_acts_corpus"); g84 = importlib.import_module("scripts.84_acts_s3a"); g85 = importlib.import_module("scripts.85_acts_s3ap")
g86 = importlib.import_module("scripts.86_acts_s3b"); g88 = importlib.import_module("scripts.88_acts_s3bpp"); g89 = importlib.import_module("scripts.89_acts_s3c")
OUT = ROOT / "results/acts/s3c"; S2 = ROOT / "results/acts/s2"; S3B = ROOT / "results/acts/s3b"
meta, arr = g89.load_data(OUT); tasks = meta["tasks"]; groups = np.array(meta["groups"]); order = meta["order"]; FOLDS = meta["folds"]; N = g89.N_SH
data = g85.load_all(); D = data["pool"]; b_ref = g85.orient_bref(D); end0, end1 = D["end0"], D["end1"]
held = [i for i, g in enumerate(groups) if g in FOLDS]
ref = g84.loso_evaluate(D["Z"], groups, end0, end1, {"probe": g84.probe_feature(D["acc25"], D["pfloor"], D["maj"]), "land25": g84.land25_feature(D["Z25"])}, b_ref=b_ref)
T = g86._load_targets(S3B); all_t = [str(t) for t in T["tasks"]]
em_sh = {f: v[:len(tasks)] for f, v in g86._e_means(S3B, "sh", all_t, order).items()}
fe = g88.run_arm_multi(D, data["phi_sh"], (lambda s: em_sh[s]), None, list(FOLDS), g88.cross_maps(tasks, groups, order))
per = {tasks[i]: {"source": groups[i], "blind": float(ref["_blind"]["regret3"][i]), "probe": float(ref["probe"]["regret3"][i]), "land25": float(ref["land25"]["regret3"][i]),
                  "fieldE": float(g84.regret_at_k(fe["sel"]["pred_sh"][[i]], end0[[i]], end1[[i]], 3)[0])} for i in held}
for arm in g89.ARMS:
    for s in FOLDS:
        with np.load(OUT / "train" / f"{arm}_{s}.npz") as z:
            for i in held:
                t = tasks[i]
                if f"P/{t}" not in z.files:
                    continue
                P = z[f"P/{t}"]; keys = meta["keys"][t]
                o = g89.rr_order(P[:N, :N], keys[:N]); S = np.zeros((1, N)); S[0, o] = -np.arange(N)
                per[t][arm] = float(g84.regret_at_k(S, end0[[i]], end1[[i]], 3)[0])
                oa = g89.rr_order(P, keys); rk = np.empty(len(keys)); rk[oa] = np.arange(len(keys)); R0 = arr[f"{t}/R0"]
                per[t][f"{arm}_sp"] = float(spearmanr(-rk[N:N + len(R0)], R0)[0]) if len(R0) >= 10 else None
                if f"X/{t}" in z.files:
                    xs = []
                    for Px in z[f"X/{t}"]:
                        ox = g89.rr_order(Px, keys[:N]); Sx = np.zeros((1, N)); Sx[0, ox] = -np.arange(N)
                        xs.append(g84.regret_at_k(Sx, end0[[i]], end1[[i]], 3)[0])
                    per[t][f"{arm}_cross"] = float(np.mean(xs))
# 搜索
rows = dict(g83._g35.load_done(S2 / "runs.jsonl")); rows.update(g83._g35.load_done(OUT / "runs" / "runs.jsonl"))
val = {}
for c in json.loads((S2 / "plan_S.json").read_text(encoding="utf-8"))["cells"] + json.loads((OUT / "runs" / "plan_X.json").read_text(encoding="utf-8"))["cells"]:
    r = rows.get(c["key"])
    if r and not r.get("error") and r.get("checkpoint_results"):
        val[(c["task_id"], c["cls_key"], int(c["seed"]))] = float(r["checkpoint_results"][-1]["acc"])
srch = {}
for s in FOLDS:
    for t in meta["search"][s]:
        i = tasks.index(t); rec = {"source": s}
        trn = np.where(groups != groups[i])[0]
        mr = np.array([rankdata(-end0[j]) for j in trn]).mean(0); o = np.argsort(mr, kind="stable")[:3]
        rec["cat3"] = float(end1[i, o[int(np.argmax(end0[i, o]))]]); rec["oracle"] = float(end1[i, int(np.argmax(end0[i]))])
        for arm in list(g89.ARMS) + ["R"]:
            e = json.loads((OUT / "search" / f"{'A' if arm == 'R' else arm}_{s}.json").read_text(encoding="utf-8"))[t]
            ks_ = [x["cls_key"] for x in (e["random"] if arm == "R" else e["nominated"])]
            rec[arm] = g89.pick_final(ks_, {k: val[(t, k, 0)] for k in ks_}, {k: val[(t, k, 1)] for k in ks_})[1]
        srch[t] = rec
(OUT / "per_task.json").write_text(json.dumps({"rank": per, "search": srch}, ensure_ascii=False, indent=1), encoding="utf-8")
f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.3f}"
L = ["# S3-C 逐源读数（事后，口径同 89 --analyze）", "", "## 排序：shared-30 regret@3（越低越好），源内任务均值", "",
     "| 源（任务数） | 任务盲 μ | ④ 4 探针 | ③ land25 | field-E | A | B | C | D | A 换源 Δ | A random-20 Sp |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for s in FOLDS + ["全部"]:
    ts = [t for t, v in per.items() if s == "全部" or v["source"] == s]
    m = lambda k: float(np.nanmean([per[t][k] for t in ts if per[t].get(k) is not None]))
    L.append(f"| {s}（{len(ts)}） | " + " | ".join(f(m(k)) for k in ("blind", "probe", "land25", "fieldE", "A", "B", "C", "D")) + f" | {f(m('A_cross') - m('A'))} | {f(float(np.nanmedian([per[t]['A_sp'] for t in ts if per[t].get('A_sp') is not None])))} |")
L += ["", "## 搜索：最终 acc（seed 0 选、seed 1 评；越高越好），源内 4 任务均值", "",
      "| 源 | 目录 oracle | 目录任务盲前 3 | A | B | C | D | 随机 R | A − 目录前 3 | A − R |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for s in FOLDS + ["全部"]:
    ts = [t for t, v in srch.items() if s == "全部" or v["source"] == s]
    m = lambda k: float(np.mean([srch[t][k] for t in ts]))
    L.append(f"| {s} | " + " | ".join(f(m(k)) for k in ("oracle", "cat3", "A", "B", "C", "D", "R")) + f" | {m('A') - m('cat3'):+.3f} | {m('A') - m('R'):+.3f} |")
L += ["", "## 搜索逐任务（A / 目录前 3 / 目录 oracle）", "", "| 任务 | 目录 oracle | 目录前 3 | A | B | C | D | R |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for t, v in srch.items():
    L.append(f"| {t} | " + " | ".join(f(v[k]) for k in ("oracle", "cat3", "A", "B", "C", "D", "R")) + " |")
(OUT / "per_source.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))
