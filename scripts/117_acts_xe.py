"""WP17-ACTS XE（results/acts/xe/PROTOCOL.md）：换 E 的相似度形式（按偏好近亲检索质量选 E）+ 源专家混合部署。

阶段（服务器流水 scripts/pa_pod/acts_lab_run_xe.sh 依次调用）：
  --compute-et  服务器 GPU：4 个新源 32 任务（物化切片）+ 6 靶（官方 TRAIN）的 E_T（= 102 同法）→ xe/et/part<p>.jsonl（--resume）
  --merge-et    → xe/et_new.npz
  --select      零 GPU：7 种 E 形式在 15 个非靶源上互检索 → M_R → 选形式（XE-SAME / XE-SWAP-<形式>）→ xe/select.json / select.md
  --prep-md     （仅 SWAP）xe/md：meta / 预置 runs / UNI 搜索（= XS，不依赖 E）/ s0src（靶 E 换成选中形式）/ 空 ea_targets / 选中形式矩阵
  --fit         115 --fit 原样（数据 = XS 的 192 任务，E 换成选中形式）
  （部署 = 115 --search --arm ME / --plan cv / --select / --plan final，outdir results/acts/xe/md）
  --analyze     本地：115 读数 + XE 读数码 / τ / 6 靶 π 与偏好相似
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACTS = ROOT / "results" / "acts"
OUT = ACTS / "xe"
XS = ACTS / "xs"
S2 = ACTS / "s2"
S3O = ACTS / "s3o"
S3MD = ACTS / "s3md"
S3Y = ACTS / "s3y"
FORMS = {"pt": ("pt",), "A": ("A",), "T": ("T",), "pt+A": ("pt", "A"), "pt+T": ("pt", "T"), "A+T": ("A", "T"), "pt+A+T": ("pt", "A", "T")}
CURRENT = "pt+A"
ME_NEW, UNI_NEW, BLIND_REF = 0.6302, 0.6189, 0.6377                 # XS ME_new / UNI_new，S3-Z NEW_B
EARLY = 50                                                          # = 102.EARLY


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def build_form(blocks: Mapping[str, np.ndarray], form: str) -> np.ndarray:
    return np.hstack([np.asarray(blocks[b], float) for b in FORMS[form]])


def impute_nan(M: np.ndarray, ref_rows: Sequence[int]) -> np.ndarray:
    """NaN → 参照行（非靶源任务）的列均值（列全缺 → 0）。"""
    M = np.array(M, float)
    mu = np.nan_to_num(np.nanmean(M[list(ref_rows)], axis=0)) if np.isnan(M).any() else None
    if mu is not None:
        r, c = np.where(np.isnan(M)); M[r, c] = mu[c]
    return M


def e_similarity(Eq: np.ndarray, S: np.ndarray) -> np.ndarray:
    """查询任务 E（n_q × d）对候选源均值 E（k × d）的余弦，按查询任务取均值 → (k,)。"""
    Eq = np.atleast_2d(np.asarray(Eq, float)); S = np.asarray(S, float)
    num = Eq @ S.T
    den = np.linalg.norm(Eq, axis=1, keepdims=True) * np.linalg.norm(S, axis=1)[None, :] + 1e-12
    return (num / den).mean(0)


def retrieval_score(esim: np.ndarray, pref: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(esim, pref).correlation)


def hit_at3(esim: np.ndarray, pref: np.ndarray) -> float:
    return len(set(np.argsort(-np.asarray(esim))[:3].tolist()) & set(np.argsort(-np.asarray(pref))[:3].tolist())) / 3.0


def pick_form(scores: Mapping[str, float], dims: Mapping[str, int], current: str = CURRENT) -> str:
    best = max(scores.values())
    tied = [f for f, v in scores.items() if best - v < 1e-9]
    return current if current in tied else min(tied, key=lambda f: (dims[f], f))


def verdict_xe(me_x: float, uni: float, me_new: float = ME_NEW, blind: float = BLIND_REF) -> str:
    v = _g("108_acts_s3v")
    ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    code = "XE-BEATS-BOTH" if (me_x > inf and me_x > ts) else ("XE-BEATS-TS2VEC" if me_x > ts else "XE-BELOW")
    return code + ("+SWAP" if me_x > me_new else "") + ("+BLIND" if me_x > blind else "") + ("+E" if me_x > uni else "")


# ================================================================ 辅助
def _json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write(p: Path, o) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def _targets() -> list:
    return list(_g("108_acts_s3v").DATASETS)


def _jobs(xs: Path) -> list:
    return [("task", t) for t in _json(xs / "new.json")["tasks"]] + [("target", d) for d in _targets()]


def _job_Xy(kind: str, name: str, xs: Path):
    if kind == "task":
        return _g("101_acts_s3n")._task_Xy(name, _json(xs / "tasks.json")["tasks"])
    ac, v = _g("acts_common"), _g("108_acts_s3v")
    X, y = ac.load_dataset_full(name); n = int(_json(S3Y / "meta.json")["tasks_meta"][v.off_tid(name)]["n_train_official"])
    wa = ac.whole_arrays_official(X, y, n_train=n)                  # 只用官方 TRAIN（TEST 不读）
    return wa["Xtr"], wa["ytr"]


def blocks(xs: Path, outdir: Path) -> dict:
    """192 任务（旧 160 = 114.load_all 顺序 ⊕ 新 32 = xs/new.json 顺序）× {pt, A, T} 与 6 靶的同块向量；NaN 用非靶源任务列均值插补。"""
    m114, g116 = _g("114_acts_s3me"), _g("116_acts_xs")
    data = g116.merged_data(m114.load_all, xs); tasks = data["tasks"]; groups = np.asarray(data["groups"]); n_old = len(m114.load_all()["tasks"])
    with np.load(S3O / "et.npz", allow_pickle=True) as z:
        et_old = dict(zip([str(t) for t in z["tasks"]], np.hstack([z["ET50"], z["ETF"]])))
    with np.load(outdir / "et_new.npz", allow_pickle=True) as z:
        et_new = dict(zip([str(t) for t in z["keys"]], np.hstack([z["ET50"], z["ETF"]])))
    T = np.array([et_old[t] if i < n_old else et_new[t] for i, t in enumerate(tasks)], float)
    D = _targets(); ref = [i for i, g in enumerate(groups) if g not in set(D)]
    s0 = np.load(xs / "md" / "s0src.npz"); eat = np.load(S3MD / "ea_targets.npz")
    tgt = {"pt": np.array([s0[f"Ept_target/{d}"] for d in D], float), "A": np.array([eat[d] for d in D], float),
           "T": np.array([et_new[f"TARGET/{d}"] for d in D], float)}
    B = {"pt": np.asarray(data["Ept"], float), "A": np.asarray(data["EA"], float), "T": T}
    out = {"tasks": tasks, "groups": groups, "ref_rows": ref, "data": data}
    for b in B:
        full = impute_nan(np.vstack([B[b], tgt[b]]), ref)
        out[b] = full[: len(tasks)]; out[f"target/{b}"] = full[len(tasks):]
    return out


def source_profiles(xs: Path) -> dict:
    """每源（21）切片画像：shared-30 两种子均值钉底 z 的均值（旧源 = S2 labels，新源 = xs/new.npz 的 L4 前 30 条）。"""
    ac, g83 = _g("acts_common"), _g("83_acts_corpus")
    q = float(ac.load_acts_config(profile="s2")["scale"]["clip_quantile"])
    keys = [k["cls_key"] for k in _json(S2 / "catalog.json")["shared"]]
    pool = _json(S2 / "analysis_S.json")["pool_after_gates"]
    by: dict = {}
    for r in g83.read_jsonl(S2 / "labels.jsonl"):
        if r.get("arm") == "shared":
            by.setdefault(r["task_id"], {})[r["cls_key"]] = float(r["r_prime"])
    prof = {}
    for s in pool:
        Z = [ac.clip_z(np.array([by[t][k] for k in keys]), q)[0] for t in by if t.rsplit("_a", 1)[0] == s and all(k in by[t] for k in keys)]
        prof[s] = np.mean(Z, axis=0)
    nj = _json(xs / "new.json")
    with np.load(xs / "new.npz") as z:
        for s in nj["kept"]:
            prof[s] = np.mean([ac.clip_z(z[f"{t}/r"][:len(keys)], q)[0] for t, g in zip(nj["tasks"], nj["groups"]) if g == s], axis=0)
    return prof


def _pref_sims(prof: Mapping, q: str, cands: Sequence[str]) -> np.ndarray:
    from scipy.stats import spearmanr
    ref = np.mean([prof[c] for c in cands], axis=0)
    return np.array([spearmanr(prof[q] - ref, prof[c] - ref).correlation for c in cands])


# ================================================================ 阶段
def cmd_compute_et(args) -> None:
    import torch
    from autocls.data.mech_features import REP_PROFILE_COLS, rep_profile
    from autocls.task_embed import TS2VecWrapper
    from autocls.task_types import TASK_CLASSIFICATION
    g86, g102 = _g("86_acts_s3b"), _g("102_acts_s3o")
    xs, outdir = Path(args.xs), Path(args.outdir)
    jobs = [j for i, j in enumerate(_jobs(xs)) if args.parts <= 1 or i % args.parts == args.part]
    if args.limit:
        jobs = jobs[: args.limit]
    d = outdir / "et"; d.mkdir(parents=True, exist_ok=True); fp = d / f"part{args.part}.jsonl"
    done = {json.loads(l)["key"] for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()} if fp.exists() else set()
    with open(fp, "a", encoding="utf-8") as fh:
        for kind, name in jobs:
            key = name if kind == "task" else f"TARGET/{name}"
            if key in done:
                continue
            t0 = time.time(); X, y = _job_Xy(kind, name, xs)
            n, t, c = X.shape; pool = X.transpose(0, 2, 1).reshape(n * c, t, 1)
            iters = g86.ts2vec_iters(pool.size) if args.iters is None else int(args.iters)
            split_len, fit_pool = 0, pool
            w = TS2VecWrapper(input_dims=1, output_dims=g86.REPR, hidden_dims=64, depth=10, device=args.device, batch_size=g86.PT_BATCH, seed=0)
            try:
                w.fit(fit_pool, n_iters=EARLY)
            except Exception as e:                                   # = 102：OOM → 切段重训
                if "out of memory" not in str(e).lower():
                    raise
                w = None; torch.cuda.empty_cache()
                split_len, fit_pool = g86.SH_MAX_LEN, g86.split_long(pool, g86.SH_MAX_LEN)
                w = TS2VecWrapper(input_dims=1, output_dims=g86.REPR, hidden_dims=64, depth=10, device=args.device, batch_size=g86.PT_BATCH, seed=0)
                w.fit(fit_pool, n_iters=EARLY)
            enc = g102.channel_mean_encoder(w)
            p50 = rep_profile(enc, X, y, task_type=TASK_CLASSIFICATION)
            w.fit(fit_pool, n_iters=iters)                           # = 102：续训到原码总次数
            pF = rep_profile(enc, X, y, task_type=TASK_CLASSIFICATION)
            rec = {"key": key, "kind": kind, "shape": [int(n), int(t), int(c)], "iters": int(iters), "split_len": int(split_len), "sec": round(time.time() - t0, 1),
                   "et50": [p50[k] for k in REP_PROFILE_COLS], "etF": [pF[k] for k in REP_PROFILE_COLS]}
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
            print(f"[et] {key} X={X.shape} iters={iters} split={split_len} {rec['sec']}s", flush=True)
            del w
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def cmd_merge_et(args) -> None:
    from autocls.data.mech_features import REP_PROFILE_COLS
    xs, outdir = Path(args.xs), Path(args.outdir)
    rows = {}
    for p in sorted((outdir / "et").glob("part*.jsonl")):
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l); rows[r["key"]] = r
    keys = [n if k == "task" else f"TARGET/{n}" for k, n in _jobs(xs)]
    miss = [k for k in keys if k not in rows]
    if miss and not args.allow_missing:
        raise SystemExit(f"E_T 缺 {len(miss)}：{miss[:5]}")
    to = lambda f: np.array([[np.nan if (k not in rows or v is None) else float(v) for v in (rows[k][f] if k in rows else [None] * len(REP_PROFILE_COLS))] for k in keys], float)  # noqa: E731
    np.savez_compressed(outdir / "et_new.npz", ET50=to("et50"), ETF=to("etF"), keys=np.array(keys), cols=np.array(REP_PROFILE_COLS))
    print(f"[merge-et] {len(keys)} 个（缺 {len(miss)}）；NaN et50 {int(np.isnan(to('et50')).sum())} / etF {int(np.isnan(to('etF')).sum())}", flush=True)


def cmd_select(args) -> dict:
    xs, outdir = Path(args.xs), Path(args.outdir)
    B = blocks(xs, outdir); prof = source_profiles(xs); groups = B["groups"]; D = _targets()
    srcs = list(dict.fromkeys(groups.tolist())); nt = [s for s in srcs if s not in set(D)]
    ref = B["ref_rows"]; tab, dims, per_q, tgt_desc = {}, {}, {}, {}
    for f in FORMS:
        M = build_form(B, f); mu, sd = M[ref].mean(0), M[ref].std(0) + 1e-9; Ms = (M - mu) / sd
        Tv = (build_form({b: B[f"target/{b}"] for b in ("pt", "A", "T")}, f) - mu) / sd
        smean = {s: Ms[groups == s].mean(0) for s in srcs}
        R, H = [], []
        for q in nt:
            cands = [s for s in nt if s != q]
            es = e_similarity(Ms[groups == q], np.array([smean[c] for c in cands])); pr = _pref_sims(prof, q, cands)
            R.append(retrieval_score(es, pr)); H.append(hit_at3(es, pr))
        tab[f] = {"M_R": float(np.mean(R)), "hit3": float(np.mean(H)), "R_min": float(np.min(R)), "n_pos": int(sum(r > 0 for r in R))}
        dims[f] = int(M.shape[1]); per_q[f] = dict(zip(nt, map(float, R)))
        desc = {}
        for i, d in enumerate(D):
            cands = [s for s in srcs if s != d]
            es = e_similarity(Tv[i][None, :], np.array([smean[c] for c in cands])); pr = _pref_sims(prof, d, cands)
            desc[d] = retrieval_score(es, pr)
        tgt_desc[f] = desc
    chosen = pick_form({f: tab[f]["M_R"] for f in FORMS}, dims)
    code = "XE-SAME" if chosen == CURRENT else f"XE-SWAP-{chosen}"
    o = {"code": code, "chosen": chosen, "table": tab, "dims": dims, "per_query": per_q, "targets_desc": tgt_desc, "queries": nt}
    _write(outdir / "select.json", o)
    L = ["# XE 第一步：E 形式的近亲检索质量（15 个非靶源互检索，靶盲）", "",
         "| 形式 | 维 | **M_R**（Spearman 均值） | 最小 | R > 0 的查询 | 近亲命中@3 | 6 靶描述（π 与偏好相似 Spearman 均值） |", "|---|---:|---:|---:|---:|---:|---:|"]
    for f in FORMS:
        t = tab[f]
        L.append(f"| {f}{'（现用）' if f == CURRENT else ''} | {dims[f]} | {t['M_R']:+.3f} | {t['R_min']:+.2f} | {t['n_pos']}/{len(nt)} | {t['hit3']:.2f} | "
                 f"{np.mean(list(tgt_desc[f].values())):+.3f} |")
    L += ["", f"读数码：**{code}**（选中 {chosen}）", "", "6 靶逐集（描述）：" + "；".join(f"{d[:8]} " + "/".join(f"{tgt_desc[f][d]:+.2f}" for f in (CURRENT, chosen)) for d in D)
          + f"（{CURRENT} / {chosen}）"]
    (outdir / "select.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return o


def cmd_prep_md(args) -> None:
    xs, outdir = Path(args.xs), Path(args.outdir); sel = _json(outdir / "select.json")
    if sel["code"] == "XE-SAME" and not args.force_form:
        raise SystemExit("[prep-md] XE-SAME：不部署")
    f = args.force_form or sel["chosen"]
    B = blocks(xs, outdir); md = outdir / "md"; (md / "runs").mkdir(parents=True, exist_ok=True); (md / "search").mkdir(parents=True, exist_ok=True)
    shutil.copy2(S3Y / "meta.json", md / "meta.json")
    if not (md / "runs" / "runs.jsonl").exists():
        shutil.copy2(xs / "md" / "runs" / "runs.jsonl", md / "runs" / "runs.jsonl")
    for p in (xs / "md" / "search").glob("UNI_*.json"):
        shutil.copy2(p, md / "search" / p.name)                       # UNI 不依赖 E：与 XS 逐位相同
    with np.load(xs / "md" / "s0src.npz") as z:
        s0 = {k: z[k] for k in z.files}
    Tv = build_form({b: B[f"target/{b}"] for b in ("pt", "A", "T")}, f)
    for i, d in enumerate(_targets()):
        s0[f"Ept_target/{d}"] = Tv[i]
    np.savez_compressed(md / "s0src.npz", **s0)
    np.savez_compressed(md / "ea_targets.npz", **{d: np.zeros(0) for d in _targets()})
    np.savez_compressed(outdir / "form.npz", E=build_form(B, f), form=np.array(f), tasks=np.array(B["tasks"]))
    print(f"[prep-md] 形式 {f}：E {build_form(B, f).shape}，靶 E 已换，UNI 搜索 {len(list((md / 'search').glob('UNI_*.json')))} 个复制", flush=True)


def cmd_fit(args) -> None:
    m114, g115, g116 = _g("114_acts_s3me"), _g("115_acts_s3md"), _g("116_acts_xs")
    xs, outdir = Path(args.xs), Path(args.outdir); base = m114.load_all
    data = g116.merged_data(base, xs)
    with np.load(outdir / "form.npz", allow_pickle=True) as z:
        assert [str(t) for t in z["tasks"]] == list(data["tasks"]); E = np.asarray(z["E"], float)
    data["Ept"] = E; data["EA"] = np.zeros((len(E), 0))                # 115 的 Ecat = [Ept ⊕ EA] → 选中形式
    m114.load_all = lambda: data
    try:
        g115.cmd_fit(argparse.Namespace(outdir=str(outdir / "md"), folds=args.folds, members=args.members))
    finally:
        m114.load_all = base


def cmd_analyze(args) -> dict:
    v, g115 = _g("108_acts_s3v"), _g("115_acts_s3md")
    xs, outdir = Path(args.xs), Path(args.outdir); md = outdir / "md"; D = v.DATASETS
    o = g115.cmd_analyze(argparse.Namespace(outdir=str(md)))
    sel = _json(outdir / "select.json"); fit = _json(md / "fit.json"); ex = np.load(md / "experts.npz"); m = o["means"]
    prof = source_profiles(xs); exs = np.load(xs / "md" / "experts.npz"); fxs = _json(xs / "md" / "fit.json")
    rho = {}
    for d in D:
        for tag, E_, F_ in (("X", ex, fit), ("pt+A", exs, fxs)):
            srcs = F_[d]["ME"]["sources"]; rho.setdefault(d, {})[tag] = retrieval_score(E_[f"ME/{d}/pi"], _pref_sims(prof, d, srcs))
    xe = {"verdict": verdict_xe(m["ME_test"], m["UNI_test"]), "form": sel["chosen"], "tau": {d: fit[d]["ME"]["tau"] for d in D},
          "gamma": {d: fit[d]["ME"]["gamma"] for d in D}, "pi_max": {d: float(ex[f"ME/{d}/pi"].max()) for d in D}, "pi_pref_rho": rho,
          "ME_new": ME_NEW, "UNI_new": UNI_NEW, "BLIND": BLIND_REF}
    xe["predictions"] = {"P1": sel["chosen"] != CURRENT, "P2": "T" in sel["chosen"].split("+"), "P3": m["ME_test"] > ME_NEW, "P4": m["ME_test"] > BLIND_REF,
                         "P5": sum(fit[d]["ME"]["tau"] < 0.5 for d in D) >= 3, "P6": m["ME_test"] > float(np.mean(list(v.INFOTS_S.values())))}
    _write(outdir / "analysis_xe.json", xe)
    z = _json(ACTS / "s3z" / "analysis.json"); xsa = _json(xs / "md" / "analysis.json")
    L = [f"# XE 部署读数（官方 TEST，3 折 CV 选 1，3 种子均值；E = {sel['chosen']}，21 源，主干 S3-Z NEW_B）", "",
         "| 数据集 | **ME_X** | ME_new（pt+A） | UNI_new | BLIND | TS2Vec | InfoTS_s | τ | π 最大 | π 与偏好相似 X / pt+A |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for d in D:
        L.append(f"| {d} | {o['per_arm']['ME'][d]['test']:.3f} | {xsa['per_arm']['ME'][d]['test']:.3f} | {o['per_arm']['UNI'][d]['test']:.3f} | "
                 f"{z['per_arm']['NEW_B'][d]['test']:.3f} | {v.TS2VEC[d]:.3f} | {v.INFOTS_S[d]:.3f} | {xe['tau'][d]:g} | {xe['pi_max'][d]:.2f} | "
                 f"{rho[d]['X']:+.2f} / {rho[d]['pt+A']:+.2f} |")
    L.append(f"| **均值** | **{m['ME_test']:.4f}** | {ME_NEW:.4f} | {m['UNI_test']:.4f} | {BLIND_REF:.4f} | {float(np.mean(list(v.TS2VEC.values()))):.4f} | "
             f"{float(np.mean(list(v.INFOTS_S.values()))):.4f} | | | {np.mean([rho[d]['X'] for d in D]):+.2f} / {np.mean([rho[d]['pt+A'] for d in D]):+.2f} |")
    L += ["", f"读数码：**{xe['verdict']}**；预测 {sum(xe['predictions'].values())}/6", "",
          f"- U 最好 / 平均（种子 0）：ME_X {m['ME_U_best_s0']:.4f} / {m['ME_U_mean_s0']:.4f}；CV 选择损失 {m['ME_sel_loss_s0']:+.4f}"]
    (outdir / "tables_xe.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return xe


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("compute_et", "merge_et", "select", "prep_md", "fit", "analyze", "allow_missing"):
        ap.add_argument(f"--{f.replace('_', '-')}", dest=f, action="store_true")
    ap.add_argument("--force-form", default=None, choices=tuple(FORMS), help="smoke：跳过选形式规则")
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke：每分片只算前 n 个")
    ap.add_argument("--iters", type=int, default=None, help="smoke：覆盖满次数")
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--members", nargs="*", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--xs", default=str(XS))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for flag, fn in (("compute_et", cmd_compute_et), ("merge_et", cmd_merge_et), ("select", cmd_select), ("prep_md", cmd_prep_md),
                     ("fit", cmd_fit), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
