"""WP17-ACTS XS（results/acts/xs/PROTOCOL.md）：UEA 按偏好代表性扩源 + 源专家混合部署。

阶段（服务器流水 scripts/pa_pod/acts_lab_run_xs.sh 依次调用；83 的切片 / 准入 / 计划 / 门、86 / 101 的 E、115 的专家 / 搜索 / 选择原样复用）：
  （阶段 A 准入 = 83 --profile xsadm --slice-all --plan-adm / --run --phase M / --probe0 / --analyze-adm，outdir results/acts/xsadm）
  --select      准入源 > 4 时按残差偏好聚类取 4 个代表（靶盲：参照 = S2 池 11 个非靶源）→ xs/selected.json
  --prep-s      xs profile 注入 sources.probation = 选中源 → 切片 / 并入 xsadm 已跑格 / plan_S（outdir results/acts/xs）
  （阶段 B 补全 = 83 --profile xs --run --phase S / --probe0）
  --gate        8 行 G-P 门（= S2 试用源门）→ xs/gate.json
  --labels      新任务 L4 标签 / vec / 失败（与旧 160 任务同口径）→ xs/new.npz + new.json
  --extract-pt  GPU：新任务 E_pt（86 同参数，只换任务表）→ xs/E_pt/*.npz + _means.npz
  --ea          CPU：新任务 E_A（101 同口径）→ xs/ea_new.npz
  --score-new   S3-Z NEW_B 10 成员 × 6 折对新任务已标配置打分 → xs/md/s0new.npz
  --merge       xs/md：meta.json（S3-Y）/ 预置 runs（S3-MD）/ ea_targets（S3-MD）/ s0src = S3-MD ⊕ 新任务
  --fit         115 --fit 原样（数据 = 旧 160 任务 ⊕ 新任务）
  （部署 = 115 --search / --plan cv / --select / --plan final，outdir results/acts/xs/md）
  --analyze     本地：115 读数 + XS 读数码 / π 新源份额 / 残差幅度
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
OUT = ACTS / "xs"
ADM = ACTS / "xsadm"
S2 = ACTS / "s2"
S3MD = ACTS / "s3md"
S3Y = ACTS / "s3y"
S3Z = ACTS / "s3z"
REJECTED = {"HandMovementDirection", "SelfRegulationSCP2"}       # S2 G-P 已剔除
L_MAX = 1751                                                       # 现源池最长序列（Ethanol）
N_MIN = 200
K_REPS = 4
ADM_ROWS = (1, 7, 3, 5)
ME_OLD, UNI_OLD, BLIND_REF = 0.6334, 0.6154, 0.6377                # S3-MD ME / UNI，S3-Z NEW_B


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def eligible_reasons(name: str, row: Mapping, *, targets, rejected, target_pool, l_max: int, n_min: int = N_MIN) -> list:
    """§1.1 可行集：返回排除理由（空 = 可行）。row = uea_survey.csv 一行。"""
    L, N, eq = int(row["length_max"]), int(row["n_total"]), str(row["equal_length"]) == "True"
    r = []
    if name in targets: r.append("部署靶")
    if name in rejected: r.append("S2 已剔除")
    if name in target_pool: r.append("target_pool（铁律 6）")
    if L < 50: r.append(f"L {L} < 50")
    if L > l_max: r.append(f"L {L} > {l_max}")
    if N < n_min: r.append(f"N {N} < {n_min}")
    if not eq: r.append("变长")
    return r


def _spearman_matrix(R: Sequence[np.ndarray]) -> np.ndarray:
    from scipy.stats import spearmanr
    return np.array([[spearmanr(a, b).correlation for b in R] for a in R])


def select_reps(profiles: Mapping[str, np.ndarray], ref: np.ndarray, k: int = K_REPS) -> list:
    """§1.3：≤ k 全选；否则残差（画像 − 参照）Spearman 距离平均连接 maxclust k，每类取类内平均相关最高者（并列按名字序）。返回名字序。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    names = sorted(profiles)
    if len(names) <= k:
        return names
    C = _spearman_matrix([np.asarray(profiles[n], float) - np.asarray(ref, float) for n in names])
    D = 1.0 - C
    lab = fcluster(linkage(D[np.triu_indices(len(names), 1)], method="average"), k, criterion="maxclust")
    reps = []
    for g in sorted(set(lab.tolist())):
        idx = [i for i in range(len(names)) if lab[i] == g]
        sub = C[np.ix_(idx, idx)]
        reps.append(names[idx[int(np.argmax(sub.mean(1)))]])
    return sorted(reps)


def l4_labels(e0: np.ndarray, e1: np.ndarray, R0: np.ndarray, pfloor: float) -> tuple[np.ndarray, np.ndarray]:
    """= 91._labels("L4") + 105 的失败定义：shared 两种子终点均值 ⊕ random 种子 0 终点；失败 = 不高于探针地板。"""
    r = np.concatenate([(np.asarray(e0, float) + np.asarray(e1, float)) / 2.0, np.asarray(R0, float)])
    return r, r <= float(pfloor) + 1e-12


def merge_s0(old: Mapping, new: Mapping) -> dict:
    """s0src 合并：lens 与每个 {折}/{成员} 按任务顺序拼接（旧 160 在前）；Ept_target 原样；新侧缺的 {折}/{成员} 丢弃。"""
    out = {"lens": np.concatenate([np.asarray(old["lens"]), np.asarray(new["lens"])])}
    for k in old:
        if k == "lens":
            continue
        if k.startswith("Ept_target/"):
            out[k] = np.asarray(old[k])
        elif k in new:
            out[k] = np.concatenate([np.asarray(old[k]), np.asarray(new[k])])
    return out


def verdict_xs(me: float, uni: float, me_old: float = ME_OLD, blind: float = BLIND_REF) -> str:
    v = _g("108_acts_s3v")
    ts, inf = float(np.mean(list(v.TS2VEC.values()))), float(np.mean(list(v.INFOTS_S.values())))
    code = "XS-BEATS-BOTH" if (me > inf and me > ts) else ("XS-BEATS-TS2VEC" if me > ts else "XS-BELOW")
    return code + ("+SRC" if me > me_old else "") + ("+BLIND" if me > blind else "") + ("+E" if me > uni else "")


# ================================================================ 辅助
def _json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write(p: Path, o) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def _profiles(labels: Path, keys: Sequence[str], srcs: Sequence[str], q: float) -> dict:
    """每源：准入 4 行 shared-30 两种子均值（labels r_prime）钉底 z 的均值；有缺的行不计。"""
    ac, g83 = _g("acts_common"), _g("83_acts_corpus")
    by: dict = {}
    for r in g83.read_jsonl(labels):
        if r.get("arm") == ac.ARM_SHARED:
            by.setdefault(r["task_id"], {})[r["cls_key"]] = float(r["r_prime"])
    out = {}
    for s in srcs:
        Z = []
        for row in ADM_ROWS:
            v = np.array([by.get(f"{s}_a{row:02d}", {}).get(k, np.nan) for k in keys], float)
            if np.isfinite(v).all():
                Z.append(ac.clip_z(v, q)[0])
        if Z:
            out[s] = np.mean(Z, axis=0)
    return out


def _cfg_xs(selected: Sequence[str]) -> dict:
    ac = _g("acts_common")
    cfg = ac.load_acts_config(profile="xs")
    cfg["sources"] = dict(cfg["sources"]); cfg["sources"]["probation"] = list(selected)
    return cfg


def _kept(outdir: Path) -> list:
    return list(_json(outdir / "gate.json")["kept"])


# ================================================================ 阶段
def cmd_select(args) -> None:
    ac, v = _g("acts_common"), _g("108_acts_s3v")
    adm, outdir = Path(args.adm), Path(args.outdir)
    q = float(ac.load_acts_config(profile="xsadm")["scale"]["clip_quantile"])
    admitted = list(_json(adm / "analysis_M.json")["admission"]["admitted"])
    keys = [x["cls_key"] for x in _json(adm / "catalog.json")["shared"]]
    ref_src = [s for s in _json(S2 / "analysis_S.json")["pool_after_gates"] if s not in set(v.DATASETS)]
    prof = _profiles(adm / "labels.jsonl", keys, admitted, q)
    ref_prof = _profiles(S2 / "labels.jsonl", keys, ref_src, q)
    ref = np.mean(list(ref_prof.values()), axis=0)
    sel = select_reps(prof, ref) if prof else []
    names = sorted(prof)
    C = _spearman_matrix([prof[n] - ref for n in names]).round(3).tolist() if len(names) > 1 else []
    _write(outdir / "selected.json", {"admitted": admitted, "with_profile": names, "selected": sel, "ref_sources": sorted(ref_prof),
                                      "residual_spearman": {"names": names, "C": C}})
    print(f"[select] 准入 {len(admitted)} {admitted} → 选中 {sel}（参照 {len(ref_prof)} 个非靶源）", flush=True)


def cmd_prep_s(args) -> None:
    g83 = _g("83_acts_corpus")
    outdir = Path(args.outdir); sel = list(args.keep) if args.keep else _json(outdir / "selected.json")["selected"]
    if not sel:
        raise SystemExit("[prep-s] 无选中源（XS-NOSRC）")
    cfg = _cfg_xs(sel)
    g83.cmd_slice_all(cfg, outdir)
    g83.cmd_merge_seeds(cfg, outdir)
    g83.cmd_plan_s2(cfg, outdir)


def cmd_gate(args) -> None:
    g83 = _g("83_acts_corpus")
    outdir = Path(args.outdir); sel = list(args.keep) if args.keep else _json(outdir / "selected.json")["selected"]
    cfg = _cfg_xs(sel)
    plan, cat = _json(outdir / "plan_S.json"), _json(outdir / "catalog.json")
    rows = g83._g35.load_done(outdir / "runs.jsonl"); meta = plan["tasks_meta"]
    raw = g83._raw_seed_frames(plan, rows, cat, cfg)
    res = {}
    for s in sel:
        tids = [t for t in plan["probe_tasks"] if meta[t]["dataset"] == s]
        res[s] = g83.gate_rows({t: raw[t] for t in tids}, {t: float(meta[t]["majority"]) for t in tids}, cfg["probation"])
    kept = [s for s in sel if res[s]["ok"]] if not args.force_keep else list(sel)
    _write(outdir / "gate.json", {"selected": sel, "gate": res, "kept": kept, "forced": bool(args.force_keep)})
    for s in sel:
        print(f"[gate] {s}: v3′ {res[s]['pass']}/{res[s]['n']}、护栏 {res[s]['guard_pass']} → {'保留' if res[s]['ok'] else '剔除'}", flush=True)
    print(f"[gate] 保留 {kept}{'（smoke 强制）' if args.force_keep else ''}", flush=True)


def cmd_labels(args) -> None:
    ac, g83 = _g("acts_common"), _g("83_acts_corpus")
    from autocls.space.cls_encoder import CLSEncoder
    from autocls.space.cls_space import CLS
    outdir = Path(args.outdir); kept = _kept(outdir); cfg = _cfg_xs(kept)
    plan, cat = _json(outdir / "plan_S.json"), _json(outdir / "catalog.json")
    rows = g83._g35.load_done(outdir / "runs.jsonl"); meta = plan["tasks_meta"]
    frames = g83._frames_c(plan, rows, cat, cfg); raw = g83._raw_seed_frames(plan, rows, cat, cfg)
    floors = g83.floors_cv(g83.load_probe0(outdir), meta)
    n_sh = len(cat["shared"]); enc = CLSEncoder(ac.acts_space(cfg), mode="binned")
    phi_sh = np.array([enc.encode(CLS(**s["cls"])) for s in cat["shared"]], float)
    rcls = {tid: {x["cls_key"]: x["cls"] for x in lst} for tid, lst in cat["random"].items()}
    tasks = [t for t in plan["probe_tasks"] if meta[t]["dataset"] in set(kept)]
    empty = [t for t in tasks if not (np.isfinite(raw[t][0]).any() or np.isfinite(raw[t][1]).any())]
    if empty:
        print(f"[labels] !! 跳过无任何 shared 终点的任务 {len(empty)}：{empty[:4]}", flush=True)
    tasks = [t for t in tasks if t not in set(empty)]
    M = g83._fill_pairs({"end_0": np.array([raw[t][0] for t in tasks]), "end_1": np.array([raw[t][1] for t in tasks])})
    out, nf = {}, []
    for i, t in enumerate(tasks):
        f = frames[t]; rk = f["random_keys"]
        R0 = np.asarray(f["frame_vals"], float)[n_sh:]
        phi_r = np.array([enc.encode(CLS(**rcls[t][k])) for k in rk], float).reshape(len(rk), enc.dim)
        pf = float(np.mean(floors[t]["probe_accs"])) if floors[t]["probe_accs"] else float(meta[t]["majority"])
        r, fail = l4_labels(M["end_0"][i], M["end_1"][i], R0, pf)
        out[f"{t}/vec"] = np.vstack([phi_sh, phi_r]).astype(np.float32); out[f"{t}/r"] = r; out[f"{t}/fail"] = fail
        nf.append(float(fail.mean()))
    np.savez_compressed(outdir / "new.npz", **out)
    _write(outdir / "new.json", {"tasks": tasks, "groups": [meta[t]["dataset"] for t in tasks], "kept": kept})
    print(f"[labels] {len(tasks)} 任务（{kept}）；失败占比 {np.mean(nf) if nf else float('nan'):.3f}", flush=True)


def cmd_extract_pt(args) -> None:
    g86 = _g("86_acts_s3b")
    outdir = Path(args.outdir); nj = _json(outdir / "new.json"); tj = _json(outdir / "tasks.json")["tasks"]
    g86.task_meta = lambda: (list(nj["tasks"]), tj, list(dict.fromkeys(nj["groups"])), [])      # 只换任务表，抽取参数 = 86 原样
    g86.cmd_extract_pt(argparse.Namespace(outdir=str(outdir), tasks=None, parts=args.parts, part=args.part, resume=True, iters_cap=None, device=args.device))


def cmd_pack_pt(args) -> None:
    _g("86_acts_s3b").cmd_pack_means(argparse.Namespace(outdir=str(args.outdir)))


def cmd_ea(args) -> None:
    g101 = _g("101_acts_s3n")
    from autocls.data.mech_features import RESPONSE_PROFILE_COLS, response_profile
    from autocls.task_types import TASK_CLASSIFICATION
    outdir = Path(args.outdir); nj = _json(outdir / "new.json"); tj = _json(outdir / "tasks.json")["tasks"]
    EA = []
    for t in nj["tasks"]:
        X, y = g101._task_Xy(t, tj)
        r = response_profile(X, y, task_type=TASK_CLASSIFICATION)
        EA.append([float(r[k]) for k in RESPONSE_PROFILE_COLS])
    np.savez_compressed(outdir / "ea_new.npz", EA=np.array(EA, float), tasks=np.array(nj["tasks"]), cols=np.array(RESPONSE_PROFILE_COLS))
    print(f"[ea] {len(EA)} 任务", flush=True)


def cmd_score_new(args) -> None:
    import torch
    r105, v, g115 = _g("105_acts_s3r"), _g("108_acts_s3v"), _g("115_acts_s3md")
    outdir = Path(args.outdir); md = outdir / "md"; md.mkdir(parents=True, exist_ok=True)
    nj = _json(outdir / "new.json")
    with np.load(outdir / "new.npz") as z:
        Vs = [np.asarray(z[f"{t}/vec"], np.float32) for t in nj["tasks"]]
    V = np.vstack(Vs); out = {"lens": np.array([len(x) for x in Vs])}
    for d in (args.folds or list(v.DATASETS)):
        for s in (args.members or g115.MEMBERS):
            m = r105._build("B", V.shape[1], s, args.device, head="field")
            m.load_state_dict(torch.load(Path(args.models) / f"B_s{s}_{d}.pt", map_location=args.device)); m.eval()
            with torch.no_grad():
                out[f"{d}/{s}"] = m.score(torch.as_tensor(V, device=args.device)).float().cpu().numpy()
    np.savez_compressed(md / "s0new.npz", **out)
    print(f"[score-new] {len(Vs)} 任务 × {len(out) - 1} 个（折 × 成员）", flush=True)


def cmd_merge(args) -> None:
    outdir = Path(args.outdir); md = outdir / "md"; (md / "runs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(S3Y / "meta.json", md / "meta.json")
    shutil.copy2(S3MD / "ea_targets.npz", md / "ea_targets.npz")
    if not (md / "runs" / "runs.jsonl").exists():
        shutil.copy2(S3MD / "runs" / "runs.jsonl", md / "runs" / "runs.jsonl")
    with np.load(S3MD / "s0src.npz") as a, np.load(md / "s0new.npz") as b:
        m = merge_s0({k: a[k] for k in a.files}, {k: b[k] for k in b.files})
    np.savez_compressed(md / "s0src.npz", **m)
    print(f"[merge] s0src 任务 {len(m['lens'])}（旧 {len(m['lens']) - len(np.load(md / 's0new.npz')['lens'])}），键 {len(m) - 1}", flush=True)


def merged_data(base_load, outdir: Path) -> dict:
    """114.load_all（旧 160 任务）⊕ 新任务；字段与 115 --fit 所需一致（tasks / groups / X / Z / Ept / EA / order）。"""
    old = base_load(); nj = _json(outdir / "new.json"); tasks = list(nj["tasks"])
    with np.load(outdir / "new.npz") as z:
        X = [np.asarray(z[f"{t}/vec"], float) for t in tasks]
        Z = []
        for t in tasks:
            r = np.asarray(z[f"{t}/r"], float); Z.append((r - r.mean()) / (r.std() + 1e-9))   # = 114.load_all
    with np.load(outdir / "E_pt" / "_means.npz", allow_pickle=True) as z:
        em = dict(zip([str(t) for t in z["tasks"]], z["E_mean"]))
    with np.load(outdir / "ea_new.npz", allow_pickle=True) as z:
        ea = dict(zip([str(t) for t in z["tasks"]], z["EA"]))
    groups = np.concatenate([np.asarray(old["groups"]), np.array(nj["groups"])])
    return {"tasks": list(old["tasks"]) + tasks, "groups": groups, "X": list(old["X"]) + X, "Z": list(old["Z"]) + Z,
            "Ept": np.vstack([old["Ept"], np.array([em[t] for t in tasks], float)]), "EA": np.vstack([old["EA"], np.array([ea[t] for t in tasks], float)]),
            "order": list(dict.fromkeys(groups.tolist()))}


def cmd_fit(args) -> None:
    m114, g115 = _g("114_acts_s3me"), _g("115_acts_s3md")
    outdir = Path(args.outdir); base = m114.load_all
    data = merged_data(base, outdir)
    m114.load_all = lambda: data                                     # 115 --fit 原样，只换数据
    try:
        g115.cmd_fit(argparse.Namespace(outdir=str(outdir / "md"), folds=args.folds, members=args.members))
    finally:
        m114.load_all = base


def cmd_analyze(args) -> dict:
    v, g115, m114 = _g("108_acts_s3v"), _g("115_acts_s3md"), _g("114_acts_s3me")
    outdir = Path(args.outdir); md = outdir / "md"; D = v.DATASETS
    o = g115.cmd_analyze(argparse.Namespace(outdir=str(md)))
    old = _json(S3MD / "analysis.json"); fit = _json(md / "fit.json"); ex = np.load(md / "experts.npz")
    kept = set(_kept(outdir)); m = o["means"]
    pi_tab = {}
    for d in D:
        for a in ("ME", "UNI"):
            pi = ex[f"{a}/{d}/pi"]; srcs = fit[d][a]["sources"]; top = np.argsort(-pi)[:3]
            pi_tab[f"{a}/{d}"] = {"new_share": float(sum(pi[j] for j, s in enumerate(srcs) if s in kept)), "max": float(pi.max()),
                                  "top3": [[srcs[j], float(pi[j])] for j in top], "uniform": 1.0 / len(srcs)}
    # 残差幅度：旧源 vs 新源（10 成员 s₀ 均值的任务内 z，按折平均）
    data = merged_data(m114.load_all, outdir); s0 = np.load(md / "s0src.npz"); lens = s0["lens"]; g = data["groups"]
    res_mag = {"old": [], "new": []}
    for d in D:
        S = [np.split(np.asarray(s0[f"{d}/{mm}"], float), np.cumsum(lens)[:-1]) for mm in g115.MEMBERS]
        for i in range(len(g)):
            if g[i] == d:
                continue
            e = np.mean([g115.task_z(S[q][i]) for q in range(len(S))], axis=0)
            res_mag["new" if g[i] in kept else "old"].append(float(np.std(np.asarray(data["Z"][i]) - g115.task_z(e))))
    adm = _json(ADM / "analysis_M.json")["admission"]
    xs = {"verdict": verdict_xs(m["ME_test"], m["UNI_test"]), "ME_old": ME_OLD, "UNI_old": UNI_OLD, "BLIND": BLIND_REF,
          "admitted": adm["admitted"], "selected": _json(outdir / "selected.json")["selected"], "kept": sorted(kept), "pi": pi_tab,
          "residual_std": {k: float(np.mean(x)) if x else None for k, x in res_mag.items()},
          "per_dataset_old": {a: {d: old["per_arm"][a][d]["test"] for d in D} for a in ("ME", "UNI")}}
    xs["predictions"] = {"P1": len(adm["admitted"]) >= 3, "P2": m["ME_test"] > ME_OLD, "P3": m["ME_test"] > BLIND_REF, "P4": m["ME_test"] > m["UNI_test"],
                         "P5": sum(any(s in kept for s, _p in pi_tab[f"ME/{d}"]["top3"]) for d in D) >= 3,
                         "P6": m["ME_test"] > float(np.mean(list(v.INFOTS_S.values())))}
    _write(outdir / "analysis_xs.json", xs)
    L = ["# XS 部署读数（官方 TEST，3 折 CV 选 1，3 种子均值；主干 = S3-Z NEW_B，源 17 → 17 + " + str(len(kept)) + "）", "",
         "| 数据集 | **ME_new** | UNI_new | ME_old（S3-MD） | UNI_old | BLIND | TS2Vec | InfoTS_s | ME π 新源份额 | ME π 前 3 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    blind_d = {d: _json(S3Z / "analysis.json")["per_arm"]["NEW_B"][d]["test"] for d in D}
    for d in D:
        a, b, p = o["per_arm"]["ME"][d], o["per_arm"]["UNI"][d], pi_tab[f"ME/{d}"]
        L.append(f"| {d} | {a['test']:.3f} | {b['test']:.3f} | {xs['per_dataset_old']['ME'][d]:.3f} | {xs['per_dataset_old']['UNI'][d]:.3f} | "
                 f"{blind_d[d]:.3f} | {v.TS2VEC[d]:.3f} | {v.INFOTS_S[d]:.3f} | "
                 f"{p['new_share']:.2f} | " + " / ".join(f"{s[:10]} {w:.2f}" for s, w in p["top3"]) + " |")
    L.append(f"| **均值** | **{m['ME_test']:.4f}** | {m['UNI_test']:.4f} | {ME_OLD:.4f} | {UNI_OLD:.4f} | {BLIND_REF:.4f} | "
             f"{float(np.mean(list(v.TS2VEC.values()))):.4f} | {float(np.mean(list(v.INFOTS_S.values()))):.4f} | "
             f"{np.mean([pi_tab[f'ME/{d}']['new_share'] for d in D]):.2f} | |")
    L += ["", f"读数码：**{xs['verdict']}**；预测 {sum(xs['predictions'].values())}/6", "",
          f"- 准入 {len(adm['admitted'])}/6：{adm['admitted']}；选中 {xs['selected']}；8 行门后保留 {xs['kept']}",
          f"- 残差幅度（任务内 z − 主干，std 均值）：旧源 {xs['residual_std']['old']:.3f}、新源 {xs['residual_std']['new']:.3f}",
          f"- U 最好 / 平均（种子 0）：ME {m['ME_U_best_s0']:.4f} / {m['ME_U_mean_s0']:.4f}、UNI {m['UNI_U_best_s0']:.4f} / {m['UNI_U_mean_s0']:.4f}；"
          f"CV 选择损失 ME {m['ME_sel_loss_s0']:+.4f}、UNI {m['UNI_sel_loss_s0']:+.4f}",
          "- ME 各折超参：" + "；".join(f"{d[:6]} γ {fit[d]['ME']['gamma']:g} τ {fit[d]['ME']['tau']:g} α_w {fit[d]['ME']['alpha_w']:g}" for d in D)]
    (outdir / "tables_xs.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    return xs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("select", "prep_s", "gate", "labels", "extract_pt", "pack_pt", "ea", "score_new", "merge", "fit", "analyze", "force_keep"):
        ap.add_argument(f"--{f.replace('_', '-')}", dest=f, action="store_true")
    ap.add_argument("--keep", nargs="*", default=None, help="smoke：跳过 selected.json，直接指定源")
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--members", nargs="*", type=int, default=None)
    ap.add_argument("--models", default=str(S3Z / "models"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--parts", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--adm", default=str(ADM))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for flag, fn in (("select", cmd_select), ("prep_s", cmd_prep_s), ("gate", cmd_gate), ("labels", cmd_labels), ("extract_pt", cmd_extract_pt),
                     ("pack_pt", cmd_pack_pt), ("ea", cmd_ea), ("score_new", cmd_score_new), ("merge", cmd_merge), ("fit", cmd_fit), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
