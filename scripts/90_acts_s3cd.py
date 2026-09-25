"""WP17-ACTS S3-C-D（results/acts/s3cd/PROTOCOL.md）：6 个 UEA 数据集的部署口径检验——T-CLSC（A）/ 任务盲比较器（D）/ 4 探针（④）。

官方 train / test，整数据集 E（只用 TRAIN），按终点 test acc 在前 3 里选 1（WP3 旧口径），seed 0。
比较器 = S3-C 该数据集留出那一折的模型；④ 低秩模型 = S3-A 该折（只在训练源拟合）。真跑走 83 phase W（`--profile s1pp`）。
阶段：
  --prep          本地：靶尺寸 / tasks_meta、④ 每靶 4 条探针配置、shared-30 目录 → meta.json
  --embed         服务器：每靶官方 TRAIN 上 fit 单变量 TS2Vec（= S3-B E_pt 参数）→ E/<D>.npz
  --search        服务器：--arm A|D --dataset <D> → search/<arm>_<D>.json（= S3-C 搜索）
  --plan-a        服务器：A / D 前 3 + ④ 探针 → runs/plan_W.json（phase W，官方 eval）
  --score-probe   本地：探针 @25% test acc + iters=0 地板 → ridge → μ+âB_K → probe_top3.json
  --plan-b        本地：plan_W 并入 ④ 前 3（已有格不重复）
  --analyze       本地：逐靶表
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "acts" / "s3cd"
S3C = ROOT / "results" / "acts" / "s3c"
UEA30_TABLE = ROOT / "results" / "uea30" / "table30.json"


# ================================================================ 纯函数
def task_id(d: str) -> str:
    return f"{d}_whole_official"


def pick_by_test(keys: Sequence[str], test: Mapping[str, float]) -> tuple[str, float]:
    """WP3 旧口径：终点 test acc 最高者（并列取先出现）。"""
    best = max(keys, key=lambda k: test[k])
    return best, float(test[best])


def target_probe_X(acc25: Mapping[str, float], probe_keys: Sequence[str], pfloor: float, maj: float) -> np.ndarray:
    """与 84.probe_feature 同构的 5 维特征：[d1, d2, d1/den, d2/den, lvl − maj]，den = max(0.01, lvl − pfloor)。probe_keys 顺序 = 84.probe_configs。"""
    v = np.array([acc25[k] for k in probe_keys], float)
    d1, d2 = v[0] - v[1], v[2] - v[3]
    lvl = v.mean()
    den = max(0.01, lvl - pfloor)
    return np.array([d1, d2, d1 / den, d2 / den, lvl - maj])


# ================================================================ prep（本地）
def cmd_prep(args) -> None:
    ac = importlib.import_module("scripts.acts_common"); g83 = importlib.import_module("scripts.83_acts_corpus")
    g84 = importlib.import_module("scripts.84_acts_s3a"); g85 = importlib.import_module("scripts.85_acts_s3ap"); g89 = importlib.import_module("scripts.89_acts_s3c")
    sizes = g83._uea_sizes(list(g89.FOLDS))
    cat = json.loads((ROOT / "results/acts/s2/catalog.json").read_text(encoding="utf-8"))
    shared = [(s["cls_key"], s["cls"]) for s in cat["shared"]]
    data = g85.load_all(); D = data["pool"]; groups = D["groups"]; b_ref = g85.orient_bref(D)
    tm, probes = {}, {}
    for d in g89.FOLDS:
        sz = sizes[d]
        tm[task_id(d)] = {"dataset": d, "n_train": int(sz["n_train"]), "T": int(sz["L"]), "C": int(sz["C"]), "eval_mode": "official",
                          "n_train_official": int(sz["n_train"]), "n_test_official": int(sz["n_test"]), "tag": "whole_official"}
        tr = np.where(groups != d)[0]
        ctx = g84._fold(D["Z"], tr, 3, b_ref)
        probes[d] = [shared[j][0] for j in g84.probe_configs(ctx)]
    meta = {"datasets": list(g89.FOLDS), "tasks_meta": tm, "probes": probes, "shared": [{"cls_key": k, "cls": c} for k, c in shared]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[prep] {len(tm)} 靶；④ 探针 {probes}", flush=True)


def load_meta(outdir: Path) -> dict:
    return json.loads((outdir / "meta.json").read_text(encoding="utf-8"))


# ================================================================ embed / search（服务器）
def cmd_embed(args) -> None:
    import torch
    ac = importlib.import_module("scripts.acts_common"); g86 = importlib.import_module("scripts.86_acts_s3b")
    from autocls.task_embed import TS2VecWrapper
    outdir = Path(args.outdir); meta = load_meta(outdir); (outdir / "E").mkdir(parents=True, exist_ok=True)
    for d in (args.datasets or meta["datasets"]):
        p = outdir / "E" / f"{d}.npz"
        if p.exists() and args.resume:
            continue
        m = meta["tasks_meta"][task_id(d)]
        X, y = ac.load_dataset_full(d)
        Xtr = ac.whole_arrays_official(X, y, n_train=m["n_train_official"])["Xtr"]
        n, t, c = Xtr.shape
        pool = Xtr.transpose(0, 2, 1).reshape(n * c, t, 1)
        iters = g86.ts2vec_iters(pool.size); split_len = 0
        t0 = time.time()
        try:
            w = TS2VecWrapper(input_dims=1, output_dims=g86.REPR, hidden_dims=64, depth=10, device=args.device, batch_size=g86.PT_BATCH, seed=0)
            w.fit(pool, n_iters=iters)
        except Exception as e:  # noqa: BLE001
            if "out of memory" not in str(e).lower():
                raise
            w = None; torch.cuda.empty_cache(); split_len = g86.SH_MAX_LEN
            w = TS2VecWrapper(input_dims=1, output_dims=g86.REPR, hidden_dims=64, depth=10, device=args.device, batch_size=g86.PT_BATCH, seed=0)
            w.fit(g86.split_long(pool, g86.SH_MAX_LEN), n_iters=iters)
        w.freeze()
        r = g86.encode_windows(w, Xtr, g86.window_idx(n, task_id(d)))
        np.savez_compressed(p, windows_repr=r, iters=iters, split_len=split_len, seconds=time.time() - t0)
        print(f"[embed] {d} Xtr={Xtr.shape} iters={iters} split={split_len} {time.time() - t0:.0f}s", flush=True)
        del w; torch.cuda.empty_cache() if torch.cuda.is_available() else None


def cmd_search(args) -> None:
    import torch
    ac = importlib.import_module("scripts.acts_common"); g89 = importlib.import_module("scripts.89_acts_s3c")
    from autocls.labeling import run_seed_for
    from autocls.search.evolution import evolutionary_search
    from autocls.space.cls_encoder import CLSEncoder
    outdir = Path(args.outdir); d, arm, dev = args.dataset, args.arm, args.device
    op = outdir / "search" / f"{arm}_{d}.json"; op.parent.mkdir(parents=True, exist_ok=True)
    if op.exists() and args.resume:
        print(f"[search] skip {arm}_{d}"); return
    cfg = ac.load_acts_config(profile="s2"); space = ac.acts_space(cfg); enc = CLSEncoder(space, mode="binned")
    model = g89.build_model(arm, enc.dim, dev)
    model.load_state_dict(torch.load(Path(args.s3c) / "models" / f"{arm}_{d}.pt", map_location=dev)); model.eval()
    sc = g89.Scorer(model, arm, dev)
    T = None
    if arm in ("A", "B"):
        with np.load(outdir / "E" / f"{d}.npz") as z:
            W = z["windows_repr"]
        with torch.no_grad():
            T = sc.task_vec(W)
    tid = task_id(d); ks = args.ks or g89.KS; gens = args.gens if args.gens is not None else g89.GENS
    t0 = time.time()
    rng = np.random.default_rng(run_seed_for(tid, "s3c_ks"))
    cands = ac._sample_catalog(space, cfg, rng, ks, set())
    Vc = np.stack([enc.encode(c) for c, _k in cands]).astype(np.float32); keys = [k for _c, k in cands]
    top = g89.chunk_select(len(cands), lambda idx: sc.pmat(Vc[np.asarray(idx)], T), keys)

    def score_pairs(a, b):
        with torch.no_grad():
            at = torch.as_tensor(np.asarray(a, np.float32), device=dev); bt = torch.as_tensor(np.asarray(b, np.float32), device=dev)
            Tb = None if T is None else T.unsqueeze(0).expand(len(at), -1)
            return sc.forward(at, bt, Tb).float().cpu().numpy()

    evo = evolutionary_search(score_pairs, space, enc, ks=0, kp=g89.KP, generations=gens, p_cross=g89.P_CROSS, p_mut=g89.P_MUT, top_k=3, batch_size=65536,
                              rng=np.random.default_rng(run_seed_for(tid, "s3c_evo")), seed_population=[cands[j][0] for j in top],
                              accept=lambda c: bool(space.validate(c)) and not ac.is_pruned(c, cfg))
    res = {"nominated": [{"cls_key": e["cls_key"], "cls": e["cls"], "wins": e["wins"], "generation_found": e["generation_found"]} for e in evo],
           "init_top": [keys[j] for j in top], "ks": len(cands), "gens": gens, "seconds": time.time() - t0}
    op.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"[search] {arm}_{d} → {[e['cls_key'] for e in evo]} {time.time() - t0:.0f}s", flush=True)


# ================================================================ 计划（phase W）
def _cells(meta: Mapping, entries: Sequence[tuple[str, str, Mapping, str]]) -> list[dict]:
    """entries = [(dataset, cls_key, cls, origin)] → 83 phase W 官方口径格（seed 0，去重）。"""
    ac = importlib.import_module("scripts.acts_common"); g35 = importlib.import_module("scripts.35_protocol_alignment")
    from autocls.space.cls_space import CLS
    cfg = ac.load_acts_config(profile="s1pp"); protos = g35.load_protocols()
    out, seen = [], set()
    for d, k, c, origin in entries:
        tid = task_id(d)
        cell = ac.make_cell(tid, meta["tasks_meta"][tid], origin, CLS(**c), k, 0, protos, cfg, extra={"eval": "official", "whole_mode": "official"})
        if cell["key"] not in seen:
            seen.add(cell["key"]); out.append(cell)
    return out


def _write_plan(outdir: Path, meta: Mapping, cells: Sequence[dict]) -> None:
    rd = outdir / "runs"; rd.mkdir(parents=True, exist_ok=True)
    plan = {"phase": "W", "profile": "s1pp", "n_cells": len(cells), "n_tasks": len(meta["tasks_meta"]), "probe_tasks": list(meta["tasks_meta"]),
            "protocol": "ts2vec_official", "tasks_meta": meta["tasks_meta"], "cells": list(cells)}
    (rd / "plan_W.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    print(f"[plan] {len(cells)} 格 → {rd / 'plan_W.json'}", flush=True)


def cmd_plan_a(args) -> None:
    ac = importlib.import_module("scripts.acts_common")
    outdir = Path(args.outdir); meta = load_meta(outdir)
    shared = {s["cls_key"]: s["cls"] for s in meta["shared"]}
    ent = []
    for d in meta["datasets"]:
        for arm in ("A", "D"):
            r = json.loads((outdir / "search" / f"{arm}_{d}.json").read_text(encoding="utf-8"))
            ent += [(d, e["cls_key"], e["cls"], ac.ARM_RANDOM) for e in r["nominated"]]
        ent += [(d, k, shared[k], ac.ARM_SHARED) for k in meta["probes"][d]]
    _write_plan(outdir, meta, _cells(meta, ent))


def _rows(outdir: Path) -> dict:
    out = {}
    p = outdir / "runs" / "runs.jsonl"
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if not r.get("error") and r.get("checkpoint_results"):
                out[(r["task_id"], r["cls_key"])] = r
    return out


def cmd_score_probe(args) -> None:
    g84 = importlib.import_module("scripts.84_acts_s3a"); g85 = importlib.import_module("scripts.85_acts_s3ap")
    outdir = Path(args.outdir); meta = load_meta(outdir)
    rows = _rows(outdir)
    probe0 = [json.loads(l) for l in (outdir / "runs" / "probe0.jsonl").read_text(encoding="utf-8").splitlines()]
    data = g85.load_all(); D = data["pool"]; groups = D["groups"]; b_ref = g85.orient_bref(D)
    ref = g84.loso_evaluate(D["Z"], groups, D["end0"], D["end1"], {"probe": g84.probe_feature(D["acc25"], D["pfloor"], D["maj"])}, b_ref=b_ref)
    kfold = dict(zip(list(dict.fromkeys(groups.tolist())), ref["probe"]["k_chosen"]))
    shared_keys = [s["cls_key"] for s in meta["shared"]]
    out = {}
    for d in meta["datasets"]:
        tid = task_id(d)
        tr = np.where(groups != d)[0]
        ctx = g84._fold(D["Z"], tr, 3, b_ref)
        assert [shared_keys[j] for j in g84.probe_configs(ctx)] == meta["probes"][d]
        acc25 = {k: float(rows[(tid, k)]["checkpoint_results"][0]["acc"]) for k in meta["probes"][d]}
        pr = [r for r in probe0 if r["task_id"] == tid]
        pfloor = float(np.mean([r["acc"] for r in pr])); maj = float(pr[0]["majority"])
        x_t = target_probe_X(acc25, meta["probes"][d], pfloor, maj)
        X_pool = g84.probe_feature(D["acc25"], D["pfloor"], D["maj"])(ctx)
        X = np.vstack([X_pool, x_t[None, :]]); gall = np.concatenate([groups, np.array(["__target__"])])
        A = np.vstack([ctx["A"][:, :3], np.zeros((1, 3))])
        a_hat = g84._predict_a(X, A, gall, tr, np.array([len(X) - 1]), g84.ALPHAS)[0]
        K = int(kfold[d])
        s = g84.scores_from(ctx["mu"], ctx["B"][:, :K], a_hat[None, :K])[0]
        top = [shared_keys[j] for j in np.argsort(-s, kind="stable")[:3]]
        out[d] = {"top3": top, "a_hat": a_hat.tolist(), "K": K, "acc25": acc25, "pfloor": pfloor, "majority": maj, "features": x_t.tolist()}
        print(f"[score-probe] {d} K={K} â₁={a_hat[0]:+.2f} top3={top}", flush=True)
    (outdir / "probe_top3.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_plan_b(args) -> None:
    ac = importlib.import_module("scripts.acts_common")
    outdir = Path(args.outdir); meta = load_meta(outdir)
    shared = {s["cls_key"]: s["cls"] for s in meta["shared"]}
    plan = json.loads((outdir / "runs" / "plan_W.json").read_text(encoding="utf-8"))
    pt = json.loads((outdir / "probe_top3.json").read_text(encoding="utf-8"))
    new = _cells(meta, [(d, k, shared[k], ac.ARM_SHARED) for d in meta["datasets"] for k in pt[d]["top3"]])
    have = {c["key"] for c in plan["cells"]}
    add = [c for c in new if c["key"] not in have]
    _write_plan(outdir, meta, plan["cells"] + add)
    print(f"[plan-b] 新增 ④ 前 3 格 {len(add)}（其余与探针 / 搜索提名重合）", flush=True)


# ================================================================ analyze（本地）
def cmd_analyze(args) -> None:
    from scipy.stats import wilcoxon
    outdir = Path(args.outdir); meta = load_meta(outdir); rows = _rows(outdir)
    pt = json.loads((outdir / "probe_top3.json").read_text(encoding="utf-8"))
    t30 = {r["dataset"]: r for r in json.loads(UEA30_TABLE.read_text(encoding="utf-8"))["rows"]}
    res = {}
    for d in meta["datasets"]:
        tid = task_id(d)
        acc = lambda k: float(rows[(tid, k)]["checkpoint_results"][-1]["acc"])
        arms = {}
        for arm in ("A", "D"):
            ks = [e["cls_key"] for e in json.loads((outdir / "search" / f"{arm}_{d}.json").read_text(encoding="utf-8"))["nominated"]]
            arms[arm] = ks
        arms["④"] = pt[d]["top3"]
        rec = {"lit_ts2vec": t30.get(d, {}).get("paper_ts2vec"), "wp_sh1043_3seed": t30.get(d, {}).get("main_mean"), "wp_oracle22": t30.get(d, {}).get("oracle22")}
        for a, ks in arms.items():
            tests = {k: acc(k) for k in ks}
            k_, v = pick_by_test(ks, tests)
            rec[a] = {"pick": k_, "test": v, "nominated": tests, "mean3": float(np.mean(list(tests.values())))}
        rec["A_D_overlap"] = len(set(arms["A"]) & set(arms["D"]))
        rec["probes_acc25"] = pt[d]["acc25"]
        res[d] = rec
    ds = meta["datasets"]
    def pair(a, b):
        x = np.array([res[d][a]["test"] - res[d][b]["test"] for d in ds])
        p = float(wilcoxon(x).pvalue) if np.any(np.abs(x) > 1e-12) else 1.0
        return {"mean": float(x.mean()), "n_pos": int((x > 0).sum()), "n_neg": int((x < 0).sum()), "p": p}
    summ = {"A-D": pair("A", "D"), "A-④": pair("A", "④"), "④-D": pair("④", "D"),
            "mean": {a: float(np.mean([res[d][a]["test"] for d in ds])) for a in ("A", "D", "④")},
            "mean_lit": float(np.mean([res[d]["lit_ts2vec"] for d in ds if res[d]["lit_ts2vec"] is not None])),
            "n_ge_lit": {a: int(sum(res[d][a]["test"] >= res[d]["lit_ts2vec"] for d in ds if res[d]["lit_ts2vec"] is not None)) for a in ("A", "D", "④")}}
    out = {"per_dataset": res, "summary": summ}
    (outdir / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    f = lambda x: "—" if x is None else f"{x:.3f}"
    L = ["# S3-C-D 部署口径读数（官方 train / test，按 test acc 选，seed 0）", "",
         "| 数据集 | A T-CLSC | D 任务盲比较器 | ④ 4 探针 | TS2Vec 论文 | 旧部署 SH10-4-3（3 种子，旧目录） | 旧目录 oracle |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in ds:
        r = res[d]
        L.append(f"| {d} | {f(r['A']['test'])} | {f(r['D']['test'])} | {f(r['④']['test'])} | {f(r['lit_ts2vec'])} | {f(r['wp_sh1043_3seed'])} | {f(r['wp_oracle22'])} |")
    m = summ["mean"]
    L.append(f"| **均值** | {f(m['A'])} | {f(m['D'])} | {f(m['④'])} | {f(summ['mean_lit'])} | {f(float(np.mean([res[d]['wp_sh1043_3seed'] for d in ds])))} | — |")
    L += ["", "配对（6 集）：" + "；".join(f"{k} {v['mean']:+.3f}（{v['n_pos']}+ / {v['n_neg']}−，p {v['p']:.3f}）" for k, v in summ.items() if isinstance(v, dict) and "mean" in v and k != "mean"),
          "", f"≥ TS2Vec 论文数的数据集：A {summ['n_ge_lit']['A']}/6、D {summ['n_ge_lit']['D']}/6、④ {summ['n_ge_lit']['④']}/6", "",
          "## 各臂 3 条提名的 test acc", "", "| 数据集 | 臂 | 3 条 test acc | 3 条均值 | 选中 |", "|---|---|---|---:|---|"]
    for d in ds:
        for a in ("A", "D", "④"):
            r = res[d][a]
            L.append(f"| {d} | {a} | {' / '.join(f(v) for v in r['nominated'].values())} | {f(r['mean3'])} | {r['pick']} |")
    L += ["", "A 与 D 提名重合条数：" + "，".join(f"{d} {res[d]['A_D_overlap']}" for d in ds)]
    (outdir / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("prep", "embed", "search", "plan-a", "score-probe", "plan-b", "analyze"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--arm", choices=("A", "D"), default="A")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--ks", type=int, default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--s3c", default=str(S3C), help="S3-C 产物目录（models/）")
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for flag, fn in (("prep", cmd_prep), ("embed", cmd_embed), ("search", cmd_search), ("plan_a", cmd_plan_a),
                     ("score_probe", cmd_score_probe), ("plan_b", cmd_plan_b), ("analyze", cmd_analyze)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
