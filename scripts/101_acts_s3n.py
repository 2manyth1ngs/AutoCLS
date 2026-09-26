"""WP17-ACTS S3-N（results/acts/s3n/PROTOCOL.md）：机制特征 E_A（response_profile 25 维）进入 E 可预测偏好轴。

阶段：
  --compute-ea    服务器 CPU：--parts / --part → ea/part<p>.jsonl（每任务 25 维，--resume）
  --merge-ea      服务器 / 本地：→ ea.npz（与 S3-C 任务顺序对齐）
  --prep          本地：= 100.cmd_prep（E 候选加入 E_A 族）+ 第一道门（E_A 族最好 EV ≥ +0.02）→ prep.json / axes.npz
  第二、三阶段直接用 `100_acts_s3m.py --outdir results/acts/s3n`（--train / --rank / --search / --plan-runs / --analyze）。
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

OUT = ROOT / "results" / "acts" / "s3n"
BASE_VARS = ("sh4", "pt4", "pt4+sh4")
EA_VARS = ("ea25", "eaP4", "ea25+pt4+sh4", "eaP4+pt4+sh4")
GATE1_EV = 0.02


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def ea_block(EA: np.ndarray, groups: Sequence, fold: str, pca: Optional[int]) -> np.ndarray:
    """E_A 按折内训练侧（留出源以外）标准化；pca 给定时再在训练侧拟合 PCA、分数再标准化。"""
    EA = np.nan_to_num(np.asarray(EA, float)); tr = np.asarray(groups) != fold
    m, s = EA[tr].mean(0), EA[tr].std(0)
    Z = (EA - m) / np.where(s > 1e-12, s, 1.0)
    if pca:
        g87 = _g("87_acts_s3bp")
        f = g87.pca_fit(Z[tr], int(pca)); Z = g87.pca_apply(f, Z)
        m, s = Z[tr].mean(0), Z[tr].std(0); Z = (Z - m) / np.where(s > 1e-12, s, 1.0)
    return Z


def gate1(tab: Mapping[tuple, float]) -> bool:
    ea = [v for k, v in tab.items() if str(k[0]).startswith("ea")]
    return bool(ea) and max(ea) >= GATE1_EV


# ================================================================ E_A 计算（服务器 CPU）
def _task_Xy(tid: str, meta: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """= 86.task_X（物化切片 → 按通道在全部样本上 z-score）+ 标签。"""
    g83, g86 = _g("83_acts_corpus"), _g("86_acts_s3b")
    from autocls.data.task_slicer import materialize_task
    t = g83.tasks_from_meta({tid: meta[tid]})[tid]
    X, y = g86.dataset(meta[tid]["dataset"])
    Xt, yt = materialize_task(t, X, y)
    raw = Xt.astype(np.float64); flat = raw.reshape(-1, raw.shape[-1])
    mu, sd = flat.mean(0), flat.std(0); sd = np.where(sd < 1e-8, 1.0, sd)
    return ((raw - mu) / sd).astype(np.float32), np.asarray(yt)


def cmd_compute_ea(args) -> None:
    from autocls.data.mech_features import RESPONSE_PROFILE_COLS, response_profile
    from autocls.task_types import TASK_CLASSIFICATION
    g86, g89 = _g("86_acts_s3b"), _g("89_acts_s3c")
    meta_c, _arr = g89.load_data(Path(args.s3c)); tasks = meta_c["tasks"]
    _all, meta, _srcs, _extra = g86.task_meta()
    todo = [t for i, t in enumerate(sorted(tasks, key=lambda t: meta[t]["dataset"])) if args.parts <= 1 or i % args.parts == args.part]
    d = Path(args.outdir) / "ea"; d.mkdir(parents=True, exist_ok=True)
    fp = d / f"part{args.part}.jsonl"
    done = set()
    if fp.exists():
        done = {json.loads(l)["task"] for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()}
    t0 = time.time()
    with open(fp, "a", encoding="utf-8") as fh:
        for tid in todo:
            if tid in done:
                continue
            X, y = _task_Xy(tid, meta)
            r = response_profile(X, y, task_type=TASK_CLASSIFICATION)
            fh.write(json.dumps({"task": tid, "shape": list(X.shape), "ea": [r[c] for c in RESPONSE_PROFILE_COLS]}, default=float) + "\n"); fh.flush()
            print(f"[ea] {tid} X={X.shape} {time.time() - t0:.0f}s", flush=True)


def cmd_merge_ea(args) -> None:
    from autocls.data.mech_features import RESPONSE_PROFILE_COLS
    meta_c, _arr = _g("89_acts_s3c").load_data(Path(args.s3c)); tasks = meta_c["tasks"]
    rows = {}
    for p in sorted((Path(args.outdir) / "ea").glob("part*.jsonl")):
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l); rows[r["task"]] = r["ea"]
    miss = [t for t in tasks if t not in rows]
    assert not miss, f"E_A 缺 {len(miss)} 个任务：{miss[:5]}"
    EA = np.array([[np.nan if v is None else float(v) for v in rows[t]] for t in tasks], float)
    np.savez_compressed(Path(args.outdir) / "ea.npz", EA=EA, tasks=np.array(tasks), cols=np.array(RESPONSE_PROFILE_COLS))
    print(f"[merge] {EA.shape}，NaN {int(np.isnan(EA).sum())}", flush=True)


# ================================================================ 第一道门（本地）
def cmd_prep(args) -> None:
    g99, g100 = _g("99_acts_s3l"), _g("100_acts_s3m")
    outdir = Path(args.outdir)
    with np.load(outdir / "ea.npz") as z:
        EA = np.asarray(z["EA"], float)

    def feat(var, fold, groups, em_sh, em_pt):
        groups = np.asarray(groups)
        if not var.startswith("ea"):
            return g99.e_features(var, fold, groups, em_sh, em_pt)
        head, _, rest = var.partition("+")
        U = ea_block(EA, groups, fold, 4 if head == "eaP4" else None)
        return np.hstack([U, g99.e_features(rest, fold, groups, em_sh, em_pt)]) if rest else U
    g100.cmd_prep(args, e_vars=BASE_VARS + EA_VARS, feat=feat)
    pj = json.loads((outdir / "prep.json").read_text(encoding="utf-8"))
    tab = {tuple([k.split("|")[0], float(k.split("|")[1]), int(k.split("|")[2])]): v for k, v in pj["ev_table"].items()}
    ea_best = max(((k, v) for k, v in tab.items() if k[0].startswith("ea")), key=lambda kv: kv[1])
    base_best = max(((k, v) for k, v in tab.items() if not k[0].startswith("ea")), key=lambda kv: kv[1])
    pj.update({"gate1_ea": gate1(tab), "ea_best": [list(ea_best[0]), ea_best[1]], "base_best": [list(base_best[0]), base_best[1]]})
    (outdir / "prep.json").write_text(json.dumps(pj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gate1] E_A 族最好 {ea_best}；均值嵌入最好 {base_best} → {'通过' if pj['gate1_ea'] else '未过 → S3N-NO-GAIN'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("compute-ea", "merge-ea", "prep"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--s3c", default=str(ROOT / "results" / "acts" / "s3c"))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for flag, fn in (("compute_ea", cmd_compute_ea), ("merge_ea", cmd_merge_ea), ("prep", cmd_prep)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
