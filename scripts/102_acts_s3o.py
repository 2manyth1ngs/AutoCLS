"""WP17-ACTS S3-O（results/acts/s3o/PROTOCOL.md）：表示空间响应画像 E_T（每任务单变量 TS2Vec，50 次 / 原码总次数两个截点的 rep_profile）进入 E 可预测偏好轴。

阶段：
  --compute-et    服务器 GPU：--parts / --part → et/part<p>.jsonl（每任务 et50 / etF 各 22 维，--resume）
  --merge-et      → et.npz（与 S3-C 任务顺序对齐）
  --prep          本地：= 100.cmd_prep（E 候选加入 E_T 族）+ 第一道门（E_T 族最好 EV ≥ +0.02）
  第二、三阶段用 `100_acts_s3m.py --outdir results/acts/s3o`。
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

OUT = ROOT / "results" / "acts" / "s3o"
BASE_VARS = ("sh4", "pt4", "pt4+sh4")
ET_VARS = ("et50", "etF", "et50+etF", "etP4", "etP4+pt4+sh4")
EARLY = 50
GATE1_EV = 0.02


def _g(name: str):
    return importlib.import_module(f"scripts.{name}")


# ================================================================ 纯函数
def channel_mean_encoder(w, batch_size: int = 64):
    """(N, T, C) → 各通道单变量编码（全序列最大池化）→ 跨通道均值 → (N, D)（同 E_pt 的通道处理）。"""
    def enc(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32); n, t, c = X.shape
        r = np.asarray(w.encode(X.transpose(0, 2, 1).reshape(n * c, t, 1), encoding_window="full_series", batch_size=batch_size))
        return r.reshape(n, c, -1).mean(1)
    return enc


def gate1(tab: Mapping[tuple, float]) -> bool:
    et = [v for k, v in tab.items() if str(k[0]).startswith("et")]
    return bool(et) and max(et) >= GATE1_EV


# ================================================================ E_T 计算（服务器 GPU）
def cmd_compute_et(args) -> None:
    import torch
    from autocls.data.mech_features import REP_PROFILE_COLS, rep_profile
    from autocls.task_embed import TS2VecWrapper
    from autocls.task_types import TASK_CLASSIFICATION
    g86, g89, g101 = _g("86_acts_s3b"), _g("89_acts_s3c"), _g("101_acts_s3n")
    meta_c, _arr = g89.load_data(Path(args.s3c)); tasks = meta_c["tasks"]
    _all, meta, _srcs, _extra = g86.task_meta()
    todo = [t for i, t in enumerate(sorted(tasks, key=lambda t: meta[t]["dataset"])) if args.parts <= 1 or i % args.parts == args.part]
    d = Path(args.outdir) / "et"; d.mkdir(parents=True, exist_ok=True)
    fp = d / f"part{args.part}.jsonl"
    done = {json.loads(l)["task"] for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()} if fp.exists() else set()
    with open(fp, "a", encoding="utf-8") as fh:
        for tid in todo:
            if tid in done:
                continue
            t0 = time.time()
            X, y = g101._task_Xy(tid, meta)
            n, t, c = X.shape
            pool = X.transpose(0, 2, 1).reshape(n * c, t, 1)
            iters = g86.ts2vec_iters(pool.size)
            split_len, fit_pool = 0, pool
            w = TS2VecWrapper(input_dims=1, output_dims=g86.REPR, hidden_dims=64, depth=10, device=args.device, batch_size=g86.PT_BATCH, seed=0)
            try:
                w.fit(fit_pool, n_iters=EARLY)
            except Exception as e:                                   # 同 E_pt：OOM → 切段重训
                if "out of memory" not in str(e).lower():
                    raise
                w = None; torch.cuda.empty_cache()
                split_len, fit_pool = g86.SH_MAX_LEN, g86.split_long(pool, g86.SH_MAX_LEN)
                w = TS2VecWrapper(input_dims=1, output_dims=g86.REPR, hidden_dims=64, depth=10, device=args.device, batch_size=g86.PT_BATCH, seed=0)
                w.fit(fit_pool, n_iters=EARLY)
            enc = channel_mean_encoder(w)
            p50 = rep_profile(enc, X, y, task_type=TASK_CLASSIFICATION)
            w.fit(fit_pool, n_iters=iters)                           # 累计到原码总次数（续训时优化器重建）
            pF = rep_profile(enc, X, y, task_type=TASK_CLASSIFICATION)
            rec = {"task": tid, "shape": [int(n), int(t), int(c)], "iters": int(iters), "split_len": int(split_len), "sec": round(time.time() - t0, 1),
                   "et50": [p50[k] for k in REP_PROFILE_COLS], "etF": [pF[k] for k in REP_PROFILE_COLS]}
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
            print(f"[et] {tid} X={X.shape} iters={iters} split={split_len} {rec['sec']}s", flush=True)
            del w
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def cmd_merge_et(args) -> None:
    from autocls.data.mech_features import REP_PROFILE_COLS
    meta_c, _arr = _g("89_acts_s3c").load_data(Path(args.s3c)); tasks = meta_c["tasks"]
    rows = {}
    for p in sorted((Path(args.outdir) / "et").glob("part*.jsonl")):
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l); rows[r["task"]] = r
    miss = [t for t in tasks if t not in rows]
    assert not miss, f"E_T 缺 {len(miss)} 个任务：{miss[:5]}"
    to = lambda k: np.array([[np.nan if v is None else float(v) for v in rows[t][k]] for t in tasks], float)  # noqa: E731
    np.savez_compressed(Path(args.outdir) / "et.npz", ET50=to("et50"), ETF=to("etF"), tasks=np.array(tasks), cols=np.array(REP_PROFILE_COLS),
                        sec=np.array([rows[t]["sec"] for t in tasks]), split=np.array([rows[t]["split_len"] for t in tasks]))
    print(f"[merge] {len(tasks)} 任务；NaN et50 {int(np.isnan(to('et50')).sum())} / etF {int(np.isnan(to('etF')).sum())}；切段 {sum(rows[t]['split_len'] > 0 for t in tasks)}", flush=True)


# ================================================================ 第一道门（本地）
def cmd_prep(args) -> None:
    g99, g100, g101 = _g("99_acts_s3l"), _g("100_acts_s3m"), _g("101_acts_s3n")
    outdir = Path(args.outdir)
    with np.load(outdir / "et.npz") as z:
        E50, EF = np.asarray(z["ET50"], float), np.asarray(z["ETF"], float)
    both = np.hstack([E50, EF])

    def feat(var, fold, groups, em_sh, em_pt):
        groups = np.asarray(groups)
        if not var.startswith("et"):
            return g99.e_features(var, fold, groups, em_sh, em_pt)
        head, _, rest = var.partition("+")
        if var == "et50+etF":
            head, rest = "etBoth", ""
        U = {"et50": lambda: g101.ea_block(E50, groups, fold, None), "etF": lambda: g101.ea_block(EF, groups, fold, None),
             "etBoth": lambda: g101.ea_block(both, groups, fold, None), "etP4": lambda: g101.ea_block(both, groups, fold, 4)}[head]()
        return np.hstack([U, g99.e_features(rest, fold, groups, em_sh, em_pt)]) if rest else U
    g100.cmd_prep(args, e_vars=BASE_VARS + ET_VARS, feat=feat)
    pj = json.loads((outdir / "prep.json").read_text(encoding="utf-8"))
    tab = {(k.split("|")[0], float(k.split("|")[1]), int(k.split("|")[2])): v for k, v in pj["ev_table"].items()}
    et_best = max(((k, v) for k, v in tab.items() if k[0].startswith("et")), key=lambda kv: kv[1])
    base_best = max(((k, v) for k, v in tab.items() if not k[0].startswith("et")), key=lambda kv: kv[1])
    pj.update({"gate1_et": gate1(tab), "et_best": [list(et_best[0]), et_best[1]], "base_best": [list(base_best[0]), base_best[1]]})
    (outdir / "prep.json").write_text(json.dumps(pj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gate1] E_T 族最好 {et_best}；均值嵌入最好 {base_best} → {'通过' if pj['gate1_et'] else '未过 → S3O-NO-GAIN'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in ("compute-et", "merge-et", "prep"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--s3c", default=str(ROOT / "results" / "acts" / "s3c"))
    ap.add_argument("--outdir", default=str(OUT))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for flag, fn in (("compute_et", cmd_compute_et), ("merge_et", cmd_merge_et), ("prep", cmd_prep)):
        if getattr(args, flag):
            fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
