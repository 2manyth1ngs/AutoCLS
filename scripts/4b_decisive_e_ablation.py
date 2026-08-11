"""决定性实验：T-CLSC 训练 7 跑 + 两档 E 消融 + 协议判读
（results/decisive_e_ablation/PROTOCOL.md §4–§6，判据先于数据）。

与 3_pretrain_tclsc（路线 B）的差异：
1. **per-dataset iters/δ**（labels.yaml workpoint 段）：记录按各自数据集的工作点
   iters 取，配对/训练 δ 按 {task_id: δ} 查表（引擎 DeltaSpec，铁律 7 不进数据集概念）。
2. **任务留出划分预注册**（协议 §4）：HW 06/07/08、UniMiB 06/07/11/15 留出；
   R1–R3 mixed（种子 0/1/2，主判定）、R4 仅 HW、R5 仅 UniMiB、R6/R7 数据集互换。
3. **两档 shuffle E 为主判据**（用户拍板③）：within = 数据集内 derangement
   （杀任务级信号、留域级）、cross = 跨数据集对调（杀域级）——e_mode='assign'
   由本脚本构造映射；各 5 个 assignment 种子。zero 只作辅助。
4. **McNemar 配对显著性**：单向对上 normal 与消融的不一致格 b/c，
   单侧 binomtest(b, b+c, 0.5, greater)（方向 = 消融更差）。
   达标线（协议 §5）：mean Δ ≥ 0.02 且 5 种子中 ≥4 个 p<0.05。

用法：
    python scripts/4b_decisive_e_ablation.py --train --resume   # 7 跑（可断点续）
    python scripts/4b_decisive_e_ablation.py --verdict          # 协议 §6 解读表
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.comparator import TCLSC, TCLSCEngine  # noqa: E402
from autocls.comparator.tclsc_engine import (  # noqa: E402
    Pair, derangement, with_reversed,
)
from autocls.labeling import (  # noqa: E402
    comparator_records, load_rprime_cache, make_pairs,
)
from autocls.space.cls_encoder import CLSEncoder  # noqa: E402
from autocls.space.cls_space import CLSSpace  # noqa: E402
from autocls.task_embed import TaskSetEncoder  # noqa: E402
from autocls.task_types import is_better  # noqa: E402

DATASETS = ("Handwriting", "UniMiB-SHAR")
# 协议 §4 预注册留出（rng seed 0/1 写死，见 PROTOCOL.md）
HOLDOUT = {
    "Handwriting": {"Handwriting_06", "Handwriting_07", "Handwriting_08"},
    "UniMiB-SHAR": {"UniMiB-SHAR_06", "UniMiB-SHAR_07",
                    "UniMiB-SHAR_11", "UniMiB-SHAR_15"},
}
N_ABLATION_SEEDS = 5
PASS_DELTA = 0.02       # 协议 §5 达标线
PASS_SIG_COUNT = 4      # 5 种子 ≥4 个 p<0.05


def ds_of(task_id: str) -> str:
    return task_id.rsplit("_", 1)[0]


def forward_pairs_by_task(records, task_type: str,
                          delta_map: dict[str, float]) -> list[Pair]:
    """静态单向评估对（comparator_eval_pairs 的单向版：先 make_pairs 后不加反向
    ——反向在按任务过滤**之后**加，保证 forward/reverse 结构可用于 McNemar）。"""
    by_task: dict[str, list] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r)
    pairs: list[Pair] = []
    for tid, rs in sorted(by_task.items()):
        rprime = {r.key: r.r for r in rs}
        origin = {r.key: r.origin for r in rs}
        vec = {r.key: r.vec for r in rs}
        raw, _ = make_pairs(rprime, task_type, delta=delta_map[ds_of(tid)],
                            origin=origin)
        pairs += [Pair(task_id=tid, vec_a=vec[p["a"]], vec_b=vec[p["b"]],
                       y=p["y"], kind=p["kind"]) for p in raw]
    return pairs


def within_assignment(test_tids: list[str], seed: int) -> dict[str, str]:
    """within 档：数据集内 derangement（数据集身份不变，杀任务级信号）。"""
    rng = np.random.default_rng([seed, 202608])
    out = {}
    groups: dict[str, list[str]] = {}
    for t in sorted(test_tids):
        groups.setdefault(ds_of(t), []).append(t)
    for _ds, tids in sorted(groups.items()):
        assert len(tids) >= 2, f"within derangement 需要 ≥2 任务：{tids}"
        perm = derangement(len(tids), rng)
        for i, t in enumerate(tids):
            out[t] = tids[perm[i]]
    return out


def cross_assignment(test_tids: list[str], all_tids: list[str],
                     seed: int) -> dict[str, str]:
    """cross 档：每个测试任务的 E 换成另一数据集随机任务的 E（杀域级信号）。"""
    rng = np.random.default_rng([seed, 202609])
    by_ds: dict[str, list[str]] = {}
    for t in sorted(all_tids):
        by_ds.setdefault(ds_of(t), []).append(t)
    out = {}
    for t in sorted(test_tids):
        others = [x for d, tids in by_ds.items() if d != ds_of(t) for x in tids]
        assert others, f"{t} 没有跨数据集候选"
        out[t] = others[rng.integers(len(others))]
    return out


def mcnemar_one_sided(normal_fwd: np.ndarray, abl_fwd: np.ndarray) -> dict:
    """单向对上的 McNemar：b = normal 对而消融错，c = 反之；单侧 p（消融更差）。"""
    b = int(np.sum(normal_fwd & ~abl_fwd))
    c = int(np.sum(~normal_fwd & abl_fwd))
    p = float(binomtest(b, b + c, 0.5, alternative="greater").pvalue) if b + c else 1.0
    return {"b": b, "c": c, "p": p}


def run_one(run_id: str, seed: int, train_tids: set[str], test_tids: set[str],
            records, fwd_pairs, task_windows, delta_map, task_type,
            cfg, nz: int, init_path: Path, device: str) -> dict:
    assert not (train_tids & test_tids), f"{run_id}: 训练/测试任务重叠"
    train_records = [r for r in records if r.task_id in train_tids]
    fwd = [p for p in fwd_pairs if p.task_id in test_tids]
    directed = with_reversed(fwd)
    n_fwd = len(fwd)
    delta_by_task = {t: delta_map[ds_of(t)] for t in (train_tids | test_tids)}

    torch.manual_seed(seed)
    dim_input = next(iter(task_windows.values())).shape[-1]
    set_enc = TaskSetEncoder(dim_input=dim_input, nz=nz)
    set_enc.load_state_dict(torch.load(init_path, map_location="cpu",
                                       weights_only=True))
    encoder_dim = train_records[0].vec.shape[0]
    model = TCLSC(cls_dim=encoder_dim, set_encoder=set_enc,
                  embedding_dim=cfg["embedding_dim"], fz=cfg["fz"])
    engine = TCLSCEngine(model, task_windows, device=device,
                         lr=cfg["lr"], batch_size=cfg["batch_size"], seed=seed)

    t0 = time.time()
    history = engine.train(train_records, epochs=cfg["epochs"],
                           curriculum_epochs=cfg["curriculum_epochs"],
                           better_fn=lambda a, b: is_better(a, b, task_type),
                           delta=delta_by_task)
    train_seconds = round(time.time() - t0, 1)

    base = engine.evaluate(directed, return_correct=True)
    base_fwd = base["correct"][:n_fwd]
    all_tids = sorted(task_windows)

    def ablate(mode: str) -> list[dict]:
        rows = []
        for s in range(N_ABLATION_SEEDS):
            amap = (within_assignment(sorted(test_tids), s) if mode == "within"
                    else cross_assignment(sorted(test_tids), all_tids, s))
            r = engine.evaluate(directed, e_mode="assign", e_assignment=amap,
                                return_correct=True)
            rows.append({"seed": s, "acc": r["acc"], "loss": r["loss"],
                         "delta_acc": base["acc"] - r["acc"],
                         "assignment": amap,
                         "mcnemar": mcnemar_one_sided(base_fwd, r["correct"][:n_fwd])})
        return rows

    within_rows = ablate("within")
    cross_rows = ablate("cross")
    zero = engine.evaluate(directed, e_mode="zero", return_correct=True)

    def agg(rows: list[dict]) -> dict:
        deltas = [r["delta_acc"] for r in rows]
        n_sig = sum(r["mcnemar"]["p"] < 0.05 for r in rows)
        return {"mean_delta": float(np.mean(deltas)),
                "min_delta": float(np.min(deltas)), "max_delta": float(np.max(deltas)),
                "n_sig": n_sig,
                "pass": bool(np.mean(deltas) >= PASS_DELTA and n_sig >= PASS_SIG_COUNT)}

    fwd_acc = float(np.mean(base_fwd)) if n_fwd else float("nan")
    btest = (binomtest(int(round(fwd_acc * n_fwd)), n_fwd, 0.5,
                       alternative="greater").pvalue if n_fwd else float("nan"))
    return {
        "run_id": run_id, "seed": seed,
        "train_tids": sorted(train_tids), "test_tids": sorted(test_tids),
        "n_train_records": len(train_records),
        "n_test_pairs_directed": len(directed), "n_test_pairs_forward": n_fwd,
        "train_seconds": train_seconds,
        "bce_first": history[0]["loss"], "bce_last": history[-1]["loss"],
        "normal": {"acc": base["acc"], "loss": base["loss"],
                   "binom_p_forward": float(btest)},
        "within": {"rows": within_rows, **agg(within_rows)},
        "cross": {"rows": cross_rows, **agg(cross_rows)},
        "zero": {"acc": zero["acc"], "loss": zero["loss"],
                 "delta_acc": base["acc"] - zero["acc"],
                 "mcnemar": mcnemar_one_sided(base_fwd, zero["correct"][:n_fwd])},
    }


def build_runs(all_tids: list[str]) -> list[dict]:
    by_ds = {d: sorted(t for t in all_tids if ds_of(t) == d) for d in DATASETS}
    hw, um = by_ds["Handwriting"], by_ds["UniMiB-SHAR"]
    hw_hold, um_hold = HOLDOUT["Handwriting"], HOLDOUT["UniMiB-SHAR"]
    hw_train = set(hw) - hw_hold
    um_train = set(um) - um_hold
    mixed_train, mixed_test = hw_train | um_train, hw_hold | um_hold
    return [
        {"run_id": f"R{i + 1}_mixed_s{s}", "seed": s,
         "train": mixed_train, "test": mixed_test}
        for i, s in enumerate((0, 1, 2))
    ] + [
        {"run_id": "R4_hw_only", "seed": 0, "train": hw_train, "test": hw_hold},
        {"run_id": "R5_um_only", "seed": 0, "train": um_train, "test": um_hold},
        {"run_id": "R6_swap_hw2um", "seed": 0, "train": set(hw), "test": set(um)},
        {"run_id": "R7_swap_um2hw", "seed": 0, "train": set(um), "test": set(hw)},
    ]


def verdict_from_runs(runs_dir: Path) -> dict:
    """协议 §6 解读表（主判定取 R1–R3 中 Δ_within 的中位数种子跑）。"""
    runs = {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(runs_dir.glob("*.json"))}
    mixed = [r for k, r in runs.items() if k.startswith(("R1", "R2", "R3"))]
    assert len(mixed) == 3, f"主判定需要 R1–R3 全部完成，现有 {sorted(runs)}"
    median_run = sorted(mixed, key=lambda r: r["within"]["mean_delta"])[1]
    w, c = median_run["within"], median_run["cross"]
    r4 = runs.get("R4_hw_only")

    if w["pass"]:
        code, label = "A", "task-aware（任务级）成立——扩池值得"
    elif c["pass"]:
        code, label = "B", "仅 domain-aware——主张降级为 domain-aware CLS 排序"
    elif r4 and r4["within"]["pass"]:
        code, label = "C1", "UniMiB 破坏 label 稀释信号——剔 UniMiB/只用干净 label 重训再判"
    else:
        code, label = "C2", "干净信号缺乏跨任务多样性坐实——扩池救不了，调整主张"
    return {"verdict": code, "label": label,
            "median_run": median_run["run_id"],
            "median_within": {k: w[k] for k in ("mean_delta", "n_sig", "pass")},
            "median_cross": {k: c[k] for k in ("mean_delta", "n_sig", "pass")},
            "r4_within": ({k: r4["within"][k] for k in ("mean_delta", "n_sig", "pass")}
                          if r4 else None)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels-dir", type=Path,
                    default=ROOT / "results" / "labels_workpoint")
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "results" / "decisive_e_ablation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="跳过 runs/ 下已有结果的跑")
    ap.add_argument("--only", nargs="+", default=None, help="只跑指定 run_id")
    ap.add_argument("--verdict", action="store_true")
    args = ap.parse_args()
    if not (args.train or args.verdict):
        ap.error("至少指定 --train / --verdict")

    labels_cfg = yaml.safe_load((ROOT / "configs" / "labels.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "configs" / "pretrain_tclsc.yaml").read_text(encoding="utf-8"))
    task_type = labels_cfg["task_type"]
    wp = labels_cfg["workpoint"]
    iters_map, delta_map = wp["iters_per_dataset"], wp["delta_per_dataset"]
    missing = [d for d in DATASETS if d not in delta_map]
    if missing:
        sys.exit(f"workpoint.delta_per_dataset 缺 {missing}（先标定回填）")
    runs_dir = args.outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.train:
        space = CLSSpace.from_yaml()
        encoder = CLSEncoder(space)
        cache = load_rprime_cache(args.labels_dir / "rprime_cache.jsonl")
        records = []
        for d in DATASETS:
            rs = comparator_records(cache, encoder, iters=iters_map[d])
            assert all(ds_of(r.task_id) == d for r in rs), \
                f"iters={iters_map[d]} 捞到非 {d} 记录（iters 不再唯一区分数据集？）"
            records += rs
        all_tids = sorted({r.task_id for r in records})
        assert len(all_tids) == 28, f"应有 28 任务，得到 {len(all_tids)}"
        fwd_pairs = forward_pairs_by_task(records, task_type, delta_map)
        feat_dir = args.labels_dir / "task_features"
        task_windows = {t: np.load(feat_dir / f"{t}.npz")["windows_repr"]
                        for t in all_tids}
        init_path = args.labels_dir / "set_encoder_init.pt"

        print(f"记录 {len(records)} 条 / 28 任务 | 单向评估对全池 {len(fwd_pairs)} | "
              f"δ={delta_map} iters={iters_map} device={args.device}", flush=True)
        for spec in build_runs(all_tids):
            out_path = runs_dir / f"{spec['run_id']}.json"
            if args.only and spec["run_id"] not in args.only:
                continue
            if args.resume and out_path.exists():
                print(f"[{spec['run_id']}] 已有结果，跳过", flush=True)
                continue
            print(f"[{spec['run_id']}] train={len(spec['train'])}任务 "
                  f"test={len(spec['test'])}任务 ...", flush=True)
            res = run_one(spec["run_id"], spec["seed"], spec["train"], spec["test"],
                          records, fwd_pairs, task_windows, delta_map, task_type,
                          cfg, labels_cfg["nz"], init_path, args.device)
            out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            w, c = res["within"], res["cross"]
            print(f"[{spec['run_id']}] {res['train_seconds']}s | "
                  f"normal {res['normal']['acc']:.4f} | "
                  f"within Δ̄={w['mean_delta']:+.4f} sig {w['n_sig']}/5 "
                  f"{'PASS' if w['pass'] else '—'} | "
                  f"cross Δ̄={c['mean_delta']:+.4f} sig {c['n_sig']}/5 "
                  f"{'PASS' if c['pass'] else '—'} | "
                  f"zero Δ={res['zero']['delta_acc']:+.4f}", flush=True)

    if args.verdict:
        v = verdict_from_runs(runs_dir)
        (args.outdir / "verdict.json").write_text(
            json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(v, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
