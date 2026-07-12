"""为 T-CLSC 造训练数据 (E, cls_a, cls_b, y)（TECHNICAL_DESIGN.md 模块 7，全项目最耗算力步骤）。

流程（照 AutoCTS++ 3.2.4；域逻辑在 autocls/labeling.py）：
1. 两级切片：HAR train 分层划源池/留出池，源池切 16 任务（动/静三档 4/4/8，
   策略经 task_type 分发，铁律 7）。留出任务只记录不造 label（M3 测 T-CLSC 用）。
2. 共享 TS2Vec 只在源池 fit（铁律 6），落盘复用（GPU 重训非确定，续跑禁止重训）。
3. 每个源任务存任务特征：windows_repr (m,T,D)（SetTF 输入——AutoCTS++ 的
   task_feature 语义，M3 与 T-CLSC 联合训练用）+ E (nz,)（固定初始化 SetTF 的
   输出，sanity/格式核对用）。
4. CLS 采样：n_shared 个全任务共用 + 每任务 n_random 个独立（generate_seeds.py
   noisy_seeds 口径），配对记 kind 供 curriculum。
5. 对每 (task, cls)：early_iters 对比预训练（铁律 2，一律 early-val 口径）+
   task 内 train/eval 分层划分 + 下游评估（经 task_type 分发）得 R'，逐条落盘
   缓存（铁律 3），--resume 无损接续。
6. 配对：δ 过滤同分噪声 + y=is_better(R'_a, R'_b, task_type)（铁律 8），
   写 pairs.jsonl + summary.json（各档样本量分布、δ 弃样比例）。

用法（在仓库根目录；全量很贵，先小规模试跑验证链路）：
    python scripts/2_generate_labels.py --max-tasks 4 --max-shared 3 --max-random 2   # 试跑
    python scripts/2_generate_labels.py --resume                                      # 全量/续跑
建议全量用脱离会话的独立进程（后台任务会被会话外因素杀掉）：
    Start-Process -FilePath D:\\Anaconda\\envs\\pytorch\\python.exe `
      -ArgumentList "scripts/2_generate_labels.py","--resume" `
      -RedirectStandardOutput results\\labels\\run.log -RedirectStandardError results\\labels\\run.err
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.contrastive.cl_trainer import pretrain_encoder  # noqa: E402
from autocls.data.datasets import load_har  # noqa: E402
from autocls.data.task_slicer import (  # noqa: E402
    assert_tasks_disjoint,
    make_leaveout_split,
    materialize_task,
)
from autocls.downstream import resolve_evaluator  # noqa: E402
from autocls.encoders.registry import build_encoder  # noqa: E402
from autocls.labeling import (  # noqa: E402
    ORIGIN_RANDOM,
    ORIGIN_SHARED,
    append_jsonl,
    cls_key,
    load_rprime_cache,
    make_pairs,
    run_seed_for,
    sample_cls_sets,
    split_task_train_eval,
)
from autocls.space.cls_space import CLSSpace  # noqa: E402
from autocls.task_embed import (  # noqa: E402
    TS2VecWrapper,
    TaskSetEncoder,
    embed_task,
    encode_task_windows,
    fit_shared_ts2vec,
    task_windows,
)


def load_dataset(name: str, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """数据集分发：返回 train 切分的 (X, y)（切片/造 label 只用 train 侧）。"""
    if name == "har":
        X, y, _, _ = load_har(data_dir / "HAR" / "har.npz")
        return X, y
    raise ValueError(f"未知 dataset: {name!r}（阶段一支持 har）")


def leak_checks(split, X: np.ndarray, ts2vec_train: np.ndarray) -> None:
    """泄漏自检（铁律 6）：两池零交集、任务样本各归其池、TS2Vec 训练数据==源池。"""
    assert len(np.intersect1d(split.source_indices, split.holdout_indices)) == 0, "两池索引重叠"
    assert_tasks_disjoint(split.source_tasks, split.holdout_tasks)
    src_set = set(split.source_indices.tolist())
    hold_set = set(split.holdout_indices.tolist())
    assert all(set(t.sample_indices) <= src_set for t in split.source_tasks), "源任务样本越池"
    assert all(set(t.sample_indices) <= hold_set for t in split.holdout_tasks), "留出任务样本越池"
    # TS2Vec 训练数据 == 源池：唯一取数途径是 X[split.source_indices]（构造保证），
    # 此处复核规模一致 + 与留出池无重叠（留出池任一样本行不出现在训练数据中）
    assert ts2vec_train.shape[0] == len(split.source_indices)
    probe = split.holdout_indices[:8]
    assert all(not np.any(np.all(ts2vec_train == X[h], axis=(1, 2))) for h in probe), \
        "留出样本混入 TS2Vec 训练数据（铁律 6）"
    print("泄漏自检：两池零交集 / 任务样本各归其池 / TS2Vec 训练数据==源池  全部通过", flush=True)


def get_shared_ts2vec(path: Path, ts2vec_train: np.ndarray, cfg: dict, device: str) -> TS2VecWrapper:
    """共享 TS2Vec：已落盘则加载（续跑一致性），否则在源池 fit 后保存。"""
    if path.exists():
        w = TS2VecWrapper.load(path, device=device)
        print(f"shared TS2Vec: 从 {path} 加载（n_iters={w.n_iters}, frozen={w.frozen}）", flush=True)
        return w
    t0 = time.time()
    w = fit_shared_ts2vec(
        ts2vec_train, n_iters=cfg["ts2vec_iters"], device=device, seed=cfg["embed_seed"],
    )
    w.save(path)
    print(f"shared TS2Vec: 源池 fit {cfg['ts2vec_iters']} iters 完成并保存 "
          f"({time.time() - t0:.0f}s) → {path}", flush=True)
    return w


def get_set_encoder(path: Path, shared: TS2VecWrapper, cfg: dict) -> TaskSetEncoder:
    """SetTF 实例：初始化权重落盘一次，续跑加载同一份（跨任务 E 可比 + M3 复用）。"""
    torch.manual_seed(cfg["embed_seed"])
    enc = TaskSetEncoder(dim_input=shared.output_dims, nz=cfg["nz"])
    if path.exists():
        enc.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    else:
        torch.save(enc.state_dict(), path)
    return enc


def ensure_task_features(
    task, X: np.ndarray, y: np.ndarray, shared: TS2VecWrapper,
    set_enc: TaskSetEncoder, cfg: dict, feat_dir: Path,
) -> Path:
    """任务特征落盘（已存在则跳过）：windows_repr (m,T,D) + E (nz,)。"""
    path = feat_dir / f"{task.task_id}.npz"
    if path.exists():
        return path
    Xt, _ = materialize_task(task, X, y)  # 标签不进编码：E 是数据分布特征
    windows = task_windows(Xt, task.task_type)  # 铁律 7 分发点：task_embed 窗口逻辑
    t0 = time.time()
    reprs = encode_task_windows(windows, shared, sample_num=cfg["sample_num"], seed=cfg["embed_seed"])
    e_vec = embed_task(
        windows, shared, sample_num=cfg["sample_num"], nz=cfg["nz"],
        set_encoder=set_enc, seed=cfg["embed_seed"],
    )
    np.savez_compressed(path, windows_repr=reprs.astype(np.float32), E=e_vec.astype(np.float32))
    print(f"  任务特征 {task.task_id}: windows_repr={reprs.shape} E={e_vec.shape} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return path


def run_one(task, cls_cfg, X, y, train_cfg: dict, labels_cfg: dict, device: str) -> dict:
    """单条 (task, cls) → R'：early_iters 预训练 + 分层 train/eval 下游评估。

    失败（NaN/异常）记 r_prime=nan 不中断整体。种子与缓存键绑定（与执行顺序无关）。
    """
    key = cls_key(cls_cfg)
    run_seed = run_seed_for(task.task_id, key)
    Xt, yt = materialize_task(task, X, y)
    # 任务内划分对该任务所有 CLS 固定（同一评估协议才可比）
    tr_idx, ev_idx = split_task_train_eval(
        yt, task.task_type, eval_ratio=labels_cfg["eval_ratio"],
        seed=run_seed_for(task.task_id, "train_eval_split"),
    )

    rec: dict = {"seconds_pretrain": None, "seconds_eval": None, "error": None}
    t0 = time.perf_counter()
    try:
        torch.manual_seed(run_seed)  # 权重初始化可复现
        encoder = build_encoder(input_dims=Xt.shape[-1])
        losses = pretrain_encoder(
            encoder, cls_cfg, Xt[tr_idx],
            max_iters=train_cfg["early_iters"],  # 铁律 2：造 label 一律 early-val
            batch_size=train_cfg["batch_size"], lr=train_cfg["lr"],
            device=device, seed=run_seed,
        )
        rec["seconds_pretrain"] = round(time.perf_counter() - t0, 1)
        rec["loss_final"] = losses[-1] if losses else None
        t1 = time.perf_counter()
        evaluator = resolve_evaluator(task.task_type)  # 铁律 7 分发点：downstream 评估器
        metrics = evaluator(
            encoder, Xt[tr_idx], yt[tr_idx], Xt[ev_idx], yt[ev_idx],
            device=device, seed=run_seed,
        )
        rec["seconds_eval"] = round(time.perf_counter() - t1, 1)
        rec.update(metrics)
    except Exception as e:  # 训练发散等视作该 CLS 的坏结果，不中断实验
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["r_prime"] = float("nan")
    rec.update({
        "n_train": int(len(tr_idx)), "n_eval": int(len(ev_idx)), "run_seed": run_seed,
    })
    return rec


def build_pairs_and_summary(
    active_tasks, sampling, cache: dict, labels_cfg: dict, train_cfg: dict, outdir: Path,
) -> None:
    """从缓存生成 pairs.jsonl（幂等重建）+ summary.json（各档分布、δ 弃样比例）。"""
    early_iters = train_cfg["early_iters"]
    pairs_path = outdir / "pairs.jsonl"
    per_task, per_tier = {}, {}
    with open(pairs_path, "w", encoding="utf-8") as fh:
        for task in active_tasks:
            cand = [(ORIGIN_SHARED, c) for c in sampling.shared] + \
                   [(ORIGIN_RANDOM, c) for c in sampling.random[task.task_id]]
            rprime, origin = {}, {}
            for org, c in cand:
                rec = cache.get((task.task_id, cls_key(c), early_iters))
                if rec is not None:
                    rprime[cls_key(c)] = rec["r_prime"] if rec["r_prime"] is not None else float("nan")
                    origin[cls_key(c)] = org
            pairs, stats = make_pairs(
                rprime, task.task_type, delta=labels_cfg["delta"], origin=origin,
            )
            for p in pairs:
                fh.write(json.dumps({
                    "task_id": task.task_id, "tag": task.tag, "task_type": task.task_type, **p,
                }, ensure_ascii=False) + "\n")
            per_task[task.task_id] = {"tag": task.tag, **stats}
            tier = per_tier.setdefault(task.tag, {k: 0 for k in stats})
            for k, v in stats.items():
                tier[k] += v

    total = {k: sum(t[k] for t in per_tier.values()) for k in next(iter(per_tier.values()))} \
        if per_tier else {}
    summary = {
        "config": labels_cfg,
        "early_iters": early_iters,
        "n_active_tasks": len(active_tasks),
        "per_task": per_task,
        "per_tier": per_tier,
        "total": total,
        "delta_drop_frac": (total.get("n_delta_dropped", 0) / total["n_candidates"])
        if total.get("n_candidates") else None,
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n配对完成 → {pairs_path}", flush=True)
    print(f"{'档位':<14}{'任务':>4}{'候选对':>8}{'δ弃样':>7}{'平局弃':>7}{'失败CLS':>8}{'产出':>7}")
    for tag, s in sorted(per_tier.items()):
        n_tasks = sum(1 for t in per_task.values() if t["tag"] == tag)
        print(f"{tag:<14}{n_tasks:>4}{s['n_candidates']:>8}{s['n_delta_dropped']:>7}"
              f"{s['n_tie_dropped']:>7}{s['n_failed_cls']:>8}{s['n_kept']:>7}")
    if total:
        print(f"{'合计':<14}{len(per_task):>4}{total['n_candidates']:>8}"
              f"{total['n_delta_dropped']:>7}{total['n_tie_dropped']:>7}"
              f"{total['n_failed_cls']:>8}{total['n_kept']:>7}")
        if total["n_candidates"]:
            print(f"δ 弃样比例: {total['n_delta_dropped'] / total['n_candidates']:.1%} "
                  f"(delta={labels_cfg['delta']})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "labels.yaml")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "labels")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="跳过缓存中已完成的 (task, cls)")
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="试跑：只处理前 N 个源任务（CLS 采样仍按全任务做，缓存与全量兼容）")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="试跑：只处理指定 task_id（覆盖 --max-tasks；可跨三档取样）")
    ap.add_argument("--max-shared", type=int, default=None, help="试跑：每任务只跑前 N 个 shared CLS")
    ap.add_argument("--max-random", type=int, default=None, help="试跑：每任务只跑前 N 个 random CLS")
    args = ap.parse_args()

    labels_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    early_iters = train_cfg["early_iters"]
    task_type = labels_cfg["task_type"]

    args.outdir.mkdir(parents=True, exist_ok=True)
    feat_dir = args.outdir / "task_features"
    feat_dir.mkdir(exist_ok=True)
    cache_path = args.outdir / "rprime_cache.jsonl"
    if cache_path.exists() and not args.resume:
        sys.exit(f"{cache_path} 已存在：加 --resume 续跑，或删除后重跑")

    # ---- 1. 数据与两级切片（策略经 task_type 分发，铁律 7）----
    X, y = load_dataset(labels_cfg["dataset"], args.data_dir)
    split = make_leaveout_split(
        y, n_source_tasks=labels_cfg["n_source_tasks"],
        n_holdout_tasks=labels_cfg["n_holdout_tasks"],
        holdout_ratio=labels_cfg["holdout_ratio"], seed=labels_cfg["slice_seed"],
        task_type=task_type,
    )
    ts2vec_train = X[split.source_indices]
    leak_checks(split, X, ts2vec_train)

    source_tasks = list(split.source_tasks)
    if args.tasks:
        by_id = {t.task_id: t for t in source_tasks}
        unknown = set(args.tasks) - set(by_id)
        if unknown:
            sys.exit(f"未知 task_id: {sorted(unknown)}（可选: {sorted(by_id)}）")
        active_tasks = [by_id[tid] for tid in args.tasks]
    else:
        active_tasks = source_tasks[: args.max_tasks] if args.max_tasks else source_tasks
    print(f"数据 {labels_cfg['dataset']}: X={X.shape} → 源任务 {len(source_tasks)} "
          f"(本次处理 {len(active_tasks)}) / 留出任务 {len(split.holdout_tasks)}（只记录不造 label）",
          flush=True)
    (args.outdir / "tasks.json").write_text(json.dumps({
        "source_tasks": [dataclasses.asdict(t) for t in split.source_tasks],
        "holdout_tasks": [dataclasses.asdict(t) for t in split.holdout_tasks],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 2/3. 共享 TS2Vec（铁律 6）+ 任务特征 ----
    shared = get_shared_ts2vec(args.outdir / "shared_ts2vec.pt", ts2vec_train, labels_cfg, args.device)
    set_enc = get_set_encoder(args.outdir / "set_encoder_init.pt", shared, labels_cfg)
    for t in active_tasks:
        ensure_task_features(t, X, y, shared, set_enc, labels_cfg, feat_dir)

    # ---- 4. CLS 采样（按全部源任务采样：试跑与全量共享同一批候选与缓存）----
    space = CLSSpace.from_yaml()
    sampling = sample_cls_sets(
        space, [t.task_id for t in source_tasks],
        n_shared=labels_cfg["n_shared_cls"], n_random=labels_cfg["n_random_cls"],
        seed=labels_cfg["cls_seed"],
    )
    (args.outdir / "cls_catalog.json").write_text(json.dumps({
        "shared": [{"cls_key": cls_key(c), "cls": dataclasses.asdict(c)} for c in sampling.shared],
        "random": {tid: [{"cls_key": cls_key(c), "cls": dataclasses.asdict(c)} for c in cs]
                   for tid, cs in sampling.random.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 5. R' 主循环（铁律 3：逐条落盘 + --resume）----
    # 先全任务×shared 再全任务×random：中途被杀时 shared 网格（跨任务信号）先保全
    by_task = {t.task_id: t for t in active_tasks}
    worklist: list[tuple[str, str, object]] = []
    for t in active_tasks:
        for c in sampling.shared[: args.max_shared]:
            worklist.append((t.task_id, ORIGIN_SHARED, c))
    for t in active_tasks:
        for c in sampling.random[t.task_id][: args.max_random]:
            worklist.append((t.task_id, ORIGIN_RANDOM, c))

    cache = load_rprime_cache(cache_path)
    todo = [(tid, org, c) for tid, org, c in worklist
            if (tid, cls_key(c), early_iters) not in cache]
    print(f"设备 {args.device} | early_iters={early_iters} bs={train_cfg['batch_size']} "
          f"delta={labels_cfg['delta']} | 计划 {len(worklist)} 条，已缓存 "
          f"{len(worklist) - len(todo)}，待跑 {len(todo)}", flush=True)

    for n_done, (tid, org, c) in enumerate(todo):
        task, key = by_task[tid], cls_key(c)
        rec = run_one(task, c, X, y, train_cfg, labels_cfg, args.device)
        rec = {"task_id": tid, "tag": task.tag, "task_type": task.task_type,
               "cls_key": key, "origin": org, "iters": early_iters,
               **rec, "cls": dataclasses.asdict(c)}
        if "r_prime" in rec and not np.isfinite(rec["r_prime"]):
            rec["r_prime"] = None  # JSON 不认 NaN，落盘用 null（load 侧还原）
        append_jsonl(cache_path, rec)
        cache[(tid, key, early_iters)] = rec
        shown = rec["r_prime"]
        status = f"R'={shown:.4f}" if shown is not None else f"失败: {rec['error']}"
        print(f"[{n_done + 1}/{len(todo)}] {tid} {org}:{key} {status} "
              f"(pretrain {rec.get('seconds_pretrain')}s, eval {rec.get('seconds_eval')}s)",
              flush=True)

    # ---- 6. 配对 + 汇总（幂等重建，跑到哪配到哪）----
    build_pairs_and_summary(active_tasks, sampling, cache, labels_cfg, train_cfg, args.outdir)


if __name__ == "__main__":
    main()
