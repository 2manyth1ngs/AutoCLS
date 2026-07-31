"""路线 A 阶段2：跨源池造 T-CLSC 训练数据 (E, cls_a, cls_b, y)（checklist C2）。

与路线 B（scripts/2_generate_labels.py，HAR 单数据集）的差异——除此之外骨架同款：
1. **多源数据集**：最终源池 = tested 骨干 ∪ step0 analysis.json 通过闸者（含天花板
   重探恢复者，沿用其 recommended_n_classes_range 档，任务 id 带 +c 档标 →
   与 step0 重探 canonical 缓存逐键复用）；切片走 slice_source_tasks（注册表唯一
   真源，与 step0 同口径，铁律 3）。
2. **per-dataset δ**：配对 δ 按任务所属数据集查 labels.yaml: delta_per_dataset
   （delta_for，铁律 8 之外的可靠性口径——δ=0.02 是 HAR 特调，UEA 噪声 4× 跨度）。
3. **CLS 口径 20/任务**（labels.yaml: route_a）：shared 10 = step0 探针目录前 10
   （canonical R' 逐字段命中复用）；random 10/任务由 task_id 派生
   （sample_random_cls——池成员浮动不影响缓存，铁律 3）。
4. **共享 TS2Vec（铁律 6）**：只在最终源池 fit（各数据集单变量实例 NaN 填充对齐、
   每数据集封顶采样）；落盘 + fit 池指纹（shared_ts2vec.meta.json），续跑校验池
   一致才加载，绝不重训。目标池零参与：源池∩目标池断言 + 目标池加载 fail-fast 自检。
5. **看门狗协议（Part A-3）**：每条 R' 开跑前写 inflight.txt、落盘后清空，
   供 scripts/cloud_run.sh 统计毒条目并永久隔离。

输出（results/labels_route_a/，格式对齐路线 B，供 3_pretrain_tclsc 直接训）：
rprime_cache.jsonl / pairs.jsonl / summary.json / shared_ts2vec.pt +
shared_ts2vec.meta.json / set_encoder_init.pt / task_features/*.npz /
tasks.json / cls_catalog.json

用法（全量很贵，先小规模验证链路）：
    python scripts/2b_generate_labels_route_a.py --datasets NATOPS ERing \
        --max-tasks 1 --max-shared 2 --max-random 2 --outdir results/labels_route_a_smoke
    python scripts/2b_generate_labels_route_a.py --resume            # 全量/续跑（上云）
云上脱离进程 + 自动重拉：STAGE=labels nohup bash scripts/cloud_run.sh & （Part A-3）
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

from autocls.data.source_pool import (  # noqa: E402
    ROLE_TARGET,
    TargetPoolLeakageError,
    dataset_of_task_id,
    delta_for,
    load_registry,
    load_source_dataset,
    slice_source_tasks,
)
from autocls.data.task_slicer import materialize_task  # noqa: E402
from autocls.labeling import (  # noqa: E402
    ORIGIN_RANDOM,
    ORIGIN_SHARED,
    append_jsonl,
    clear_inflight,
    cls_key,
    evaluate_rprime,
    load_probe_cls,
    load_rprime_cache,
    make_pairs,
    mark_inflight,
    run_seed_for,
    sample_random_cls,
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

PROBE_CATALOG = ROOT / "results" / "route_a_probe" / "probe_cls_catalog.json"
DEFAULT_ANALYSIS = ROOT / "results" / "step0_snr_gate" / "analysis.json"


def canonical_seeds(task_id: str, key: str) -> tuple[int, int]:
    """全量 canonical 口径 (weight_seed, split_seed)——与 step0 _seed_specs 的 s=0
    完全同源（tests/test_labeling_route_a 断言同源），缓存逐键命中复用的前提。"""
    return run_seed_for(task_id, key), run_seed_for(task_id, "train_eval_split")


def resolve_pool(registry, analysis: dict | None, explicit: list[str] | None):
    """最终源池 + 每数据集生效类别档（None=注册表默认档）。

    默认 = tested 骨干 ∪ analysis.json passed（天花板恢复者用 recommended 档）；
    --datasets 显式覆盖（小规模验证/分片跑），此时仍应用 analysis 里的恢复档。
    """
    ranges: dict[str, tuple[int, int] | None] = {}
    if explicit:
        names = list(explicit)
    else:
        if analysis is None:
            sys.exit("缺 step0 analysis.json：先跑 scripts/step0_snr_gate.py --analyze，"
                     "或用 --datasets 显式指定（小规模验证）")
        names = [e.name for e in registry.source if e.tested]
        for ds, d in analysis.get("datasets", {}).items():
            if d.get("passed") and ds not in names:
                names.append(ds)
    for name in names:
        entry = registry.get(name)  # 不在注册表 → KeyError fail-fast
        rng = None
        if analysis is not None:
            d = analysis.get("datasets", {}).get(name)
            if d and d.get("passed") and d.get("recommended_n_classes_range"):
                rec = tuple(d["recommended_n_classes_range"])
                if rec != entry.n_classes_range:
                    rng = rec  # 天花板恢复档：阶段2 沿用更难档（Part A-1）
        ranges[name] = rng
    return names, ranges


def leak_checks(registry, pool: list[str]) -> None:
    """泄漏自检（铁律 6）：源池∩目标池=∅ + 目标池走源加载路径必须 fail-fast。"""
    targets = sorted(registry.names(ROLE_TARGET))
    overlap = set(pool) & set(targets)
    assert not overlap, f"最终源池混入目标池 {overlap}（铁律 6）"
    for name in targets:
        try:
            load_source_dataset(name, registry)
        except TargetPoolLeakageError:
            continue
        except ValueError:
            continue  # external 目标（UCI-HAR）：目标闸在 external 闸之前，此分支不该到
        raise AssertionError(f"目标池 {name} 未被源加载路径拒绝（铁律 6 防护失效）")
    print(f"泄漏自检：源池∩目标池=∅；目标池 {targets} 走源加载路径全部 fail-fast（铁律 6）",
          flush=True)


def build_ts2vec_pool(per_ds: list, max_per_ds: int, seed: int) -> np.ndarray:
    """源池 → 共享 TS2Vec 训练用单变量实例池 (M, T_max, 1)。

    各数据集 (N,T,C) 通道展开成 (N*C, T)、每数据集封顶采样（种子由数据集名派生，
    确定性），跨数据集 NaN 右填充对齐到池内最长 T。用 NaN 而非零：零填充的"通道
    均值中性"语义只在各数据集自身 z-score 内成立，跨数据集拼接时 NaN 显式表示
    "无观测"，吃 TSEncoder 原生 nan_mask（ts2vec_wrapper.py ← encoder.py 原样搬运
    处），与原 TS2Vec 处理 UEA 变长的口径一致。
    """
    arrs = []
    for name, X, _y, _tasks in per_ds:
        n, t, c = X.shape
        uni = X.transpose(0, 2, 1).reshape(n * c, t)
        m = min(max_per_ds, len(uni))
        rng = np.random.default_rng([seed, run_seed_for(name, "ts2vec_pool")])
        idx = np.sort(rng.choice(len(uni), size=m, replace=False))
        arrs.append(uni[idx])
    t_max = max(a.shape[1] for a in arrs)
    pool = np.full((sum(len(a) for a in arrs), t_max, 1), np.nan, dtype=np.float32)
    o = 0
    for a in arrs:
        pool[o:o + len(a), :a.shape[1], 0] = a
        o += len(a)
    return pool


def get_shared_ts2vec(outdir: Path, per_ds: list, pool: list[str], ra: dict,
                      device: str) -> TS2VecWrapper:
    """共享 TS2Vec：已落盘则校验 fit 池指纹后加载（续跑禁止重训），否则源池 fit。

    指纹校验防"半池预训"口径漂移：小规模验证/分片跑留下的 shared_ts2vec.pt 若 fit
    池 ≠ 当前最终源池，加载即静默泄漏错误口径——直接 fail-fast 要求人工处置。
    """
    path, meta_path = outdir / "shared_ts2vec.pt", outdir / "shared_ts2vec.meta.json"
    if path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        if meta is None or meta.get("fit_datasets") != sorted(pool):
            sys.exit(f"{path} 的 fit 池 {meta and meta.get('fit_datasets')} ≠ 当前最终源池 "
                     f"{sorted(pool)}：口径漂移（铁律 6/续跑一致性）。确认池组成后删除该文件重 fit。")
        w = TS2VecWrapper.load(path, device=device)
        print(f"shared TS2Vec: 从 {path} 加载（n_iters={w.n_iters}, frozen={w.frozen}, "
              f"fit 池 {len(meta['fit_datasets'])} 数据集）", flush=True)
        return w
    data = build_ts2vec_pool(per_ds, ra["ts2vec_max_instances_per_dataset"], ra["embed_seed"])
    print(f"shared TS2Vec: 源池 {len(per_ds)} 数据集 → {data.shape} 单变量实例池，"
          f"fit {ra['ts2vec_iters']} iters ...", flush=True)
    t0 = time.time()
    w = fit_shared_ts2vec(data, n_iters=ra["ts2vec_iters"], device=device, seed=ra["embed_seed"])
    w.save(path)
    meta_path.write_text(json.dumps({
        "fit_datasets": sorted(pool), "n_instances": int(data.shape[0]),
        "t_max": int(data.shape[1]), "n_iters": ra["ts2vec_iters"],
        "max_instances_per_dataset": ra["ts2vec_max_instances_per_dataset"],
        "embed_seed": ra["embed_seed"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"shared TS2Vec: fit 完成并保存 ({time.time() - t0:.0f}s) → {path}", flush=True)
    return w


def get_set_encoder(path: Path, shared: TS2VecWrapper, ra: dict) -> TaskSetEncoder:
    """SetTF 实例：初始化权重落盘一次，续跑加载同一份（跨任务 E 可比 + M3 复用）。"""
    torch.manual_seed(ra["embed_seed"])
    enc = TaskSetEncoder(dim_input=shared.output_dims, nz=ra["nz"])
    if path.exists():
        enc.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    else:
        torch.save(enc.state_dict(), path)
    return enc


def ensure_task_features(task, X, y, shared, set_enc, ra: dict, feat_dir: Path) -> Path:
    """任务特征落盘（已存在则跳过）：windows_repr (m,T,D) + E (nz,)。"""
    path = feat_dir / f"{task.task_id}.npz"
    if path.exists():
        return path
    Xt, _ = materialize_task(task, X, y)  # 标签不进编码：E 是数据分布特征
    windows = task_windows(Xt, task.task_type)  # 铁律 7 分发点：task_embed 窗口逻辑
    t0 = time.time()
    reprs = encode_task_windows(windows, shared, sample_num=ra["sample_num"], seed=ra["embed_seed"])
    e_vec = embed_task(windows, shared, sample_num=ra["sample_num"], nz=ra["nz"],
                       set_encoder=set_enc, seed=ra["embed_seed"])
    np.savez_compressed(path, windows_repr=reprs.astype(np.float32), E=e_vec.astype(np.float32))
    print(f"  任务特征 {task.task_id}: windows_repr={reprs.shape} E={e_vec.shape} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return path


def patch_missing_origin(cache: dict, cache_path: Path, tid: str, key: str,
                         iters: int, origin: str) -> None:
    """step0 canonical 记录无 origin 字段（step0 不分 shared/random）——补一条带
    origin 的同键记录（append-only，load_rprime_cache 同键后行覆盖前行），
    3_pretrain 的 comparator_records 才能吃。幂等：已带 origin 不再追加。"""
    rec = cache.get((tid, key, iters))
    if rec is not None and "origin" not in rec:
        patched = {**rec, "origin": origin}
        append_jsonl(cache_path, patched)
        cache[(tid, key, iters)] = patched


def build_pairs_and_summary(task_candidates: list, cache: dict, labels_cfg: dict,
                            early_iters: int, outdir: Path) -> None:
    """缓存 → pairs.jsonl（幂等重建）+ summary.json。δ 按任务所属数据集查表
    （delta_for——与路线 B 单 labels_cfg['delta'] 的唯一差异，方案 A-2 主流程 6）。"""
    pairs_path = outdir / "pairs.jsonl"
    per_task: dict = {}
    per_dataset: dict = {}
    with open(pairs_path, "w", encoding="utf-8") as fh:
        for task, cand in task_candidates:
            rprime, origin = {}, {}
            for org, _c, key in cand:
                rec = cache.get((task.task_id, key, early_iters))
                if rec is not None:
                    rprime[key] = rec["r_prime"] if rec["r_prime"] is not None else float("nan")
                    origin[key] = org
            ds = dataset_of_task_id(task.task_id)
            delta = delta_for(ds, labels_cfg)
            pairs, stats = make_pairs(rprime, task.task_type, delta=delta, origin=origin)
            for p in pairs:
                fh.write(json.dumps({
                    "dataset": ds, "task_id": task.task_id, "tag": task.tag,
                    "task_type": task.task_type, **p,
                }, ensure_ascii=False) + "\n")
            per_task[task.task_id] = {"dataset": ds, "delta": delta, **stats}
            agg = per_dataset.setdefault(ds, {"delta": delta, "n_tasks": 0,
                                              **{k: 0 for k in stats}})
            agg["n_tasks"] += 1
            for k, v in stats.items():
                agg[k] += v

    total = {k: sum(d[k] for d in per_dataset.values())
             for k in ("n_cls", "n_failed_cls", "n_candidates", "n_delta_dropped",
                       "n_tie_dropped", "n_kept")} if per_dataset else {}
    summary = {
        "route": "A",
        "early_iters": early_iters,
        "n_datasets": len(per_dataset),
        "n_tasks": len(per_task),
        "per_task": per_task,
        "per_dataset": per_dataset,
        "total": total,
        "delta_drop_frac": (total.get("n_delta_dropped", 0) / total["n_candidates"])
        if total.get("n_candidates") else None,
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n配对完成 → {pairs_path}", flush=True)
    print(f"{'数据集':<26}{'δ':>6}{'任务':>4}{'候选对':>8}{'δ弃样':>7}{'平局弃':>7}{'失败CLS':>8}{'产出':>7}")
    for ds, s in sorted(per_dataset.items()):
        print(f"{ds:<26}{s['delta']:>6}{s['n_tasks']:>4}{s['n_candidates']:>8}"
              f"{s['n_delta_dropped']:>7}{s['n_tie_dropped']:>7}{s['n_failed_cls']:>8}{s['n_kept']:>7}")
    if total:
        print(f"{'合计':<26}{'':>6}{len(per_task):>4}{total['n_candidates']:>8}"
              f"{total['n_delta_dropped']:>7}{total['n_tie_dropped']:>7}"
              f"{total['n_failed_cls']:>8}{total['n_kept']:>7}")
        if total["n_candidates"]:
            print(f"δ 弃样比例: {total['n_delta_dropped'] / total['n_candidates']:.1%}"
                  f"（per-dataset δ）", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "labels.yaml")
    ap.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS,
                    help="step0 分析结果（最终源池 + 天花板恢复档的来源）")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "labels_route_a")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="跳过缓存中已完成的 (task, cls)")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="显式指定源数据集（小规模验证/多卡分片；默认 tested 骨干 ∪ analysis 通过者）")
    ap.add_argument("--max-tasks", type=int, default=None, help="试跑：每数据集只处理前 N 个任务")
    ap.add_argument("--max-shared", type=int, default=None, help="试跑：每任务只跑前 N 个 shared CLS")
    ap.add_argument("--max-random", type=int, default=None, help="试跑：每任务只跑前 N 个 random CLS")
    args = ap.parse_args()

    labels_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    ra = labels_cfg["route_a"]
    early_iters = train_cfg["early_iters"]  # 铁律 2：造 label 一律 early-val 口径

    args.outdir.mkdir(parents=True, exist_ok=True)
    feat_dir = args.outdir / "task_features"
    feat_dir.mkdir(exist_ok=True)
    cache_path = args.outdir / "rprime_cache.jsonl"
    inflight_path = args.outdir / "inflight.txt"
    if cache_path.exists() and not args.resume:
        sys.exit(f"{cache_path} 已存在：加 --resume 续跑，或删除后重跑")

    # ---- 1. 最终源池（tested ∪ step0 通过者）+ 泄漏自检（铁律 6）----
    registry = load_registry()
    analysis = (json.loads(args.analysis.read_text(encoding="utf-8"))
                if args.analysis.exists() else None)
    pool, ranges = resolve_pool(registry, analysis, args.datasets)
    leak_checks(registry, pool)

    # ---- 2. 逐源数据集：加载 + 切全部 planned_tasks（与 step0 同口径，铁律 3）----
    per_ds = []
    active_tasks = []
    for name in pool:
        entry = registry.get(name)
        if name not in labels_cfg.get("delta_per_dataset", {}):
            print(f"⚠️ [{name}] 不在 delta_per_dataset，δ 回退默认 {labels_cfg['delta']}"
                  f"（reserve 回填应先补 δ，checklist C1）", flush=True)
        X, y = load_source_dataset(name, registry)
        eff, tasks = slice_source_tasks(entry, registry, y, n_classes_range=ranges[name])
        acts = tasks[: args.max_tasks] if args.max_tasks else tasks
        rng_note = (f" 恢复档 c{eff.n_classes_range[0]}-{eff.n_classes_range[1]}"
                    if ranges[name] else "")
        print(f"[{name}] X={X.shape} planned={entry.planned_tasks}{rng_note} "
              f"→ 本次处理 {len(acts)} tasks", flush=True)
        per_ds.append((name, X, y, acts))
        active_tasks += acts
    (args.outdir / "tasks.json").write_text(json.dumps({
        "pool": pool,
        "n_classes_range_used": {n: list(ranges[n]) if ranges[n] else None for n in pool},
        "source_tasks": [dataclasses.asdict(t) for t in active_tasks],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 3. 共享 TS2Vec（铁律 6：只在最终源池 fit）+ 任务特征 ----
    shared = get_shared_ts2vec(args.outdir, per_ds, pool, ra, args.device)
    set_enc = get_set_encoder(args.outdir / "set_encoder_init.pt", shared, ra)
    for name, X, y, tasks in per_ds:
        for t in tasks:
            ensure_task_features(t, X, y, shared, set_enc, ra, feat_dir)

    # ---- 4. CLS：shared = 探针目录前 10（canonical 复用）+ random 10/任务 ----
    space = CLSSpace.from_yaml()
    shared_cls = load_probe_cls(PROBE_CATALOG, ra["n_shared_cls"])[0]
    shared_keys = {k for _, k in shared_cls}
    random_cls = {t.task_id: [(c, cls_key(c)) for c in sample_random_cls(
        space, t.task_id, n=ra["n_random_cls"], seed=ra["cls_seed"],
        exclude_keys=shared_keys)] for t in active_tasks}
    (args.outdir / "cls_catalog.json").write_text(json.dumps({
        "shared": [{"cls_key": k, "cls": dataclasses.asdict(c)} for c, k in shared_cls],
        "random": {tid: [{"cls_key": k, "cls": dataclasses.asdict(c)} for c, k in cs]
                   for tid, cs in random_cls.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 5. R' 主循环（铁律 3：逐条落盘 + --resume；先全 shared 后全 random）----
    task_candidates = []   # [(task, [(origin, cls, key)])] —— 配对阶段复用
    by_task_ds: dict[str, tuple] = {}
    for name, X, y, tasks in per_ds:
        for t in tasks:
            cand = [(ORIGIN_SHARED, c, k) for c, k in shared_cls[: args.max_shared]] + \
                   [(ORIGIN_RANDOM, c, k) for c, k in random_cls[t.task_id][: args.max_random]]
            task_candidates.append((t, cand))
            by_task_ds[t.task_id] = (name, X, y)
    worklist = [(t, org, c, key) for org_want in (ORIGIN_SHARED, ORIGIN_RANDOM)
                for t, cand in task_candidates for org, c, key in cand if org == org_want]

    cache = load_rprime_cache(cache_path)
    for t, org, c, key in worklist:  # step0 canonical 复用：补 origin 字段（幂等）
        patch_missing_origin(cache, cache_path, t.task_id, key, early_iters, org)
    todo = [(t, org, c, key) for t, org, c, key in worklist
            if (t.task_id, key, early_iters) not in cache]
    print(f"设备 {args.device} | early_iters={early_iters} bs={train_cfg['batch_size']} "
          f"| 计划 {len(worklist)} 条，已缓存 {len(worklist) - len(todo)}"
          f"（含 step0 canonical 复用），待跑 {len(todo)}", flush=True)

    bs, lr, evr = train_cfg["batch_size"], train_cfg["lr"], labels_cfg["eval_ratio"]
    for n_done, (t, org, c, key) in enumerate(todo):
        name, X, y = by_task_ds[t.task_id]
        w_seed, sp_seed = canonical_seeds(t.task_id, key)
        mark_inflight(inflight_path, t.task_id, key, early_iters)  # 看门狗毒条目协议
        rec = evaluate_rprime(t, c, X, y, weight_seed=w_seed, split_seed=sp_seed,
                              iters=early_iters, eval_ratio=evr, batch_size=bs, lr=lr,
                              device=args.device)
        rp = rec["r_prime"]
        full = {"dataset": name, "task_id": t.task_id, "tag": t.tag,
                "task_type": t.task_type, "classes": list(t.classes),
                "cls_key": key, "origin": org, "iters": early_iters,
                **rec, "r_prime": None if not np.isfinite(rp) else float(rp),
                "cls": dataclasses.asdict(c)}
        append_jsonl(cache_path, full)
        clear_inflight(inflight_path)
        cache[(t.task_id, key, early_iters)] = full
        st = (f"R'={full['r_prime']:.4f}" if full["r_prime"] is not None
              else f"失败: {rec['error']}")
        print(f"[{n_done + 1}/{len(todo)}] {name}/{t.task_id} {org}:{key[:8]} {st} "
              f"({rec['seconds']}s, n_ev={rec.get('n_eval')})", flush=True)

    # ---- 6. 配对 + 汇总（per-dataset δ；幂等重建，跑到哪配到哪）----
    build_pairs_and_summary(task_candidates, cache, labels_cfg, early_iters, args.outdir)


if __name__ == "__main__":
    main()
