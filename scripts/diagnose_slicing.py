"""切片管线诊断（不跑训练）：重叠窗泄漏 / 划分方式 / segmented 对照。

背景：第3步 7/7 判 inert（`results/candidate_survey/REPORT_STEP3.md`）。在做
full-iters 判别实验前，先排除"问题出在切片管线本身"这个更靠上游的嫌疑。

本脚本回答三件事，全部是**精确计数**而非估计：

1. **重叠孪生**（`--twins`）：流式候选按 win=128/stride=64 开窗（`candidates.py`），
   窗 i 与窗 i+1 共享 64 步**同一批原始采样**。驱动的按通道 z-score 是全局仿射变换
   （`prepare_candidates.py` 对整个数组算一次 mean/std），故这 64 步在两个窗里
   **逐位相同**——`X[i, 64:] == X[i+1, :64]` 可以逐字节判定，不需要任何相似度阈值。
   于是"哪些窗互为孪生"是可精确枚举的：对每个窗算 head/tail 两个半窗的哈希，
   tail 哈希表里查 head 即得全部孪生边。O(N)。

2. **任务级泄漏**（`--task-leak`）：孪生存在 ≠ 泄漏。真正的问题是
   `split_task_train_eval` 对分类走**按类分层的纯随机划分**（`labeling.py`），
   它不知道窗口重叠——于是一对孪生可能一个进 train、一个进 eval。本步复现调研
   实际用的任务与 canonical split（同 `baseline_probe` 口径），逐任务数出
   "eval 样本里有多少个在 train 侧存在孪生"。

3. **segmented 对照**：已切好样本的档案（UniMiB/UCR/UEA…）不走滑窗，应当零孪生。
   "走滑窗的有泄漏、不走滑窗的没有"若成立，就是把嫌疑指向切片管线的强证据。

4. **泄漏归因**（`--leak-decompose`）：泄漏率本身不是结论——"eval 里 5% 的样本在
   train 有孪生"要变成"基线因此虚高了多少"，还得知道那 5% 是不是真的白送分。
   本步跑**一模一样的基线协议**（iters=0、随机初始化 encoder、协议 A 的 SVM-RBF、
   canonical 划分），只是把 eval 准确率拆成**泄漏子集**与**非泄漏子集**两半。
   不改划分、不新建缓存，因此没有 task_id 错配风险（crop 那个坑）。
   判据先于测量写死，见 `LEAK_GAP_FLOOR` / `LEAK_ATTRIB_*`。

用法：
    python scripts/diagnose_slicing.py --twins
    python scripts/diagnose_slicing.py --task-leak
    python scripts/diagnose_slicing.py --leak-decompose --datasets WISDM2019 PAMAP2
    python scripts/diagnose_slicing.py --twins --task-leak --out results/slicing_diagnosis

**不加载目标池**（铁律 6）：数据集清单只取注册表的 source/reserve 两角色，
`load_source_dataset` 对目标池 fail-fast 是第二道闸。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):        # 本机控制台默认 GBK
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.baseline_probe import baseline_seed_specs, probe_scope  # noqa: E402
from autocls.data.candidates import (  # noqa: E402
    CANDIDATE_ROOT,
    WINDOW_LEN,
    WINDOW_STRIDE,
)
from autocls.data.source_pool import (  # noqa: E402
    load_registry,
    load_source_dataset,
    slice_source_tasks,
)
from autocls.data.task_slicer import materialize_task  # noqa: E402
from autocls.labeling import split_task_train_eval  # noqa: E402

CANDIDATE_REGISTRY = ROOT / "configs" / "candidate_pool.yaml"
UEA_REGISTRY = ROOT / "configs" / "source_pool.yaml"
EVAL_RATIO = 0.3          # configs/labels.yaml，与 evaluate_rprime 默认一致
DEFAULT_OUT = ROOT / "results" / "slicing_diagnosis"

# §4.1 步骤3 只用前 2 个任务；基线用全部 planned_tasks。两个口径都报。
STEP3_TASKS = 2

# —— 泄漏归因判据（**先于测量写死**，延续协议锁惯例）——
# 归因抬升 = 泄漏比例 × (泄漏子集 acc − 非泄漏子集 acc)：把"被白送的分"从基线里扣掉。
LEAK_GAP_FLOOR = 0.15      # 两子集 acc 差 ≥ 此值才认为泄漏样本确实在白送分
LEAK_ATTRIB_MAJOR = 0.05   # 归因抬升 ≥ 典型 δ → 泄漏显著，之前的基线数字要重算
LEAK_ATTRIB_MINOR = 0.02   # 0.02–0.05 → 有影响但非贴顶主因；<0.02 → 可忽略
LEAK_MIN_SUBSET = 20       # 子集样本数下限：低于此不出判定（比例噪声压不住）


def _digest(a: np.ndarray) -> bytes:
    """半窗 → 16 字节摘要。逐字节哈希（不是特征哈希）：孪生半窗逐位相同，
    非孪生半窗碰撞概率 ~2^-128，故等价于精确比较。"""
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=16).digest()


def candidate_meta(name: str) -> dict:
    p = CANDIDATE_ROOT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def twin_edges(X: np.ndarray, half: int) -> tuple[dict[int, list[int]], dict]:
    """全部"半窗重叠"孪生边：i 的后 half 步 == j 的前 half 步 → i 与 j 是孪生。

    返回 (邻接表, 统计)。邻接表是**无向**的（i→j 与 j→i 都记），因为泄漏问题
    不分方向：只要一对孪生分居 train/eval 两侧就构成泄漏。

    退化窗（自身 head==tail，即整窗常值）单列计数：它们会与任何同样常值的窗
    匹配上，是真重复而非开窗造成的重叠，混在一起会高估滑窗的责任。
    """
    n = X.shape[0]
    heads = [_digest(X[i, :half]) for i in range(n)]
    tails = [_digest(X[i, -half:]) for i in range(n)]

    by_tail: dict[bytes, list[int]] = {}
    for i, t in enumerate(tails):
        by_tail.setdefault(t, []).append(i)

    adj: dict[int, list[int]] = {}
    n_degenerate = sum(1 for i in range(n) if heads[i] == tails[i])
    n_edges = 0
    for j, h in enumerate(heads):
        for i in by_tail.get(h, ()):
            if i == j:
                continue
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)
            n_edges += 1

    n_with_twin = len(adj)
    return adj, {
        "n_windows": n,
        "half": half,
        "n_twin_edges": n_edges,
        "n_windows_with_twin": n_with_twin,
        "frac_windows_with_twin": n_with_twin / n if n else 0.0,
        "n_degenerate_constant_windows": n_degenerate,
    }


def scan_twins(names: list[str], registry, *, out_dir: Path) -> list[dict]:
    rows = []
    for name in names:
        t0 = time.perf_counter()
        try:
            X, _y = load_source_dataset(name, registry)
        except Exception as e:  # 某个候选缺文件不该拖垮整批诊断
            print(f"[{name}] 跳过：{type(e).__name__}: {e}", flush=True)
            continue
        meta = candidate_meta(name)
        segmented = bool(meta.get("segmented", False))
        native = meta.get("source", registry.get(name).source)
        T = X.shape[1]
        # 流式候选的重叠量由预注册口径固定（win-stride=64）；segmented/UEA 不开窗，
        # 用 T//2 做同一把尺子量它们（预期 0，正是对照要证的）。
        half = (WINDOW_LEN - WINDOW_STRIDE) if (T == WINDOW_LEN and not segmented) else T // 2
        adj, stat = twin_edges(X, half)
        row = {
            "dataset": name, "source": native, "segmented": segmented,
            "windowed": (not segmented) and native == "npz" and T == WINDOW_LEN,
            "n_samples": int(X.shape[0]), "seq_len": T, "n_channels": int(X.shape[2]),
            "subsampled": bool(meta.get("subsampled", False)),
            "n_samples_raw": meta.get("n_samples_raw"),
            **stat, "seconds": round(time.perf_counter() - t0, 1),
        }
        rows.append(row)
        print(f"[{name}] N={row['n_samples']} L={T} 孪生窗 "
              f"{stat['n_windows_with_twin']}/{stat['n_windows']} "
              f"({stat['frac_windows_with_twin']:.1%}) 退化窗 "
              f"{stat['n_degenerate_constant_windows']} {row['seconds']}s", flush=True)
        del X, adj
        gc.collect()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "twins.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def task_leakage(names: list[str], registry, *, out_dir: Path) -> list[dict]:
    """复现调研实际用的任务与 canonical 划分，逐任务数 eval→train 的孪生泄漏。"""
    rows = []
    for name in names:
        try:
            X, y = load_source_dataset(name, registry)
        except Exception as e:
            print(f"[{name}] 跳过：{type(e).__name__}: {e}", flush=True)
            continue
        meta = candidate_meta(name)
        segmented = bool(meta.get("segmented", False))
        T = X.shape[1]
        half = (WINDOW_LEN - WINDOW_STRIDE) if (T == WINDOW_LEN and not segmented) else T // 2
        adj, _stat = twin_edges(X, half)
        entry = registry.get(name)
        _eff, tasks = slice_source_tasks(entry, registry, y)

        for ti, task in enumerate(tasks):
            _Xt, yt = materialize_task(task, X, y)
            split_seed = baseline_seed_specs(task.task_id, 0)[1]   # canonical 划分
            tr_idx, ev_idx = split_task_train_eval(
                yt, task.task_type, eval_ratio=EVAL_RATIO, seed=split_seed)
            glob = np.asarray(task.sample_indices)
            train_g, eval_g = set(glob[tr_idx].tolist()), set(glob[ev_idx].tolist())
            n_leak = sum(1 for e in eval_g if any(t in train_g for t in adj.get(e, ())))
            n_twin_within = sum(1 for e in eval_g if any(t in eval_g for t in adj.get(e, ())))
            rows.append({
                "dataset": name, "task_id": task.task_id, "task_index": ti,
                "in_step3": ti < STEP3_TASKS,
                "n_train": len(train_g), "n_eval": len(eval_g),
                "n_eval_with_twin_in_train": n_leak,
                "frac_eval_leaked": n_leak / len(eval_g) if eval_g else 0.0,
                "n_eval_with_twin_in_eval": n_twin_within,
            })
        done = [r for r in rows if r["dataset"] == name]
        fr = [r["frac_eval_leaked"] for r in done]
        print(f"[{name}] {len(done)} 任务，eval 泄漏率 "
              f"均值 {np.mean(fr):.1%} 范围 {min(fr):.1%}–{max(fr):.1%}", flush=True)
        del X, adj
        gc.collect()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_leakage.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def leak_decompose(
    names: list[str], registry, *, out_dir: Path, n_seeds: int = 5,
    max_tasks: int = STEP3_TASKS, device: str | None = None,
) -> list[dict]:
    """基线协议原样重跑，把 eval acc 拆成泄漏 / 非泄漏两子集（判据见文件头 §4）。

    与 `baseline_probe` 逐项同口径：iters=0（encoder 停在随机初始化、CLS 不参与）、
    协议 A 的 max-pool + SVM-RBF、canonical 划分、weight_seed 走 baseline_seed_specs。
    唯一的差别是保留了逐样本预测——所以这里的 overall acc 应当**复现**已有基线数字，
    不复现就说明本脚本的口径接错了（报告里要对账）。

    附带 raw-1NN：把展平的原始窗口直接做最近邻分类。它不经过任何 encoder，是
    "这个任务能不能靠模式匹配解掉"的下界参照。
    """
    import torch
    from joblib import parallel_backend
    from sklearn.metrics import accuracy_score

    from autocls.downstream.linear_eval import encode_features, fit_svm
    from autocls.encoders.registry import build_encoder

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    # threading 后端：Windows 上 loky 每条 spawn 进程池占 99% 耗时，且逐位同值
    # （linear_eval 注释 + tests/test_baseline_probe.py 已记明）
    with parallel_backend("threading", n_jobs=1):
        for name in names:
            try:
                X, y = load_source_dataset(name, registry)
            except Exception as e:
                print(f"[{name}] 跳过：{type(e).__name__}: {e}", flush=True)
                continue
            meta = candidate_meta(name)
            segmented = bool(meta.get("segmented", False))
            T = X.shape[1]
            half = (WINDOW_LEN - WINDOW_STRIDE) if (T == WINDOW_LEN and not segmented) else T // 2
            adj, _ = twin_edges(X, half)
            entry = registry.get(name)
            _eff, tasks = slice_source_tasks(entry, registry, y)

            for task in tasks[:max_tasks]:
                Xt, yt = materialize_task(task, X, y)
                split_seed = baseline_seed_specs(task.task_id, 0)[1]
                tr_idx, ev_idx = split_task_train_eval(
                    yt, task.task_type, eval_ratio=EVAL_RATIO, seed=split_seed)
                glob = np.asarray(task.sample_indices)
                train_g = set(glob[tr_idx].tolist())
                leaked = np.array([
                    any(t in train_g for t in adj.get(int(g), ()))
                    for g in glob[ev_idx]])

                # raw-1NN：展平原始窗口的最近邻，不经 encoder
                ftr = Xt[tr_idx].reshape(len(tr_idx), -1)
                fev = Xt[ev_idx].reshape(len(ev_idx), -1)
                # 展开式 ‖a−b‖² = ‖a‖² − 2a·b + ‖b‖²，省掉 (n_ev, n_tr, d) 的广播中间量
                # （307×717×768 要 675 MB）；‖a‖² 对同一行是常数，argmin 不受影响故略去。
                nn = np.argmin(-2 * (fev @ ftr.T) + (ftr ** 2).sum(1)[None, :], axis=1)
                nn_pred = yt[tr_idx][nn]

                for s in range(n_seeds):
                    w_seed, _ = baseline_seed_specs(task.task_id, s)
                    torch.manual_seed(w_seed)
                    enc = build_encoder(input_dims=Xt.shape[-1])
                    tr_f = encode_features(enc, Xt[tr_idx], device=device)
                    ev_f = encode_features(enc, Xt[ev_idx], device=device)
                    pred = fit_svm(tr_f, yt[tr_idx], seed=w_seed).predict(ev_f)
                    ok = (pred == yt[ev_idx])
                    rows.append({
                        "dataset": name, "task_id": task.task_id, "seed": s,
                        "n_eval": int(len(ev_idx)),
                        "n_leaked": int(leaked.sum()),
                        "frac_leaked": float(leaked.mean()),
                        "acc_all": float(ok.mean()),
                        "acc_leaked": float(ok[leaked].mean()) if leaked.any() else None,
                        "acc_clean": float(ok[~leaked].mean()) if (~leaked).any() else None,
                        "nn_acc_all": float(accuracy_score(yt[ev_idx], nn_pred)),
                        "nn_acc_leaked": float((nn_pred == yt[ev_idx])[leaked].mean())
                        if leaked.any() else None,
                        "nn_acc_clean": float((nn_pred == yt[ev_idx])[~leaked].mean())
                        if (~leaked).any() else None,
                    })
                r = rows[-1]
                print(f"[{name}/{task.task_id}] eval={r['n_eval']} 泄漏={r['n_leaked']} "
                      f"acc={r['acc_all']:.3f} (泄漏 {r['acc_leaked']} / 干净 "
                      f"{r['acc_clean']:.3f}) raw1NN={r['nn_acc_all']:.3f}", flush=True)
            del X, adj
            gc.collect()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leak_decompose.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--twins", action="store_true", help="全库重叠孪生扫描")
    ap.add_argument("--task-leak", action="store_true", help="任务级 eval→train 泄漏计数")
    ap.add_argument("--leak-decompose", action="store_true",
                    help="基线 acc 拆成泄漏/非泄漏子集（判据 LEAK_* 已写死）")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--max-tasks", type=int, default=STEP3_TASKS)
    ap.add_argument("--datasets", nargs="+", default=None, help="限定数据集（默认全部）")
    ap.add_argument("--registry", default=str(CANDIDATE_REGISTRY))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    if not (args.twins or args.task_leak or args.leak_decompose):
        ap.error("指定 --twins / --task-leak / --leak-decompose 之一")

    registry = load_registry(args.registry)
    names = args.datasets or probe_scope(registry)   # 源池 ∪ reserve，目标池绝不入内
    out_dir = Path(args.out)
    if args.twins:
        scan_twins(names, registry, out_dir=out_dir)
    if args.task_leak:
        task_leakage(names, registry, out_dir=out_dir)
    if args.leak_decompose:
        leak_decompose(names, registry, out_dir=out_dir,
                       n_seeds=args.n_seeds, max_tasks=args.max_tasks)


if __name__ == "__main__":
    main()
