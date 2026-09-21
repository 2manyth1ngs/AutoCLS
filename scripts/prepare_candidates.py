"""候选数据集准备驱动（换池调研第1步）：下载 → 解析 → 开窗 → canonical npz + 真实统计。

用法：
    python scripts/prepare_candidates.py --list
    python scripts/prepare_candidates.py --datasets wisdm mhealth
    python scripts/prepare_candidates.py --all --skip-existing
    python scripts/prepare_candidates.py --table            # 汇总统计表（第1步产出）

adapter 契约见 `scripts/candidates/__init__.py`。驱动负责预注册的统一后处理
（`results/candidate_survey/PROTOCOL.md` §1）：

1. **剔稀类**：样本数 <MIN_CLASS_COUNT 的类整类剔除后重新编号 0..k-1。
2. **规模上限**：N>MAX_SAMPLES 时按类分层随机子采样到 MAX_SAMPLES（seed 固定）。
   原始 N 记入 meta 的 `n_samples_raw`，统计表两个都报。
3. **按通道 z-score**（float64 统计 → float32 落盘），与 uea.py / datasets.py 同口径。

这三步写死在驱动里而不是各 adapter 里：8 个 adapter 各写一遍必然发散，而它们直接
决定统计表里的 N/K——第1步的初筛判据就吃这两个数。
"""
from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):        # 本机控制台默认 GBK
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from autocls.baseline_probe import prescreen, propose_slice  # noqa: E402
from autocls.data.candidates import (  # noqa: E402
    CANDIDATE_ROOT,
    WINDOW_LEN,
    WINDOW_STRIDE,
    describe,
    drop_rare_classes,
    save_candidate,
    zscore_channels,
)

# —— 预注册后处理常量（PROTOCOL §1，跑数据前写死）——
MIN_CLASS_COUNT = 64     # 类样本下限：低于此整类剔除（registry min_samples=32，留一倍余量）
MAX_SAMPLES = 30000      # 规模上限：切片每任务只取 1024，30k 足够且 npz <1 GB
SUBSAMPLE_SEED = 0

PKG = "scripts.candidates"
FAIL_LOG = ROOT / "results" / "candidate_survey" / "prepare_failures.jsonl"


def adapter_names() -> list[str]:
    pkg = importlib.import_module(PKG)
    return sorted(m.name for m in pkgutil.iter_modules(pkg.__path__)
                  if not m.name.startswith("_"))


def stratified_subsample(X, y, cap: int, seed: int):
    """按类比例分层子采样到 cap 条（保持类别分布，不改变不平衡比的量级）。"""
    if X.shape[0] <= cap:
        return X, y, False
    rng = np.random.default_rng(seed)
    keep = []
    vals, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, np.round(counts / counts.sum() * cap).astype(int))
    for v, q in zip(vals, quota):
        idx = np.where(y == v)[0]
        keep.append(rng.permutation(idx)[:q])
    sel = np.sort(np.concatenate(keep))
    return X[sel], y[sel], True


def effective_min_class_count(spec: dict) -> int:
    """剔稀类阈值：adapter SPEC 可用 `min_class_count` 覆盖驱动默认（WP17-ACTS UCR-ADM：UCR 十源取 5 = ACTS 切片器每类 ≥ 5 的
    对应物；调研期默认 64 会把 Adiac / ShapesAll / FiftyWords 整个剔空）。未写则逐位沿用 MIN_CLASS_COUNT。"""
    return int(spec.get("min_class_count", MIN_CLASS_COUNT))


def prepare_one(mod_name: str, *, skip_existing: bool) -> dict:
    mod = importlib.import_module(f"{PKG}.{mod_name}")
    spec = dict(mod.SPEC)
    name = spec["name"]
    if skip_existing and (CANDIDATE_ROOT / f"{name}.npz").exists():
        meta = json.loads((CANDIDATE_ROOT / f"{name}.json").read_text(encoding="utf-8"))
        print(f"[{name}] 已存在，跳过（N={meta['n_samples']}）", flush=True)
        return meta

    t0 = time.time()
    X, y, meta = mod.build()
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)
    # 连续流候选统一开窗到 WINDOW_LEN；已切好样本的档案（segmented=True，如 UniMiB 的
    # 151 步、Sleep-EDF 的 30 s epoch、UCR/UEA）保留档案原生长度，只查 L≥50 的 encoder 边界
    if X.ndim != 3:
        raise ValueError(f"{name}: build() 应返回 (N, T, C)，实得 {X.shape}")
    if spec.get("segmented"):
        if X.shape[1] < 50:
            raise ValueError(f"{name}: 原生长度 {X.shape[1]} < 50，低于 encoder 适用边界")
    elif X.shape[1] != WINDOW_LEN:
        raise ValueError(f"{name}: 流式候选应开窗到 (N, {WINDOW_LEN}, C)，实得 {X.shape}")
    n_raw, k_raw = X.shape[0], len(np.unique(y))

    min_cc = effective_min_class_count(spec)
    X, y = drop_rare_classes(X, y, min_cc)
    if X.shape[0] == 0:
        raise ValueError(f"{name}: 剔稀类（<{min_cc}）后无样本")
    X, y, capped = stratified_subsample(X, y, MAX_SAMPLES, SUBSAMPLE_SEED)
    X = zscore_channels(X)

    full = {**spec, **meta,
            "n_samples_raw": n_raw, "n_classes_raw": k_raw,
            "n_classes_dropped": k_raw - len(np.unique(y)),
            "subsampled": capped, "window_len": WINDOW_LEN, "window_stride": WINDOW_STRIDE,
            "min_class_count": min_cc, "seconds": round(time.time() - t0, 1)}
    save_candidate(name, X, y, full)
    d = describe(X, y)
    print(f"[{name}] N={d['n_samples']}(raw {n_raw}) L={d['seq_len']} C={d['n_channels']} "
          f"K={d['n_classes']}(raw {k_raw}) imb={d['imbalance_ratio']:.2f} "
          f"{time.time() - t0:.0f}s", flush=True)
    return {**full, **d}


UEA_SURVEY = ROOT / "results" / "uea_survey" / "uea_survey.csv"
UEA_TARGETS = {"Epilepsy", "FaceDetection"}          # 铁律 6（UCI-HAR 不在 UEA 档案里）
UEA_PROBED = {                                        # 已进 baseline_probe 的 16 个
    "ArticularyWordRecognition", "CharacterTrajectories", "Cricket", "ERing",
    "EthanolConcentration", "FingerMovements", "HandMovementDirection", "Handwriting",
    "Heartbeat", "MotorImagery", "NATOPS", "PhonemeSpectra", "SelfRegulationSCP1",
    "SelfRegulationSCP2", "SpokenArabicDigits", "UWaveGestureLibrary",
}


def _screen_line(name, domain, n, L, C, K, imb, extra="") -> str:
    ok, role, why = prescreen(n_samples=n, seq_len=L, n_classes=K)
    mark = {"source": "✔源", "reserve": "△备", "reject": "✘"}[role]
    return (f"{name:<27}{domain:<10}{n:>7}{L:>6}{C:>5}{K:>4}{imb:>8.2f}"
            f"{1.0 / K:>7.3f}{extra:>10}  {mark:<4}{'；'.join(why)}")


def table() -> None:
    """第1步产出：候选真实统计 + 初筛（PROTOCOL §2 硬条件 H1/H2/H3）。"""
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(CANDIDATE_ROOT.glob("*.json"))]
    if not rows:
        print("尚无候选，先跑 --all")
        return
    head = (f"{'数据集':<27}{'域':<10}{'N':>7}{'L':>6}{'C':>5}{'K':>4}{'不平衡':>8}"
            f"{'猜测':>7}{'fs/受试':>10}  {'初筛':<4}未过条件")
    print(f"{'=' * 130}\n新下载候选（实际加载确认，非论文数字）\n{'=' * 130}")
    print(head)
    n_src = 0
    for r in sorted(rows, key=lambda r: (r.get("domain", ""), -r.get("n_samples", 0))):
        ok, role, _ = prescreen(n_samples=r["n_samples"], seq_len=r["seq_len"],
                                n_classes=r["n_classes"])
        n_src += role == "source"
        print(_screen_line(r["name"], r.get("domain", "?"), r["n_samples"], r["seq_len"],
                           r["n_channels"], r["n_classes"], r["imbalance_ratio"],
                           f"{r.get('fs', '?')}/{r.get('n_subjects') or '?'}"))
    print(f"\n新候选 {len(rows)} 个：入源 {n_src}，其余见「未过条件」列")

    if UEA_SURVEY.exists():
        print(f"\n{'=' * 130}\nUEA 存量对照（results/uea_survey 的实测统计，同一把尺子）"
              f"\n{'=' * 130}")
        print(head)
        import csv

        kept = 0
        with open(UEA_SURVEY, encoding="utf-8") as f:
            for d in csv.DictReader(f):
                if d["name"] in UEA_TARGETS:
                    continue
                n, K = int(d["n_total"]), int(d["n_classes"])
                # 编码长度口径：变长数据集按 pad 到的 L_max 算（与 uea.py 落地形状一致）
                L = int(d["length_max"])
                ok, role, _ = prescreen(n_samples=n, seq_len=L, n_classes=K)
                kept += role == "source"
                tag = "已测" if d["name"] in UEA_PROBED else "未测"
                print(_screen_line(d["name"], d["domain"], n, L, int(d["n_channels"]), K,
                                   float(d["imbalance_ratio"]), tag))
        print(f"\nUEA 28 个（除目标池 2 个）：过初筛 {kept} 个")


# —— 成本模型（PROTOCOL §7，实测锚点）——
BASELINE_SEC = 0.22      # baseline_probe REPORT §7.3：685 条 / 2.5 min
TRAINED_SEC = 65.0       # baseline_probe --fill-trained：60 条 / 65 min
N_SEEDS = 5              # PROTOCOL §3
STEP3_ROWS = 20          # PROTOCOL §4.1：10 shared CLS × 前 2 任务 × 1 种子


def _l_factor(L: int) -> float:
    """iters=290 的耗时随 L 的粗修正（iters=0 由 SVM 主导、几乎与 L 无关）。

    两个锚点来自 PROTOCOL §7：L=1000 按 ×1.5、L=3000 按 ×3，中间线性内插。
    """
    return max(1.0, 1.0 + (L - 300) / 1350.0)


def plan() -> None:
    """第1步的规划与成本估算：每候选切多少任务、n_eval 够不够、步骤2/3 要多久。"""
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(CANDIDATE_ROOT.glob("*.json"))]
    rows = [r for r in rows
            if prescreen(n_samples=r["n_samples"], seq_len=r["seq_len"],
                         n_classes=r["n_classes"])[0]]
    if not rows:
        print("没有过初筛的候选")
        return

    print(f"{'数据集':<24}{'角色':<7}{'类别档':>9}{'任务':>5}{'任务N':>7}{'n_eval':>7}"
          f"{'R噪声':>8}{'L':>6}{'步骤2 s':>9}{'步骤3 s':>9}  n_eval")
    t2 = t3 = t3_src = 0.0
    n_tasks_source = 0
    for r in sorted(rows, key=lambda r: -r["n_samples"]):
        _ok, role, _ = prescreen(n_samples=r["n_samples"], seq_len=r["seq_len"],
                                 n_classes=r["n_classes"])
        p = propose_slice(n_samples=r["n_samples"], n_classes=r["n_classes"],
                          n_channels=r["n_channels"])
        c2 = p["planned_tasks"] * N_SEEDS * BASELINE_SEC
        c3 = STEP3_ROWS * TRAINED_SEC * _l_factor(r["seq_len"])
        t2 += c2
        t3 += c3
        if role == "source":
            n_tasks_source += p["planned_tasks"]
            t3_src += c3
        lo, hi = p["n_classes_range"]
        print(f"{r['name']:<24}{role:<7}{f'[{lo},{hi}]':>9}{p['planned_tasks']:>5}"
              f"{p['task_n_est']:>7}{p['n_eval_est']:>7}{p['rprime_noise_est']:>8.3f}"
              f"{r['seq_len']:>6}{c2:>9.0f}{c3:>9.0f}  "
              f"{'✔' if p['n_eval_ok'] else '✘ 噪声超 δ'}")

    print(f"\n步骤2（基线，iters=0，{N_SEEDS} 种子）：{t2 / 60:.1f} min")
    print(f"步骤3（训练能拉动，iters=290，每候选 {STEP3_ROWS} 条）：全部 {t3 / 3600:.1f} h / "
          f"**仅源角色 {t3_src / 3600:.1f} h**")
    print(f"  ← 两重上界：① reserve 候选（K<5）无论如何进不了源池，可不跑；"
          f"② 只对过 G1（基线≤0.90）的候选跑，贴顶的直接出局")
    print(f"\n源池任务合计 {n_tasks_source}（floor=100，"
          f"{'过闸' if n_tasks_source >= 100 else '不足，需更多候选'}）"
          f"——注意这是**初筛后**的上界，步骤3 判完 usable 还会再掉")


CANDIDATE_REGISTRY = ROOT / "configs" / "candidate_pool.yaml"


def emit_registry(path: Path = CANDIDATE_REGISTRY) -> None:
    """按初筛结果生成候选池注册表（步骤 2 的输入）。

    与 configs/source_pool.yaml 同 schema，故 baseline_probe --registry 能直接吃。
    **这不是新的源池定稿**——只是"要测基线的候选清单"；谁最终入池由步骤 3 判完
    usable 才定（PROTOCOL §5）。目标池原样带过来，让铁律 6 的 fail-fast 自检有对象。
    """
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(CANDIDATE_ROOT.glob("*.json"))]
    src, res = [], []
    for r in sorted(rows, key=lambda r: -r["n_samples"]):
        ok, role, _ = prescreen(n_samples=r["n_samples"], seq_len=r["seq_len"],
                                n_classes=r["n_classes"])
        if not ok:
            continue
        p = propose_slice(n_samples=r["n_samples"], n_classes=r["n_classes"],
                          n_channels=r["n_channels"])
        lo, hi = p["n_classes_range"]
        line = (f"  - {{name: {r['name']}, source: npz, domain: {r.get('domain', 'Other')},"
                f" n_channels: {r['n_channels']}, n_classes: {r['n_classes']},"
                f" l_enc: {r['seq_len']}, crop_len: null,"
                f" n_classes_range: [{lo}, {hi}],"
                f" planned_tasks: {p['planned_tasks']}, tested: false}}")
        (src if role == "source" else res).append(line)

    body = [
        "# =====================================================================",
        "# 换源池调研——候选池注册表（scripts/prepare_candidates.py --emit-registry 生成）",
        "#",
        "# 这不是新源池的定稿，只是**要测基线的候选清单**：谁最终入池由步骤 3 判完",
        "# usable 才定（results/candidate_survey/PROTOCOL.md §5）。",
        "# 角色按 §2 初筛：K>=5 记 source，K=2-4 记 reserve（照样测基线，但撑不起任务多样性）。",
        "# 切片档由 propose_slice 提（类别档 [0.4K, 0.8K]，与 source_pool.yaml 现有档同型）。",
        "#",
        "# δ 仍在 configs/labels.yaml（唯一真源，防 split-brain）——候选的 δ 待步骤 3 后标定。",
        "# =====================================================================",
        "",
        "encode_length_floor: 50",
        "",
        "slice:",
        "  seed: 0",
        "  n_samples: 1024      # 与 source_pool.yaml 同参，保证任务定义口径一致",
        "  min_samples: 32",
        "",
        "# 目标池原样带过来：让 baseline_probe 的铁律 6 自检有对象（fail-fast 必须仍生效）",
        "target_pool:",
        "  - {name: UCI-HAR,       source: external, domain: HAR}",
        "  - {name: Epilepsy,      source: uea,      domain: HAR}",
        "  - {name: FaceDetection, source: uea,      domain: EEG/MEG}",
        "",
        "source_pool:",
        *(src or ["  []"]),
        "",
        "reserve:",
        *(res or ["  []"]),
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")
    print(f"候选池注册表 → {path}（source {len(src)} / reserve {len(res)}）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None, help="adapter 模块名（小写）")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--plan", action="store_true", help="切片规划 + 步骤2/3 成本估算")
    ap.add_argument("--emit-registry", action="store_true",
                    help="按初筛结果生成 configs/candidate_pool.yaml（步骤2 的输入）")
    args = ap.parse_args()

    if args.emit_registry:
        emit_registry()
        return

    if args.list:
        print("\n".join(adapter_names()) or "（尚无 adapter）")
        return
    if args.table:
        table()
        return
    if args.plan:
        plan()
        return

    names = adapter_names() if args.all else (args.datasets or [])
    if not names:
        ap.error("指定 --datasets / --all / --list / --table 之一")
    ok, bad = [], []
    for n in names:
        try:
            prepare_one(n, skip_existing=args.skip_existing)
            ok.append(n)
        except Exception as e:  # 一个候选失败不该拖垮整批调研
            bad.append((n, f"{type(e).__name__}: {e}"))
            print(f"[{n}] 失败 {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
    if bad:
        with open(FAIL_LOG, "a", encoding="utf-8") as f:
            for n, msg in bad:
                f.write(json.dumps({"adapter": n, "error": msg}, ensure_ascii=False) + "\n")
    print(f"\n成功 {len(ok)}/{len(names)}" + (f"，失败 {[b[0] for b in bad]}" if bad else ""))


if __name__ == "__main__":
    main()
