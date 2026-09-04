"""UCR 128 单变量档案调查 + 预取（WP7-UCR 语料；零 GPU）。

改自 results/uea_survey/scripts/uea_survey.py 的四处：
1. 名单 = aeon.datasets.tsc_datasets.univariate（128）− 硬排除 9（与源池/靶同源，铁律 6）；
2. 缓存 = data/raw/ucr_cache（scripts/candidates/_ucr.UCR_CACHE），不混进 data/uea_cache；
3. 元信息只读 .ts 文件头（@seriesLength/@classLabel/@equalLength/@missing）+ 行计数，不解析数据体；
4. 输出列换成"整数据集当一个任务"需要的量与 usable 判据。
下载复用 scripts/candidates/_ucr._ensure_local（Zenodo、默认 UA、.part 原子替换、@problemName 校验）。
可续跑：per_dataset/<name>.json 存在即跳过。

用法：python results/ucr_survey/scripts/ucr_survey.py            # 全部
      python results/ucr_survey/scripts/ucr_survey.py Beef Coffee  # 指定
      python results/ucr_survey/scripts/ucr_survey.py --aggregate  # 只汇总 → ucr_survey.csv + configs/ucr_pool.yaml
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "ucr_survey"
PER = OUT / "per_dataset"
CSV = OUT / "ucr_survey.csv"
POOL_YAML = ROOT / "configs" / "ucr_pool.yaml"

# 与源池 / 靶同源的 UCR 档案（铁律 6：语料绝不含靶的同源数据集）
HARD_EXCLUDE = {
    "CricketX", "CricketY", "CricketZ",                                   # UEA Cricket
    "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ",
    "UWaveGestureLibraryAll",                                             # UEA UWaveGestureLibrary
    "Phoneme",                                                            # UEA PhonemeSpectra
    "EthanolLevel",                                                       # UEA EthanolConcentration
}
# 名字巧合、经核查非同源（REPORT 显式记录）：
NAME_COINCIDENCE = {"SemgHandMovementCh2": "sEMG 前臂肌电，非 HandMovementDirection 的 MEG",
                    "FaceAll": "人脸轮廓图像→1D，非 FaceDetection 的 MEG",
                    "FaceFour": "同上", "FacesUCR": "同上",
                    "HandOutlines": "手部 X 光轮廓，非 Handwriting 的笔尖加速度"}
# 同一采集的三种标签视角，三选一（留 Movement）
SEMG_DROP = {"SemgHandGenderCh2", "SemgHandSubjectCh2"}

MIN_L = 50            # 固定 encoder 适用边界（CLAUDE.md limitation）
N_CAP = 1024          # 整数据集任务的样本上限（= 源池切片 n_samples）
EVAL_RATIO = 0.3
CROP_LEN = 512        # T > CROP_LEN 时前缀截断（用户 2026-09-04 拍板 512：长序列档可 4 并发，墙钟减半；源池先例 Ethanol 1024 / Cricket 256）


def parse_ts_header(path: Path) -> dict:
    """只读 .ts 头部 + 逐行计数：{problem, series_length, equal_length, missing, univariate, classes, counts, n}。"""
    meta: dict = {"problem": None, "series_length": None, "equal_length": None, "missing": None,
                  "univariate": None, "classes": []}
    counts: Counter = Counter()
    n = 0
    in_data = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if not in_data:
                low = s.lower()
                if low.startswith("@data"):
                    in_data = True
                    continue
                if low.startswith("#"):
                    continue
                if low.startswith("@problemname"):
                    meta["problem"] = s.split(None, 1)[1].strip() if len(s.split()) > 1 else ""
                elif low.startswith("@serieslength"):
                    meta["series_length"] = int(s.split()[1])
                elif low.startswith("@equallength"):
                    meta["equal_length"] = s.split()[1].lower() == "true"
                elif low.startswith("@missing"):
                    meta["missing"] = s.split()[1].lower() == "true"
                elif low.startswith("@univariate"):
                    meta["univariate"] = s.split()[1].lower() == "true"
                elif low.startswith("@classlabel"):
                    toks = s.split()
                    meta["classes"] = toks[2:] if len(toks) > 2 and toks[1].lower() == "true" else []
                continue
            n += 1
            counts[s.rsplit(":", 1)[-1].strip()] += 1
    meta["n"] = n
    meta["counts"] = dict(counts)
    return meta


def usable_rule(rec: dict, *, min_l: int = MIN_L, n_cap: int = N_CAP, eval_ratio: float = EVAL_RATIO,
                min_n_eval: int = 50) -> tuple[bool, str]:
    """整数据集任务可用判据（先写死）：等长、无缺失、单变量、L ≥ min_l、每类 ≥ 2、n_eval ≥ min_n_eval。"""
    if rec["name"] in HARD_EXCLUDE:
        return False, "hard_exclude"
    if rec["name"] in SEMG_DROP:
        return False, "semg_duplicate"
    if not rec["equal_length"]:
        return False, "variable_length"
    if rec["missing"]:
        return False, "missing_values"
    if not rec["univariate"]:
        return False, "multivariate"
    if rec["length"] is None or rec["length"] < min_l:
        return False, f"L<{min_l}"
    if rec["min_class"] < 2:
        return False, "class<2"
    n_eff = min(rec["n_total"], n_cap)
    if round(n_eff * eval_ratio) < min_n_eval:
        return False, f"n_eval<{min_n_eval}"
    return True, "ok"


def survey_one(name: str) -> dict:
    from scripts.candidates._ucr import UCR_CACHE, _ensure_local
    fp = PER / f"{name}.json"
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    t0 = time.time()
    ddir = _ensure_local(name, UCR_CACHE)
    tr = parse_ts_header(ddir / f"{name}_TRAIN.ts")
    te = parse_ts_header(ddir / f"{name}_TEST.ts")
    counts = Counter(tr["counts"])
    counts.update(te["counts"])
    lens = {tr["series_length"], te["series_length"]}
    rec = {"name": name, "n_train": tr["n"], "n_test": te["n"], "n_total": tr["n"] + te["n"],
           "n_channels": 1 if (tr["univariate"] is not False) else None,
           "length": max(v for v in lens if v is not None) if any(v is not None for v in lens) else None,
           "equal_length": bool(tr["equal_length"] and te["equal_length"] and len(lens) == 1),
           "missing": bool(tr["missing"] or te["missing"]), "univariate": bool(tr["univariate"]),
           "n_classes": len(counts), "class_counts": dict(sorted(counts.items())),
           "min_class": min(counts.values()), "imbalance_ratio": round(max(counts.values()) / min(counts.values()), 3),
           "seconds": round(time.time() - t0, 1)}
    rec["usable"], rec["reason"] = usable_rule(rec)
    PER.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[done] {name}: N={rec['n_total']} L={rec['length']} K={rec['n_classes']} min_class={rec['min_class']} "
          f"usable={rec['usable']} ({rec['reason']}) {rec['seconds']}s", flush=True)
    return rec


def aggregate() -> list[dict]:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(PER.glob("*.json"))]
    for r in recs:
        r["usable"], r["reason"] = usable_rule(r)
    cols = ["name", "n_train", "n_test", "n_total", "n_channels", "length", "equal_length", "missing",
            "univariate", "n_classes", "min_class", "imbalance_ratio", "usable", "reason"]
    with open(CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({c: r.get(c) for c in cols})
    usable = [r for r in recs if r["usable"]]
    reasons = Counter(r["reason"] for r in recs)
    print(f"surveyed {len(recs)}, usable {len(usable)}; reasons {dict(reasons)}", flush=True)
    for thr in (36, 50, 75, 100):
        n = sum(1 for r in recs if usable_rule(r, min_n_eval=thr)[0])
        print(f"  usable if n_eval >= {thr}: {n}", flush=True)
    write_pool_yaml(usable)
    return recs


def pool_entries(usable: list[dict]) -> list[dict]:
    out = []
    for r in sorted(usable, key=lambda r: r["name"]):
        L = int(r["length"])
        out.append({"name": r["name"], "domain": "UCR", "source": "uea", "n_channels": 1,
                    "n_classes": int(r["n_classes"]), "l_enc": min(L, CROP_LEN),
                    "crop_len": CROP_LEN if L > CROP_LEN else None,
                    "n_classes_range": [int(r["n_classes"]), int(r["n_classes"])], "planned_tasks": 1,
                    "n_total": int(r["n_total"])})
    return out


def write_pool_yaml(usable: list[dict]) -> None:
    ents = pool_entries(usable)
    lines = ["# WP7-UCR 语料注册表（由 results/ucr_survey/scripts/ucr_survey.py --aggregate 生成，勿手改）",
             "# 整数据集 = 一个任务：n_classes_range=[K,K]、planned_tasks=1；N 上限见 slice.n_samples（=源池切片同值）；",
             "# T>1024 前缀截断（crop_len），与源池 Ethanol 同口径。缓存目录 data/raw/ucr_cache（不混 uea_cache）。",
             f"# 硬排除（与源池/靶同源，铁律 6）：{sorted(HARD_EXCLUDE)}",
             f"# 名字巧合已核非同源：{sorted(NAME_COINCIDENCE)}；Semg 三选一（丢 {sorted(SEMG_DROP)}）",
             "encode_length_floor: 50",
             "slice:", "  seed: 0", f"  n_samples: {N_CAP}", "  min_samples: 32",
             "target_pool: []", "source_pool:"]
    for e in ents:
        crop = "null" if e["crop_len"] is None else e["crop_len"]
        lines.append(f"  - {{name: {e['name']}, domain: {e['domain']}, source: uea, n_channels: 1, "
                     f"n_classes: {e['n_classes']}, l_enc: {e['l_enc']}, crop_len: {crop}, "
                     f"n_classes_range: [{e['n_classes']}, {e['n_classes']}], planned_tasks: 1}}  # N={e['n_total']}")
    POOL_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {POOL_YAML} with {len(ents)} datasets", flush=True)


def main(argv: list[str]) -> None:
    from aeon.datasets import tsc_datasets as tsc
    if argv and argv[0] == "--aggregate":
        aggregate()
        return
    names = argv if argv else [n for n in sorted(tsc.univariate) if n not in HARD_EXCLUDE]
    for nm in names:
        try:
            survey_one(nm)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {nm}: {type(e).__name__}: {e}", flush=True)
    aggregate()


if __name__ == "__main__":
    main(sys.argv[1:])
