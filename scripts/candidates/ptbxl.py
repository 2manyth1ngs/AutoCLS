"""PTB-XL 12 导联临床 ECG → 5 个诊断超类。

来源：PhysioNet `ptb-xl` 1.0.3。TF-C 等相关工作用它做 ECG 预训练/迁移的标准基准。
口径：**100 Hz 版本（records100）、10 s 记录、12 导联** → 每样本 1000 步 × 12 通道。

**已切好样本**：PTB-XL 的记录本身就是定长 10 s 临床 ECG，样本 = 一条记录。故 SPEC 带
`segmented=True`，不走 `window_stream`——ECG 诊断的判据（ST 段形态、QRS 时限、导联间
对比）需要完整心动周期，切成 1.28 s 的 128 步窗会把被分类的对象本身破坏掉。

已知坑（按踩到的顺序）：
1. **完整 zip 1.7 GB，不能整包下**。改按记录取 `records100/` 下的 `.dat`/`.hea`
   （单条 ~24.6 KB），4000 条约 100 MB。元数据两个 csv 另计 6.6 MB。
2. **PhysioNet 并发上限**：≥16 路并发（叠加其他下载任务时）会稳定报
   `SSLEOFError: EOF occurred in violation of protocol`——服务端主动断连，不是本地证书
   问题。故限 `DOWNLOAD_WORKERS=12` 并对每个文件做**指数退避重试**；单条失败不该
   拖垮 4000 条的批次。
3. **多标签歧义**：`scp_codes` 是 {诊断码: 置信度} 字典，一条记录可映射到多个超类。
   全量 21799 条里 411 条映射到 0 个超类、5144 条映射到 ≥2 个。本 adapter **只保留恰好
   属于 1 个超类的 16244 条**，丢弃数记入 meta——AutoCLS 的下游评估器是单标签分类器
   （linear/SVM），多标签记录塞进去只会制造标签噪声。
4. **置信度不参与判定**：`scp_codes` 的值有大量 0.0（标注者未给置信度，不是"该诊断不成立"）。
   PTB-XL 官方示例代码 `aggregate_diagnostic` 直接取字典的 key、忽略置信度，本模块照做。
5. **HYP 超类稀少**：单标签记录里 HYP 只有 535 条，做不到 5 类各 800 的严格均衡。
   用"注水式"配额（先按均分，缺额在有余量的类间再均分）填满 4000，不平衡比 ~1.6。

不重采样：直接用 100 Hz 版本（`filename_lr`），不碰 500 Hz 的 `filename_hr`。
不插值：wfdb 读出含非有限值的记录整条丢弃并计数。
"""
from __future__ import annotations

import ast
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from autocls.data.candidates import RAW_ROOT, fetch_archive

BASE = "https://physionet.org/files/ptb-xl/1.0.3/"

SPEC = {
    "name": "PTB-XL",
    "domain": "ECG",
    "fs": 100,                 # records100 版本（.hea 头里读出来的实际值）
    "url": BASE,
    "n_subjects": None,        # build() 用实际选中记录的 patient_id 去重数覆盖
    "segmented": True,         # 10 s 定长记录已是样本，不开窗
    "note": "PhysioNet ptb-xl 1.0.3 records100 抽 4000 条；12 导联 10 s；"
            "5 诊断超类 NORM/MI/STTC/CD/HYP；"
            "坑：1.7GB 整包不下改按记录取、≥16 并发被服务端断连、多超类记录必须丢",
}

N_RECORDS = 4000               # 下载量上限（~100 MB）；分层均衡抽样
DOWNLOAD_WORKERS = 12          # 坑 2：再高就撞 PhysioNet 的并发断连
RETRIES = 4
SAMPLE_SEED = 0

# 固定类序 = 字母序：驱动的 drop_rare_classes 按 sorted(标签值) 重编号，
# 用字母序赋 id 可保证"没有类被剔除"时 meta 里的 class_names 与最终 y 一一对应。
CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
CLASS_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

SIG_LEN = 1000                 # 10 s @100 Hz
N_LEADS = 12


def _fetch_retry(url: str, dest, *, tries: int = RETRIES):
    """坑 2：PhysioNet 会随机断连，指数退避重试；全败则抛给调用方计数。"""
    last = None
    for i in range(tries):
        try:
            return fetch_archive(url, dest, timeout=120)
        except Exception as e:      # SSLError / ChunkedEncodingError / ReadTimeout
            last = e
            # 半份文件不会被当缓存（fetch_archive 用 .part + 原子替换），直接重试即可
            time.sleep(2 ** i)
    raise last


def _load_metadata(raw_dir):
    """取两个 csv → (记录表 db, 诊断码→超类 映射 agg)。"""
    import pandas as pd

    for f in ("ptbxl_database.csv", "scp_statements.csv"):
        _fetch_retry(BASE + f, raw_dir / f)
    db = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(raw_dir / "scp_statements.csv", index_col=0)
    # diagnostic==1 才是诊断类语句（其余是 form/rhythm 类，没有 diagnostic_class）
    agg = scp[scp.diagnostic == 1].diagnostic_class.dropna()
    return db, agg


def _superclasses(scp_codes: str, agg) -> list[str]:
    """坑 4：官方 aggregate_diagnostic 口径——只取字典的 key，忽略置信度。"""
    codes = ast.literal_eval(scp_codes)
    return sorted({agg[k] for k in codes if k in agg.index})


def _waterfill_quota(counts: dict[str, int], total: int) -> dict[str, int]:
    """坑 5：尽量均衡的配额——按均分给，给不满的类给光，缺额在剩余类间再均分。"""
    quota, remaining, pool = {}, total, sorted(counts, key=lambda c: counts[c])
    for i, cls in enumerate(pool):
        share = remaining // (len(pool) - i)
        take = min(counts[cls], share)
        quota[cls] = take
        remaining -= take
    return quota


def _select(db, agg) -> tuple[list, dict]:
    """全量记录 → 选中的 (ecg_id, filename_lr, 超类) 列表 + 丢弃统计。"""
    sup = db.scp_codes.map(lambda s: _superclasses(s, agg))
    n_sup = sup.map(len)
    single = db[n_sup == 1].copy()
    single["superclass"] = sup[n_sup == 1].map(lambda x: x[0])

    counts = single.superclass.value_counts().to_dict()
    quota = _waterfill_quota(counts, N_RECORDS)

    rng = np.random.default_rng(SAMPLE_SEED)
    picked = []
    for cls in CLASS_NAMES:
        idx = single.index[single.superclass == cls].to_numpy()
        sel = rng.permutation(idx)[: quota.get(cls, 0)]
        picked.extend(sorted(sel.tolist()))
    picked = sorted(picked)

    rows = [(int(i), single.loc[i, "filename_lr"], single.loc[i, "superclass"],
             int(single.loc[i, "patient_id"])) for i in picked]
    stats = {
        "n_total_records": int(len(db)),
        "n_dropped_zero_superclass": int((n_sup == 0).sum()),
        "n_dropped_multi_superclass": int((n_sup >= 2).sum()),
        "n_single_superclass": int(len(single)),
        "single_superclass_counts": {k: int(v) for k, v in sorted(counts.items())},
        "sample_quota": {k: int(v) for k, v in sorted(quota.items())},
    }
    return rows, stats


def _download(rows, raw_dir) -> int:
    """并发取 .dat/.hea。返回下载失败（重试耗尽）的记录数。"""
    jobs = [rel + ext for _, rel, _, _ in rows for ext in (".dat", ".hea")]
    failed = []

    def one(rel: str):
        try:
            _fetch_retry(BASE + rel, raw_dir / rel)
        except Exception as e:
            failed.append((rel, f"{type(e).__name__}: {e}"))

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        list(ex.map(one, jobs))
    if failed:
        print(f"  [PTB-XL] {len(failed)} 个文件重试耗尽仍失败，例：{failed[:2]}", flush=True)
    return len(failed)


def build() -> tuple[np.ndarray, np.ndarray, dict]:
    import wfdb

    raw_dir = RAW_ROOT / "ptbxl"
    raw_dir.mkdir(parents=True, exist_ok=True)

    db, agg = _load_metadata(raw_dir)
    rows, sel_stats = _select(db, agg)
    n_failed_files = _download(rows, raw_dir)

    xs, ys, patients = [], [], []
    n_unreadable, n_bad_shape, n_nonfinite = 0, 0, 0
    for _ecg_id, rel, cls, pid in rows:
        try:
            rec = wfdb.rdrecord(str(raw_dir / rel))
        except Exception:          # 下载失败/文件损坏的记录直接跳过，不补不造
            n_unreadable += 1
            continue
        sig = np.asarray(rec.p_signal, dtype=np.float64)
        if sig.shape != (SIG_LEN, N_LEADS):
            n_bad_shape += 1
            continue
        if not np.isfinite(sig).all():   # 铁律：不插值补缺失，整条丢
            n_nonfinite += 1
            continue
        xs.append(sig.astype(np.float32))
        ys.append(CLASS_ID[cls])
        patients.append(pid)

    if not xs:
        raise RuntimeError("PTB-XL：一条记录都没读成（下载全失败？）")

    X = np.stack(xs)
    y = np.asarray(ys, dtype=np.int64)
    downloaded = sum(p.stat().st_size for p in raw_dir.rglob("*")
                     if p.is_file() and p.suffix in (".dat", ".hea", ".csv"))
    meta = {
        "class_names": CLASS_NAMES,
        "leads": ["I", "II", "III", "AVR", "AVL", "AVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"],
        "n_records_requested": N_RECORDS,
        "n_records_loaded": int(X.shape[0]),
        "n_subjects": int(len(set(patients))),
        "n_failed_files": int(n_failed_files),
        "n_unreadable": int(n_unreadable),
        "n_bad_shape": int(n_bad_shape),
        "n_nonfinite_dropped": int(n_nonfinite),
        "sample_seed": SAMPLE_SEED,
        "bytes_downloaded": int(downloaded),
        **sel_stats,
    }
    return X, y, meta
