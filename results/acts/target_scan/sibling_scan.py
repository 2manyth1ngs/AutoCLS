"""池外目标筛选（探索）：17 个源 vs UEA / UCR 池外候选的 11 维统计特征距离。只用数据本身，不涉及任何模型读数。"""
import csv, importlib, json, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(r"D:\DevelopDoc\CLS"); sys.path.insert(0, str(ROOT))
SP = Path(r"C:\Users\29466\AppData\Local\Temp\claude\D--DevelopDoc-CLS\fec96ed9-c1ad-4224-84c6-0361991e4fe9\scratchpad")
UCR_DIR = SP / "ucr"
g40 = importlib.import_module("scripts.40_task_feature_scorer"); g42 = importlib.import_module("scripts.42_ucr_module")
ac = importlib.import_module("scripts.acts_common")
from autocls.data.uea import load_uea_cached

SOURCES = ["Handwriting", "UniMiB-SHAR", "PhonemeSpectra", "UWaveGestureLibrary", "Cricket", "PTB-XL", "SleepEDF", "PAMAP2",
           "NonInvasiveFetalECGThorax1", "ShapesAll", "LargeKitchenAppliances", "Adiac", "FordA", "SemgHandMovementCh2",
           "EthanolConcentration", "CinCECGTorso", "SelfRegulationSCP1"]
NEAR_DUP = {"CricketX": "Cricket", "CricketY": "Cricket", "CricketZ": "Cricket", "UWaveGestureLibraryX": "UWaveGestureLibrary",
            "UWaveGestureLibraryY": "UWaveGestureLibrary", "UWaveGestureLibraryZ": "UWaveGestureLibrary",
            "UWaveGestureLibraryAll": "UWaveGestureLibrary", "NonInvasiveFetalECGThorax2": "NonInvasiveFetalECGThorax1",
            "FordB": "FordA", "SemgHandGenderCh2": "SemgHandMovementCh2", "SemgHandSubjectCh2": "SemgHandMovementCh2",
            "EthanolLevel": "EthanolConcentration", "Phoneme": "PhonemeSpectra", "SelfRegulationSCP2": "SelfRegulationSCP1"}
COLS = ["log10_T", "log10_C", "log10_k", "spec_conc", "hf_ratio", "acf_tau", "ex_kurt", "chan_cv", "samp_cv", "trend_strength", "season_strength"]
rng = np.random.default_rng(0)


def feats(X, y, T_full):
    X = np.asarray(X, float)
    if X.shape[0] > 600:
        X = X[np.sort(rng.choice(X.shape[0], 600, replace=False))]
    Xs = X[:, :1024, :]
    Xs = np.nan_to_num(Xs)
    d = {"log10_T": float(np.log10(T_full)), "log10_C": float(np.log10(X.shape[2])), "log10_k": float(np.log10(len(np.unique(y))))}
    d.update(g40.signal_stats(Xs)); d.update(g42.decomp_strength(Xs))
    return d


out = {}
cache = SP / "sibling_feats.json"
if cache.exists():
    out = json.loads(cache.read_text(encoding="utf-8"))
# ---- 源（整集，= ACTS 训练用的数据）
for s in SOURCES:
    if s in out: continue
    try:
        X, y = ac.load_dataset_full(s)
        out[s] = {"role": "source", "bench": "pool", "N": int(len(y)), **feats(X, y, X.shape[1])}
        print("[src]", s, X.shape, flush=True)
    except Exception as e:
        print("[src] FAIL", s, e, flush=True)
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
# ---- UEA 候选（官方 TRAIN）
uea = [r for r in csv.DictReader(open(ROOT / "results/uea_survey/uea_survey.csv"))]
for r in uea:
    n = r["name"]
    if n in SOURCES or n in out: continue
    try:
        X, y = load_uea_cached(n, "TRAIN")[:2]
        Xte, yte = load_uea_cached(n, "TEST")[:2]
        out[n] = {"role": "near_dup" if n in NEAR_DUP else "cand", "bench": "UEA", "N": int(len(y)), "N_test": int(len(yte)),
                  "L_min": int(r["length_min"]), **feats(X, y, int(r["length_max"]))}
        print("[uea]", n, X.shape, flush=True)
    except Exception as e:
        print("[uea] FAIL", n, repr(e)[:120], flush=True)
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
# ---- UCR 候选（官方 TRAIN；survey 可用的 + 近重复）
from aeon.datasets import load_classification
ucr = {r["name"]: r for r in csv.DictReader(open(ROOT / "results/ucr_survey/ucr_survey.csv"))}
names = [n for n, r in ucr.items() if r["usable"] == "True" and n not in SOURCES] + [n for n in NEAR_DUP if n not in ucr and n != "SelfRegulationSCP2"]
for n in names:
    if n in out: continue
    try:
        t0 = time.time()
        X, y = load_classification(n, split="train", extract_path=str(UCR_DIR))
        Xte, yte = load_classification(n, split="test", extract_path=str(UCR_DIR))
        X = np.transpose(np.asarray(X, float), (0, 2, 1)); X = (X - X.mean()) / (X.std() + 1e-12)
        out[n] = {"role": "near_dup" if n in NEAR_DUP else "cand", "bench": "UCR", "N": int(len(y)), "N_test": int(len(yte)),
                  "L_min": int(X.shape[1]), **feats(X, y, X.shape[1])}
        print("[ucr]", n, X.shape, round(time.time() - t0, 1), flush=True)
    except Exception as e:
        print("[ucr] FAIL", n, repr(e)[:120], flush=True)
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print("done", len(out))
