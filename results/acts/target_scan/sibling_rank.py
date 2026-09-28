"""读 sibling_feats.json：按源标准化 11 维统计特征，候选到最近源的距离，对照源与源之间的最近距离。"""
import json, sys
from pathlib import Path
import numpy as np

SP = Path(r"C:\Users\29466\AppData\Local\Temp\claude\D--DevelopDoc-CLS\fec96ed9-c1ad-4224-84c6-0361991e4fe9\scratchpad")
F = json.loads((SP / "sibling_feats.json").read_text(encoding="utf-8"))
COLS = ["log10_T", "log10_C", "log10_k", "spec_conc", "hf_ratio", "acf_tau", "ex_kurt", "chan_cv", "samp_cv", "trend_strength", "season_strength"]
src = [n for n, d in F.items() if d["role"] == "source"]
S = np.array([[F[n][c] for c in COLS] for n in src]); mu, sd = S.mean(0), S.std(0); sd[sd < 1e-9] = 1
Z = lambda n: (np.array([F[n][c] for c in COLS]) - mu) / sd
Zs = np.array([Z(n) for n in src])
D = np.sqrt(((Zs[:, None] - Zs[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
nn_src = D.min(1)
print("源与源最近距离：中位 %.2f  25%% %.2f  75%% %.2f" % (np.median(nn_src), *np.quantile(nn_src, [0.25, 0.75])))
for n, d, j in zip(src, nn_src, D.argmin(1)):
    print("   %-28s → %-26s %.2f" % (n, src[j], d))
ref = float(np.median(nn_src))
rows = []
for n, d in F.items():
    if d["role"] == "source": continue
    z = Z(n); dist = np.sqrt(((Zs - z) ** 2).sum(1)); o = np.argsort(dist)
    rows.append((dist[o[0]], n, d["bench"], d["role"], src[o[0]], src[o[1]], dist[o[1]], d.get("N"), d.get("N_test"), d.get("L_min"), int(round(10 ** d["log10_k"])), int(round(10 ** d["log10_T"])), int(round(10 ** d["log10_C"]))))
rows.sort()
print("\n参照线（源间最近距离中位）%.2f；候选按到最近源的距离排序：" % ref)
print("%-6s %-28s %-4s %-8s %-26s %-26s %6s %6s %5s %5s %4s %5s" % ("dist", "候选", "库", "关系", "最近源", "次近源", "N训", "N测", "Lmin", "k", "T", "C"))
for r in rows:
    print("%6.2f %-28s %-4s %-8s %-26s %-22s(%.2f) %6s %6s %5s %4s %5s %4s" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12]))
json.dump([{"dist": float(r[0]), "name": r[1], "bench": r[2], "role": r[3], "nn": r[4], "nn2": r[5], "d2": float(r[6]), "N": r[7], "N_test": r[8], "L_min": r[9], "k": r[10], "T": r[11], "C": r[12]} for r in rows],
          open(SP / "sibling_rank.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
