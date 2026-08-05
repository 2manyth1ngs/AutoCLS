"""M2 的 ρ=0.85 里有多少是"破坏排序"撑的？（full-iters 判别实验的配套问题）

背景：M2 验证了 early-val 与完整训练的 CLS 排名 Spearman ρ=0.85，这是 early-val
这条路线的**唯一**实证支撑，整个项目的成本模型建立在它上面。但两轮探针都指向
"存活 label 由破坏 vs 非破坏主导"（解耦实验 §7）。若 ρ=0.85 也主要由"哪些 CLS 把
表征训坏了"撑着，那么这个可迁移的排名**对目标无用**——zero-shot 搜索要的是在好
配置之间分辨优劣，不是把烂配置认出来。

做法：对同一批 CLS，逐步剔掉"最差的 k 个"再算 ρ。若 ρ 随剔除迅速塌掉，说明相关性
由尾部（破坏性配置）贡献；若基本稳住，说明好配置之间也有真排序。

数据来源：`results/m2_probe/cache.jsonl` —— **M2 里程碑的历史记录，本脚本不加载任何
数据集、不新测**（口径同 candidate_survey/PROTOCOL.md §0.2：引用 HAR 历史里程碑
数字不算碰目标池）。

用法：python scripts/m2_rho_decomposition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

CACHE = ROOT / "results" / "m2_probe" / "cache.jsonl"


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman ρ（含并列的平均秩）。样本小，不引 scipy。"""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def perm_p(a: list[float], b: list[float], n_perm: int = 20000, seed: int = 0) -> float:
    """置换检验的双侧 p：|ρ| ≥ 实测的比例。n 小（10 左右）时渐近公式不可靠。"""
    import random

    obs = abs(spearman(a, b))
    rng = random.Random(seed)
    bb = list(b)
    hit = 0
    for _ in range(n_perm):
        rng.shuffle(bb)
        if abs(spearman(a, bb)) >= obs - 1e-12:
            hit += 1
    return (hit + 1) / (n_perm + 1)


def main() -> None:
    rows = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_epochs: dict[int, dict[int, float]] = {}
    for r in rows:
        if r.get("error") or r.get("acc") is None:
            continue
        by_epochs.setdefault(int(r["epochs"]), {})[int(r["cls_idx"])] = float(r["acc"])

    arms = sorted(by_epochs)
    print(f"M2 历史记录：{CACHE}")
    print(f"训练量档（epochs）：{arms}，各档 CLS 数 "
          f"{ {e: len(by_epochs[e]) for e in arms} }")
    early_e, full_e = min(arms), max(arms)
    early, full = by_epochs[early_e], by_epochs[full_e]
    common = sorted(set(early) & set(full))
    e = [early[i] for i in common]
    f = [full[i] for i in common]
    print(f"\n配对 CLS {len(common)} 条；early={early_e}ep full={full_e}ep")
    print(f"full acc 范围 [{min(f):.4f}, {max(f):.4f}]，"
          f"early acc 范围 [{min(e):.4f}, {max(e):.4f}]")

    rho_all = spearman(e, f)
    print(f"\n全体 ρ = {rho_all:.4f}   ← M2 报告的那个数")

    print(f"\n{'—'*76}")
    print("逐步剔掉 full acc 最差的 k 条后重算 ρ（测 ρ 由谁撑着）")
    print(f"{'剔除k':>6}{'剩余n':>7}{'剔掉的full范围':>22}{'剩余full范围':>22}{'ρ':>9}")
    order = sorted(range(len(f)), key=lambda i: f[i])   # full acc 升序
    for k in range(0, len(common) - 3):
        keep = order[k:]
        ek = [e[i] for i in keep]
        fk = [f[i] for i in keep]
        dropped = [f[i] for i in order[:k]]
        drng = f"[{min(dropped):.3f},{max(dropped):.3f}]" if dropped else "—"
        print(f"{k:>6}{len(keep):>7}{drng:>22}"
              f"{f'[{min(fk):.3f},{max(fk):.3f}]':>22}{spearman(ek, fk):>9.4f}")

    # 对照：只看"没被训坏"的那一半（full acc 高于中位数）
    med = sorted(f)[len(f) // 2]
    print(f"\n{'—'*76}\n按 full acc 中位数({med:.4f}) 二分，各半单独算 ρ（含置换检验）")
    print(f"{'子集':<34}{'n':>4}{'full 范围':>18}{'ρ':>9}{'p(置换)':>10}")
    for label, idx in (("full acc ≥ 中位（没被训坏的一半）",
                        [i for i in range(len(f)) if f[i] >= med]),
                       ("full acc < 中位（被训坏的一半）",
                        [i for i in range(len(f)) if f[i] < med])):
        ei, fi = [e[i] for i in idx], [f[i] for i in idx]
        r = spearman(ei, fi)
        p = perm_p(ei, fi)
        print(f"{label:<34}{len(idx):>4}{f'[{min(fi):.3f},{max(fi):.3f}]':>18}"
              f"{r:>9.4f}{p:>10.4f}")
    print("\n置换检验 = 固定 early、把 full 的配对关系随机打乱 20000 次，"
          "统计 |ρ| ≥ 实测值的比例（单样本小，不靠渐近近似）。")


if __name__ == "__main__":
    main()
