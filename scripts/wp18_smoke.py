"""WP18 端到端冒烟（服务器）：1 个 dev 靶、2 个成员、小 Ks，跑通 train → fit → search → plan-cv → run-tgt → select → plan-final → run-tgt → ensemble → analyze。"""
import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
m = importlib.import_module("scripts.122_wp18")
m.OUT = ROOT / "results" / "acts" / "wp18" / "smoke"
m.MEMBERS = (0, 1)
m.KS = 1000
m.N_MEMBER_SEARCHERS = 1
m.TOPK_ENS = 3
m.ENS_KS = (1, 3)
m.FINAL_SEEDS = (0,)
cp = importlib.import_module("scripts.wp18_comp")
cp.MAX_EPOCHS = 2
d = sys.argv[1] if len(sys.argv) > 1 else "Cricket"
dev = sys.argv[2] if len(sys.argv) > 2 else "cuda"
A = dict(targets=[d], scope="dev", device=dev, resume=False, members=None, arms=["MEFV", "BLIND"], e_blocks=None, part=0, parts=1)
NS = lambda **k: argparse.Namespace(**{**A, **k})  # noqa: E731
(m.OUT / "ve.npz").parent.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copy2(ROOT / "results" / "acts" / "wp18" / "ve.npz", m.OUT / "ve.npz")
m.cmd_train(NS())
m.cmd_fit(NS())
m.cmd_search(NS())
m.cmd_plan_cv(NS())
import json
P = json.loads((m.OUT / "plan_cv_dev.json").read_text(encoding="utf-8"))
P["cells"] = P["cells"][:9]                      # 3 个候选 × 3 折
(m.OUT / "plan_cv_dev.json").write_text(json.dumps(P), encoding="utf-8")
m.cmd_run_tgt(NS(phase="cv"))
m.cmd_select(NS())
m.cmd_plan_final(NS())
m.cmd_run_tgt(NS(phase="final"))
m.cmd_ensemble(NS())
m.cmd_analyze(NS())
print("SMOKE OK")
