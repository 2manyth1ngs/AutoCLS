"""S3-MD 接管：列出 plan_W 里仍未完成的任务（每行 "task_id 未完成格数"），可排除仍在跑的原分片的任务。
用法：python s3md_tasks.py [排除的原分片号 ...]   （原分片 = 启动时 --parts 4 的划分，当时计划格 0 已完成）"""
import importlib, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, ".")
g83 = importlib.import_module("scripts.83_acts_corpus"); ac = importlib.import_module("scripts.acts_common")
OUT = Path("results/acts/s3md/runs")
cells = json.loads((OUT / "plan_W.json").read_text(encoding="utf-8"))["cells"]
done = set(g83._g35.load_done(OUT / "runs.jsonl"))
excl = set()
if len(sys.argv) > 1:
    parts = g83.part_tasks(cells, set(), 4, ac.load_acts_config(str(ac.ACTS_YAML), profile="s1pp"))
    for k in sys.argv[1:]:
        excl |= set(parts[int(k)])
cnt = Counter(c["task_id"] for c in cells if c["key"] not in done and c["task_id"] not in excl)
for t, n in sorted(cnt.items(), key=lambda x: -x[1]):
    print(t, n)
