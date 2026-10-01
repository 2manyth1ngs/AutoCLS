"""接管式补跑的任务列表：plan_<phase>.json 里仍未完成的任务（每行 "task_id 未完成格数"，按未完成数降序）。
用法：python scripts/pa_pod/acts_fill_tasks.py <outdir> <phase>   （S3-MD 的 acts_s3md_tasks.py 的通用版，供 acts_lab_run_xs.sh 的 fill 用）"""
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
g83 = importlib.import_module("scripts.83_acts_corpus")
outdir, phase = Path(sys.argv[1]), sys.argv[2]
cells = json.loads((outdir / f"plan_{phase}.json").read_text(encoding="utf-8"))["cells"]
done = set(g83._g35.load_done(outdir / "runs.jsonl"))
for t, n in sorted(Counter(c["task_id"] for c in cells if c["key"] not in done).items(), key=lambda x: (-x[1], x[0])):
    print(t, n)
