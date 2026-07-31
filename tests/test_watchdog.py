"""看门狗（scripts/cloud_run.sh）测试：自动重拉 + 毒条目永久隔离 + resume 跳过。

Part A-3（方案 B-3；memory：后台长训练两次被外部杀，云上还有 spot 抢占/OOM-killer，
SIGKILL 不可 catch）。看门狗只管进程级崩溃：非零退出 → 带 --resume 重拉；同一
inflight 条目连崩 ≥FAIL_LIMIT → 向 R' 缓存写 r_prime=null 的隔离记录（--resume 据
缓存键跳过，防"跑到坏条目→崩→重拉→又崩"死循环）；总重启封顶 MAX_RESTARTS。

bash 集成测试注入假 worker（模拟"跑到坏条目就崩"，被隔离后正常完成）——本地
Git Bash 与云 Linux bash 同一脚本；kill -9 级别的杀进程与 exit 1 对看门狗等价
（都是非零退出），故 CI 里用 exit 1 模拟即可，真 kill 演练见本地验证。
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from autocls.labeling import load_rprime_cache

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cloud_run.sh"


def _find_bash() -> str | None:
    """优先 Git Bash（system32 的 bash.exe 是 WSL，可能无发行版）；Linux 用 which。"""
    for cand in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if Path(cand).exists():
            return cand
    return shutil.which("bash")


BASH = _find_bash()

# 假 worker：坏条目未被隔离 → 写 inflight 后 exit 1（模拟崩溃）；
# 缓存中已有隔离记录 → 跳过它、清 inflight、exit 0（模拟 --resume 正常完成）。
FAKE_WORKER = """\
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
cache = out / "rprime_cache.jsonl"
quarantined = False
if cache.exists():
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task_id") == "T_00" and r.get("cls_key") == "badc0de":
            quarantined = True
if quarantined:
    (out / "inflight.txt").write_text("", encoding="utf-8")
    print("resume: poison entry quarantined, skipped, done")
    sys.exit(0)
(out / "inflight.txt").write_text("T_00\\tbadc0de\\t290", encoding="utf-8")
print("crashing on T_00/badc0de")
sys.exit(1)
"""


def _run_watchdog(out: Path, worker: Path, *, max_restarts: int, fail_limit: int = 3):
    env = {
        **os.environ,
        "CMD": f"{Path(sys.executable).as_posix()} {worker.as_posix()} {out.as_posix()}",
        "OUT": out.as_posix(),
        "FAIL_LIMIT": str(fail_limit),
        "MAX_RESTARTS": str(max_restarts),
        "SLEEP": "0",
    }
    return subprocess.run([BASH, str(SCRIPT)], env=env, cwd=ROOT,
                          capture_output=True, text=True, timeout=180,
                          encoding="utf-8", errors="replace")  # 看门狗日志是 UTF-8 中文


@pytest.mark.skipif(BASH is None, reason="需要 bash（Git Bash / Linux）")
def test_watchdog_quarantines_poison_then_completes(tmp_path):
    out = tmp_path / "labels"
    out.mkdir()
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER, encoding="utf-8")

    r = _run_watchdog(out, worker, max_restarts=10, fail_limit=3)
    assert r.returncode == 0, f"看门狗未正常收尾:\n{r.stdout}\n{r.stderr}"

    # 连崩 3 次 → 隔离记录落缓存（r_prime=null + watchdog error 标记）
    cache = load_rprime_cache(out / "rprime_cache.jsonl")
    key = ("T_00", "badc0de", 290)
    assert key in cache
    assert cache[key]["r_prime"] is None
    assert "watchdog" in cache[key]["error"]

    # 崩溃计数：state 文件 3 行同条目
    state = (out / "watchdog_state.tsv").read_text(encoding="utf-8").strip().splitlines()
    assert len(state) == 3
    # 3 次崩 + 1 次 resume 完成 = 4 次拉起，每次独立日志（不覆盖）
    logs = sorted(p.name for p in out.glob("worker_*.log"))
    assert len(logs) == 4
    assert "隔离" in r.stdout


@pytest.mark.skipif(BASH is None, reason="需要 bash（Git Bash / Linux）")
def test_watchdog_stops_at_max_restarts(tmp_path):
    # worker 永远崩且不留 inflight（如环境坏了）→ 重启封顶后退出 1 告警，不无限循环
    out = tmp_path / "labels"
    out.mkdir()
    worker = tmp_path / "always_crash.py"
    worker.write_text("import sys; sys.exit(1)\n", encoding="utf-8")

    r = _run_watchdog(out, worker, max_restarts=2)
    assert r.returncode == 1
    assert len(list(out.glob("worker_*.log"))) == 2
    assert "上限" in r.stdout


def test_resume_skips_quarantined_entry(tmp_path):
    # 隔离记录走同一缓存键协议：load_rprime_cache 命中 → 2b/step0 的 todo 过滤跳过
    cache_path = tmp_path / "rprime_cache.jsonl"
    cache_path.write_text(json.dumps({
        "task_id": "T_00", "cls_key": "badc0de", "iters": 290,
        "r_prime": None, "error": "watchdog: quarantined after 3 crashes",
    }) + "\n", encoding="utf-8")
    cache = load_rprime_cache(cache_path)
    assert ("T_00", "badc0de", 290) in cache          # resume 视作已完成 → 跳过
    assert cache[("T_00", "badc0de", 290)]["r_prime"] is None  # 配对侧按失败 CLS 剔除
