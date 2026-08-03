"""阶段1 单卡调度器：按 l_enc 分档并发跑 step0 SNR 闸 + 串行收尾 + 判读。

为什么是单卡多进程（GATE_5090_REPORT §1.9 标定推翻了 4 卡前提）：本负载 **CPU-bound**
（`cl_trainer.py:50` 数据增强在 CPU 上做，GPU 只跑 encoder 前向，单 worker GPU 利用率
5.7%），**1 张卡 8 worker（15.2 条/分）> 4 张卡各 1（11.6 条/分）**；而 4 卡会抢同一份
cgroup 54.4 核 CPU。

并行粒度 = 数据集（一个数据集一个 `cloud_run.sh` 看门狗进程）：
- 天花板加难重探在各自进程内闭环，跨进程零竞争；
- 每进程独立 `OUT`（看门狗日志/state/inflight）与 `tasks_meta.d/<ds>.json`；
- 共享 `results/labels_route_a/rprime_cache.jsonl`（canonical，阶段2 逐键复用）与
  `noise_cache.jsonl`——行追加，单行 ≤706 B < POSIX 4096 B 原子追加边界。

三段式（对应用户要求的"OOM 降 worker 重试"）：
  ① 并发段：按 configs/runtime.yaml 分档 + 显存预算准入，`--on-oom skip`
     （OOM 不写缓存、不计毒条目，该条保持"未跑"）；
  ② 收尾段：**worker=1 串行**逐数据集重跑一遍 `--resume --on-oom record`，
     ①里被 OOM 跳过的条目在这里独占显存重试；仍 OOM 才算终局失败，
     写 `results/step0_snr_gate/oom_failures.jsonl` 显式可见；
  ③ 判读：`--analyze` 出 analysis.json + 任务数下限闸。

用法（脱离会话跑，见手册 §6）：
    python scripts/run_stage1.py --dry-run      # 预检表 + 排程 + ETA，不花钱
    tmux new -d -s stage1 'python scripts/run_stage1.py 2>&1 | tee results/step0_snr_gate/stage1.log'
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autocls.cloud_plan import (  # noqa: E402
    admission_order,
    budget_gib,
    can_admit,
    estimate_item_seconds,
    estimate_peak_gib,
    load_runtime,
    preflight,
    simulate_wall_clock,
    stage1_datasets,
    worker_env,
    workers_for,
)
from autocls.data.source_pool import load_registry  # noqa: E402

OUTDIR = ROOT / "results" / "step0_snr_gate"
WD_DIR = OUTDIR / "wd"                                  # 每数据集一个看门狗工作目录
SHARED_CACHE = ROOT / "results" / "labels_route_a" / "rprime_cache.jsonl"
NOISE_CACHE = OUTDIR / "noise_cache.jsonl"
OOM_LOG = OUTDIR / "oom_failures.jsonl"

# step0 探针协议：每数据集 n_tasks(2) × n_cls(10) × n_seeds(4)（ProbeCfg 默认档）
PLANNED_ITEMS_PER_DATASET = 80


# ---------------------------------------------------------------------------
# 进程构造（错一个环境变量名 = 静默跑错配置，故单测覆盖）
# ---------------------------------------------------------------------------


def watchdog_cmd() -> list[str]:
    """看门狗启动命令。Linux 上就是 `bash scripts/cloud_run.sh`；Windows 上要绕开
    System32 的 WSL bash（它把 `D:\\a\\b.sh` 解释成 `D:ab.sh`，rc=127），
    好让整条启动链在本机也能先验一遍再上云。"""
    exe = "bash"
    if os.name == "nt":
        for cand in (r"C:\Program Files\Git\bin\bash.exe",
                     r"C:\Program Files\Git\usr\bin\bash.exe"):
            if Path(cand).exists():
                exe = cand
                break
        else:
            exe = shutil.which("bash") or "bash"
    return [exe, (ROOT / "scripts" / "cloud_run.sh").as_posix()]


def spawn_env(name: str, cfg: dict, *, outdir: Path = OUTDIR,
              device: str = "cuda:0") -> dict:
    """一个数据集的看门狗进程环境（`scripts/cloud_run.sh` 的入参）。"""
    out = Path(outdir) / "wd" / name
    return {
        **os.environ,
        **worker_env(cfg),                        # OMP/MKL=1（§3.1 硬性启动条件）
        # 显式钉死解释器：cloud_run.sh 默认 PY=python 靠 PATH 解析，装了多套 python
        # （或 Windows 的 App Execution Alias 占位）时会拉起错的那个，且失败得很难看懂
        "PY": sys.executable,
        "STAGE": "step0",
        "STEP0_DATASETS": name,
        "OUT": out.as_posix(),
        "CACHE": SHARED_CACHE.as_posix(),         # step0 canonical 并入全量缓存
        "EXTRA_ARGS": (f"--device {device} --inflight {(out / 'inflight.txt').as_posix()} "
                       f"--on-oom skip --oom-log {OOM_LOG.as_posix()}"),
    }


def finisher_argv(name: str, *, outdir: Path = OUTDIR, device: str = "cuda:0") -> list[str]:
    """收尾段：worker=1 串行重跑一个数据集，仍 OOM 即终局失败并落盘可见。"""
    out = Path(outdir) / "wd" / name
    return [sys.executable, "-u", str(ROOT / "scripts" / "step0_snr_gate.py"),
            "--run", "--resume", "--datasets", name, "--device", device,
            "--inflight", (out / "inflight.txt").as_posix(),
            "--on-oom", "record", "--oom-log", OOM_LOG.as_posix()]


def remaining_items(name: str, caches: list[Path], planned: int) -> int:
    """该数据集还剩多少条 R' 没跑（ETA 用；按 (task_id, cls_key, seed) 去重计数）。

    天花板重探会给同一数据集多写一批（+c 档标任务），已完成数可能超过 planned——
    钳到 0，不产生负的 ETA。
    """
    done: set[tuple] = set()
    for p in caches:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("dataset") != name:
                continue
            done.add((r.get("task_id"), r.get("cls_key"), r.get("seed")))
    return max(0, planned - len(done))


# ---------------------------------------------------------------------------
# 预检 + 排程
# ---------------------------------------------------------------------------


def print_preflight(names: list[str], registry, cfg: dict) -> dict[str, tuple[int, int]]:
    """启动前预检表（显存 + 分档 + 待跑条数 + ETA），返回 {ds: (l_enc, 待跑条数)}。"""
    rows = preflight(names, registry, cfg)
    caches = [SHARED_CACHE, NOISE_CACHE]
    jobs: dict[str, tuple[int, int]] = {}
    print(f"\n{'='*96}\n阶段1 启动前预检（configs/runtime.yaml；显存预算 "
          f"{budget_gib(cfg):.1f}/{cfg['gpu']['vram_gib']} GiB，全局并发上限 "
          f"{cfg['workers']['max_concurrent']}）\n{'='*96}")
    print(f"{'数据集':<28}{'l_enc':>6}{'分档':>5}{'峰值GiB':>9}{'秒/条':>7}"
          f"{'待跑':>6}{'预计':>9}  告警")
    for r in rows:
        n = r["dataset"]
        left = remaining_items(n, caches, PLANNED_ITEMS_PER_DATASET)
        jobs[n] = (r["l_enc"], left)
        secs = estimate_item_seconds(r["l_enc"], cfg)
        print(f"{n:<28}{r['l_enc']:>6}{r['workers']:>5}{r['peak_gib']:>9.1f}{secs:>7.0f}"
              f"{left:>6}{left * secs / 3600:>8.1f}h  {r['warn'] or ''}")
    total_serial = sum(n * estimate_item_seconds(l, cfg) for l, n in jobs.values())
    wall = simulate_wall_clock(jobs, cfg)
    print(f"{'—'*96}\n单 worker 串行总量 {total_serial/3600:.1f} h → 按分档并发预计墙钟 "
          f"**{wall/3600:.1f} h**（+ 天花板重探每轮 80 条/数据集）")
    long_chain = [n for n, (l, _) in jobs.items() if workers_for(l, cfg) == 1]
    if long_chain:
        chain = sum(jobs[n][1] * estimate_item_seconds(jobs[n][0], cfg) for n in long_chain)
        print(f"长杆：{', '.join(long_chain)} 互斥串行 {chain/3600:.1f} h"
              f"（占墙钟 {chain/max(wall, 1):.0%}）——想再快只能加卡")
    return jobs


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[stage1 {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def run_concurrent(jobs: dict[str, tuple[int, int]], cfg: dict, *, device: str,
                   poll: float = 20.0) -> None:
    """① 并发段：按 can_admit 逐个准入，每数据集一个 cloud_run.sh 看门狗进程。"""
    order = admission_order(list(jobs), cfg, length_of=lambda n: jobs[n][0])
    # 已跑完的（待跑 0）排最后：它们只需再进一次做重探判定，别占着长杆的显存
    pending = [n for n in order if jobs[n][1] > 0] + [n for n in order if jobs[n][1] == 0]
    running: dict[str, tuple[subprocess.Popen, int]] = {}
    t0 = time.time()

    while pending or running:
        admitted = True
        while admitted and pending:
            admitted = False
            for name in list(pending):
                l_enc = jobs[name][0]
                ok, why = can_admit([l for _, l in running.values()], l_enc, cfg)
                if not ok:
                    continue
                WD_DIR.joinpath(name).mkdir(parents=True, exist_ok=True)
                log_path = WD_DIR / name / "watchdog.log"
                fh = open(log_path, "a", encoding="utf-8")
                p = subprocess.Popen(
                    watchdog_cmd(),
                    cwd=str(ROOT), env=spawn_env(name, cfg, device=device),
                    stdout=fh, stderr=subprocess.STDOUT)
                running[name] = (p, l_enc)
                pending.remove(name)
                admitted = True
                used = sum(estimate_peak_gib(l, cfg) for _, l in running.values())
                _log(f"起 {name}（L={l_enc}，待跑 {jobs[name][1]} 条，"
                     f"占用 {used:.1f}/{budget_gib(cfg):.1f} GiB，在跑 {len(running)} 个）"
                     f" → {log_path}")
        if not running:
            break
        time.sleep(poll)
        for name in [n for n, (p, _) in running.items() if p.poll() is not None]:
            p, l_enc = running.pop(name)
            _log(f"完 {name} rc={p.returncode}（已跑 {(time.time()-t0)/3600:.1f} h，"
                 f"剩 {len(pending)} 个待起 / {len(running)} 个在跑）"
                 + ("" if p.returncode == 0 else "  ⚠️ 看门狗非零退出，见其 watchdog.log"))
    _log(f"并发段结束，用时 {(time.time()-t0)/3600:.2f} h")


def run_finisher(names: list[str], *, device: str) -> None:
    """② 收尾段：worker=1 串行逐数据集重跑（OOM 重试 + 补完未跑完的重探轮）。"""
    for name in names:
        _log(f"收尾（worker=1，--on-oom record）{name}")
        log_path = WD_DIR / name / "finisher.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            rc = subprocess.call(finisher_argv(name, device=device), cwd=str(ROOT),
                                 env={**os.environ, **worker_env(load_runtime())},
                                 stdout=fh, stderr=subprocess.STDOUT)
        if rc != 0:
            _log(f"⚠️ {name} 收尾非零退出 rc={rc} → {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="只打印预检表/排程/ETA")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="覆盖阶段1 对象（默认未实测源 + reserve，按 l_enc 降序）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-concurrent", action="store_true", help="只跑收尾段 + 判读")
    ap.add_argument("--skip-finisher", action="store_true")
    ap.add_argument("--skip-analyze", action="store_true")
    args = ap.parse_args()

    registry = load_registry()
    cfg = load_runtime()
    names = args.datasets or stage1_datasets(registry)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    SHARED_CACHE.parent.mkdir(parents=True, exist_ok=True)   # 手册 §5 订正：漏建目录会静默失败

    jobs = print_preflight(names, registry, cfg)
    if args.dry_run:
        return

    if not args.skip_concurrent:
        run_concurrent(jobs, cfg, device=args.device)
    if not args.skip_finisher:
        run_finisher(names, device=args.device)
    if OOM_LOG.exists():
        n = len([x for x in OOM_LOG.read_text(encoding="utf-8").splitlines() if x.strip()])
        _log(f"⚠️ {n} 条即使 worker=1 也显存不足 → {OOM_LOG}（这些条目没有 R'，判读前须处置）")
    if not args.skip_analyze:
        _log("判读 --analyze")
        subprocess.call([sys.executable, "-u", str(ROOT / "scripts" / "step0_snr_gate.py"),
                         "--analyze", "--datasets", *names], cwd=str(ROOT),
                        env={**os.environ, **worker_env(cfg)})


if __name__ == "__main__":
    main()
