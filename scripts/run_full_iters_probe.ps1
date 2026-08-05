# full-iters 判别实验驱动（协议 results/full_iters_probe/PROTOCOL.md）
#
# 顺序按"决定性对照先落地"排（协议 §7）：低 iters 臂 → Handwriting（对照，最便宜）
# → WISDM2019 → InsectWingbeatSound。每条即时落盘 + --resume，中断可续跑。
#
# 本机长跑会被外部杀掉，故由 Start-Process 脱离启动本脚本，不要用 run_in_background。
# CPU-bound 负载：OMP/MKL 必须为 1（不设慢 4.2×），joblib 用 threading 后端。

$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = 1
$env:MKL_NUM_THREADS = 1
$env:PYTHONIOENCODING = "utf-8"

$PY = "D:\Anaconda\envs\pytorch\python.exe"
$ROOT = "D:\DevelopDoc\CLS"
$LOG = "$ROOT\results\full_iters_probe\run.log"
Set-Location $ROOT

function Run-Fill {
    param([string]$Name, [int]$Iters, [string]$Outdir, [string]$Registry)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LOG -Encoding utf8 -Value "`n===== [$stamp] $Name @ iters=$Iters ====="
    $a = @("scripts/baseline_probe.py", "--fill-trained", "--fill-iters", $Iters,
           "--datasets", $Name, "--outdir", $Outdir, "--resume", "--backend", "threading")
    if ($Registry) { $a += @("--registry", $Registry) }
    & $PY $a 2>&1 | Tee-Object -FilePath $LOG -Append
}

$CAND = "results/full_iters_probe/candidates"
$UEA  = "results/full_iters_probe/uea"
$CREG = "configs/candidate_pool.yaml"

Add-Content -Path $LOG -Encoding utf8 -Value "########## full-iters 判别实验开跑 $(Get-Date) ##########"

# --- 臂 2（协议 §1.1）：iters=30 低训练量对照，~3 min ---
Run-Fill -Name "Handwriting"         -Iters 30 -Outdir $UEA
Run-Fill -Name "WISDM2019"           -Iters 30 -Outdir $CAND -Registry $CREG
Run-Fill -Name "InsectWingbeatSound" -Iters 30 -Outdir $CAND -Registry $CREG

# --- 主臂（协议 §1）：iters=2320，决定性对照 Handwriting 先跑 ---
Run-Fill -Name "Handwriting"         -Iters 2320 -Outdir $UEA
Run-Fill -Name "WISDM2019"           -Iters 2320 -Outdir $CAND -Registry $CREG
Run-Fill -Name "InsectWingbeatSound" -Iters 2320 -Outdir $CAND -Registry $CREG

Add-Content -Path $LOG -Encoding utf8 -Value "########## 全部完成 $(Get-Date) ##########"
