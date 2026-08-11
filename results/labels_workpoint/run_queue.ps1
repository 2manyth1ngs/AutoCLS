# labels_workpoint fill queue (protocol: results/decisive_e_ablation/PROTOCOL.md)
# ASCII-only comments on purpose (PS5.1 BOM-less GBK incident, 2026-08-11).
# Stages: noise-probe -> fill -> features. Each stage resumable; sequential
# (8 GB VRAM: never run two jobs concurrently).
$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\Anaconda\envs\pytorch\python.exe"
Set-Location "D:\DevelopDoc\CLS"
$log = "results\labels_workpoint\run.log"

function Invoke-Stage($name, $flags) {
    Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) $name START ==="
    $argv = @("scripts\2c_generate_labels_workpoint.py") + $flags
    & $py @argv *>> $log
    Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) $name EXIT $LASTEXITCODE ==="
}

Invoke-Stage "noise-probe" @("--noise-probe", "--resume")
Invoke-Stage "fill"        @("--fill", "--resume")
Invoke-Stage "features"    @("--features")
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) QUEUE DONE ==="
