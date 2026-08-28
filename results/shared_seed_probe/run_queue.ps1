# shared_seed_probe queue (protocol: results/shared_seed_probe/PROTOCOL.md)
# ASCII-only comments on purpose (PS5.1 BOM-less GBK incident, 2026-08-11).
# Single stage: 60 fresh R' (HW@60, 2 tasks x 10 shared CLS x 3 shared seeds).
$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\Anaconda\envs\pytorch\python.exe"
Set-Location "D:\DevelopDoc\CLS"
$log = "results\shared_seed_probe\run.log"
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) RUN START ==="
& $py scripts\9_shared_seed_probe.py --run --resume *>> $log
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) RUN EXIT $LASTEXITCODE ==="
