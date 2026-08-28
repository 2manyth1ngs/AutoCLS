# eval_size_probe queue (protocol: results/eval_size_probe/PROTOCOL.md)
# ASCII-only comments on purpose (PS5.1 BOM-less GBK incident, 2026-08-11).
# Single stage: 60 retrained R' (HW@60, canonical + noise1/2 seeds) with
# per-sample correctness vectors for offline nested-subset scaling analysis.
$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\Anaconda\envs\pytorch\python.exe"
Set-Location "D:\DevelopDoc\CLS"
$log = "results\eval_size_probe\run.log"
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) RUN START ==="
& $py scripts\10_eval_size_probe.py --run --resume *>> $log
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) RUN EXIT $LASTEXITCODE ==="
