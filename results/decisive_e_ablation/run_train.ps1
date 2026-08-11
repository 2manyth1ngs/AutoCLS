# decisive_e_ablation training queue (protocol: PROTOCOL.md in this dir)
# ASCII-only comments (PS5.1 BOM-less GBK incident). Runs R1-R7 with --resume:
# completed runs (existing runs/*.json) are skipped.
$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\Anaconda\envs\pytorch\python.exe"
Set-Location "D:\DevelopDoc\CLS"
$log = "results\decisive_e_ablation\train.log"
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) TRAIN START ==="
& $py scripts\4b_decisive_e_ablation.py --train --resume *>> $log
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) TRAIN EXIT $LASTEXITCODE ==="
