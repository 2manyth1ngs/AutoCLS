# workpoint confirm stage2: UWave confirm arm (PROTOCOL.md S2 selection: iters=15)
# Selected by select_workpoint on stage-1 data: candidates [15,60,290], 0.02-margin tie -> fewest iters.
$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\Anaconda\envs\pytorch\python.exe"
Set-Location "D:\DevelopDoc\CLS"
$log = "results\workpoint_confirm\run.log"

Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) UWaveGestureLibrary iters=15 tasks=6 START ==="
& $py scripts\baseline_probe.py --fill-trained --fill-iters 15 --fill-tasks 6 --datasets UWaveGestureLibrary --outdir results\workpoint_confirm\uea --resume --backend threading *>> $log
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) UWaveGestureLibrary iters=15 EXIT $LASTEXITCODE ==="
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) STAGE2 DONE ==="
