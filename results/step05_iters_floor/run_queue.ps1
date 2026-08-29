# step0.5 iters floor scan launcher (ASCII only)
$ErrorActionPreference = "Continue"
Set-Location "D:\DevelopDoc\CLS"
$py = "D:\Anaconda\envs\pytorch\python.exe"
$log = "results\step05_iters_floor\run.log"
"=== launch $(Get-Date -Format o) ===" | Out-File -Append -Encoding utf8 $log
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,utilization.gpu --format=csv 2>&1 |
    Out-File -Append -Encoding utf8 $log
& $py scripts\12_step05_iters_floor_scan.py --run --resume 2>&1 |
    Out-File -Append -Encoding utf8 $log
"=== done $(Get-Date -Format o) exit=$LASTEXITCODE ===" | Out-File -Append -Encoding utf8 $log
