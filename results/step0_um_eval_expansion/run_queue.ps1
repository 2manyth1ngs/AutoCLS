# step0 UM eval-expansion probe launcher (ASCII only, see ps51-bomless lesson)
# Records nvidia-smi state before the run (protocol section 8 requirement).
$ErrorActionPreference = "Continue"
Set-Location "D:\DevelopDoc\CLS"
$py = "D:\Anaconda\envs\pytorch\python.exe"
$log = "results\step0_um_eval_expansion\run.log"
"=== launch $(Get-Date -Format o) ===" | Out-File -Append -Encoding utf8 $log
nvidia-smi --query-gpu=name,memory.used,temperature.gpu,utilization.gpu --format=csv 2>&1 |
    Out-File -Append -Encoding utf8 $log
& $py scripts\11_step0_um_expansion.py --run --resume 2>&1 |
    Out-File -Append -Encoding utf8 $log
"=== done $(Get-Date -Format o) exit=$LASTEXITCODE ===" | Out-File -Append -Encoding utf8 $log
