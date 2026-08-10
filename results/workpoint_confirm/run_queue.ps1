# workpoint confirm fill 队列 · 段1（协议 results/workpoint_confirm/PROTOCOL.md §1/§7）
# 顺序执行（8 GB 并发两作业互相拖死）；每点 --resume 可续；iters 入缓存键；
# Handwriting @60/@120 的任务 00-01 种子已由 seed_fill.py 幂等写入，resume 逐键跳过。
# 段2（UWave 确认臂）由选点纯函数出结果后另行启动（run_queue_stage2.ps1）。
$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = "1"          # 不设会慢 4.2x（route-a 吞吐结论）
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\Anaconda\envs\pytorch\python.exe"
Set-Location "D:\DevelopDoc\CLS"
$log = "results\workpoint_confirm\run.log"

function Invoke-Fill($dataset, $itersList, $fillTasks) {
    foreach ($it in $itersList) {
        $argv = @("scripts\baseline_probe.py", "--fill-trained",
                  "--fill-iters", "$it", "--fill-tasks", "$fillTasks",
                  "--datasets", $dataset,
                  "--outdir", "results\workpoint_confirm\uea",
                  "--resume", "--backend", "threading")
        Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) $dataset iters=$it tasks=$fillTasks START ==="
        & $py @argv *>> $log
        Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) $dataset iters=$it EXIT $LASTEXITCODE ==="
    }
}

# Handwriting 确认先行（主判定臂 @120 优先落盘），再 UWave 扫描（便宜臂在前）
Invoke-Fill "Handwriting" @(120, 60) 6
Invoke-Fill "UWaveGestureLibrary" @(3, 6, 15, 30, 60) 2
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) STAGE1 DONE ==="
