# workpoint followup fill 队列（协议 results/workpoint_followup/PROTOCOL.md §1/§3）
# 顺序执行（8 GB 并发两作业互相拖死，记忆条目已证）；每点 --resume 可续；
# iters 入缓存键；既有 100 条已复制入 outdir，resume 逐键跳过。
$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = "1"          # 不设会慢 4.2x（route-a 吞吐结论）
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\Anaconda\envs\pytorch\python.exe"
Set-Location "D:\DevelopDoc\CLS"
$log = "results\workpoint_followup\run.log"

function Invoke-Fill($dataset, $itersList, $fillTasks) {
    foreach ($it in $itersList) {
        $argv = @("scripts\baseline_probe.py", "--fill-trained",
                  "--fill-iters", "$it", "--fill-tasks", "$fillTasks",
                  "--datasets", $dataset,
                  "--registry", "configs\candidate_pool.yaml",
                  "--outdir", "results\workpoint_followup\candidates",
                  "--resume", "--backend", "threading")
        Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) $dataset iters=$it tasks=$fillTasks START ==="
        & $py @argv *>> $log
        Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) $dataset iters=$it EXIT $LASTEXITCODE ==="
    }
}

# 任务2 加测先行（确认数据最关键、也最便宜）
Invoke-Fill "InsectWingbeatSound" @(6, 12) 6
Invoke-Fill "UniMiB-SHAR" @(12, 30, 60) 6
# 任务1 重扫（每数据集 5 臂 x 20 条）
Invoke-Fill "ChestAccel" @(6, 12, 30, 60, 120) 2
Invoke-Fill "USC-HAD" @(6, 12, 30, 60, 120) 2
Invoke-Fill "ElectricDevices" @(6, 12, 30, 60, 120) 2
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) ALL DONE ==="
