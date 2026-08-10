# iters 工作点扫描 fill 队列（协议 results/iters_workpoint/PROTOCOL.md §1）
# 顺序执行（单 GPU 作业：8 GB 并发两作业会互相拖死，记忆条目已证）；
# 每点 --resume 可续；iters 在缓存键里与历史 290/2320 隔离。
$ErrorActionPreference = "Continue"
$env:OMP_NUM_THREADS = "1"          # 不设会慢 4.2x（route-a 吞吐结论）
$env:PYTHONIOENCODING = "utf-8"
$py = "D:\Anaconda\envs\pytorch\python.exe"
Set-Location "D:\DevelopDoc\CLS"
$log = "results\iters_workpoint\run.log"

function Invoke-Fill($registry, $outdir, $dataset, $itersList) {
    foreach ($it in $itersList) {
        $argv = @("scripts\baseline_probe.py", "--fill-trained",
                  "--fill-iters", "$it", "--datasets", $dataset,
                  "--outdir", $outdir, "--resume", "--backend", "threading")
        if ($registry) { $argv += @("--registry", $registry) }
        Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) $dataset iters=$it START ==="
        & $py @argv *>> $log
        Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) $dataset iters=$it EXIT $LASTEXITCODE ==="
    }
}

# 正对照先行（最便宜、判据 6 直接吃它），候选按边缘优先
Invoke-Fill $null "results\iters_workpoint\uea" "Handwriting" @(3, 6, 15, 60, 120)
Invoke-Fill "configs\candidate_pool.yaml" "results\iters_workpoint\candidates" "UniMiB-SHAR" @(6, 12, 30, 60, 120)
Invoke-Fill "configs\candidate_pool.yaml" "results\iters_workpoint\candidates" "InsectWingbeatSound" @(6, 12, 60, 120)
Invoke-Fill "configs\candidate_pool.yaml" "results\iters_workpoint\candidates" "WISDM2019" @(6, 12, 60, 120)
Invoke-Fill "configs\candidate_pool.yaml" "results\iters_workpoint\candidates" "PAMAP2" @(6, 12, 30, 60, 120)
Add-Content -Path $log -Encoding UTF8 -Value "=== $(Get-Date -Format s) ALL DONE ==="
