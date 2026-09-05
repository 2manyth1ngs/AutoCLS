# WP8-BO overnight runner (ASCII only: PS5.1 + BOM-less files mangle non-ASCII). Each step is resumable.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location "D:\DevelopDoc\CLS"
$py = "D:\Anaconda\envs\pytorch\python.exe"
$log = "results\mx4_bo\loop.log"
$uea = "Handwriting HandMovementDirection UWaveGestureLibrary Cricket EthanolConcentration PhonemeSpectra"
Add-Content -Path $log -Value ("=== run_all start " + (Get-Date -Format s))
cmd /c "$py scripts\43_bo_nominate.py --loop --resume --datasets $uea >> $log 2>&1"
Add-Content -Path $log -Value ("=== uea loop done " + (Get-Date -Format s) + " exit " + $LASTEXITCODE)
cmd /c "$py scripts\43_bo_nominate.py --finals --resume --datasets $uea >> $log 2>&1"
Add-Content -Path $log -Value ("=== uea finals done " + (Get-Date -Format s) + " exit " + $LASTEXITCODE)
cmd /c "$py scripts\43_bo_nominate.py --loop --resume --datasets HAR >> $log 2>&1"
Add-Content -Path $log -Value ("=== har loop done " + (Get-Date -Format s) + " exit " + $LASTEXITCODE)
cmd /c "$py scripts\43_bo_nominate.py --finals --resume --datasets HAR >> $log 2>&1"
Add-Content -Path $log -Value ("=== har finals done " + (Get-Date -Format s) + " exit " + $LASTEXITCODE)
cmd /c "$py scripts\43_bo_nominate.py --analyze --datasets $uea HAR >> $log 2>&1"
Add-Content -Path $log -Value ("=== analyze done " + (Get-Date -Format s) + " exit " + $LASTEXITCODE)
Set-Content -Path "results\mx4_bo\DONE.txt" -Value (Get-Date -Format s)
