#!/usr/bin/env bash
# WP16 GEO-S 检验上云（5090）：本地打包（Git Bash）。pod 上跑：82 --run（580 条官方轨迹，2 进程按估时分靶）→ --verify（不过补跑）→ 打包。
#   bash scripts/pa_pod/geos_bundle.sh <bundle_dir>
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
INPUTS="results/tclsc_geos/PROTOCOL.md results/tclsc_geos/plan.json results/tclsc_s1et_deploy/et_official.jsonl results/tclsc_s1et_uea30/et_uea30.jsonl \
results/mx3_pq/rprime_cache.jsonl results/mx3_pq/cls_catalog.json results/mx3_pq/tasks.json results/route_a_probe/probe_cls_catalog.json"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/geos_inputs.tar.gz" $INPUTS
: > "$B/md5_uea_ts.txt"
for d in ArticularyWordRecognition AtrialFibrillation BasicMotions CharacterTrajectories Cricket DuckDuckGeese ERing EigenWorms Epilepsy EthanolConcentration FaceDetection FingerMovements HandMovementDirection Handwriting Heartbeat JapaneseVowels LSST Libras MotorImagery NATOPS PEMS-SF PenDigits PhonemeSpectra RacketSports SelfRegulationSCP1 SelfRegulationSCP2 SpokenArabicDigits StandWalkJump UWaveGestureLibrary; do
  for s in TRAIN TEST; do
    f="data/uea_cache/$d/${d}_$s.ts"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }
    printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$d/${d}_$s.ts" >> "$B/md5_uea_ts.txt"
  done
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
