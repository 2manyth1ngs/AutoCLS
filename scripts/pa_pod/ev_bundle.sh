#!/usr/bin/env bash
# WP16 EV1 检验上云（5090，SetTF 段）：本地打包（Git Bash）。pod 上跑：77 --fit-folds → --train-es（2 进程分折）→ --evolve ev_es → --rprime ev_es → --verify → --rank-es → 打包。
#   bash scripts/pa_pod/ev_bundle.sh <bundle_dir>
set -euo pipefail
B="${1:?bundle dir}"
mkdir -p "$B"
INPUTS="results/tclsc_ev/PROTOCOL.md results/tclsc_ev/proposals_ev_et.json results/tclsc_ev/proposals_ev_blind.json results/tclsc_ev/proposals_rand_u.json results/tclsc_nb/nb_catalog.json results/tclsc_nb/labels_train.jsonl results/tclsc_nb/labels_truth_k1.jsonl results/tclsc_nb/rprime_nb.jsonl \
results/mx3_pq/rprime_cache.jsonl results/mx3_pq/noise_probe.jsonl results/mx3_pq/delta_calibration.json results/mx3_pq/cls_catalog.json results/mx3_pq/tasks.json \
results/tclsc_slices/tasks.json results/tclsc_slices/cls_catalog.json results/tclsc_slices/rprime_cache.jsonl results/route_a_probe/probe_cls_catalog.json \
results/tclsc_k3/labels_k3.jsonl data/candidates/UniMiB-SHAR.npz"
for f in $INPUTS; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
git -c core.autocrlf=false archive --format=tar -o "$B/code_head.tar" HEAD autocls scripts configs tests
tar --force-local -czf "$B/ev_inputs.tar.gz" $INPUTS
: > "$B/md5_uea_ts.txt"
for d in Handwriting PhonemeSpectra UWaveGestureLibrary EthanolConcentration Cricket HandMovementDirection SelfRegulationSCP2; do
  for s in TRAIN TEST; do
    f="data/uea_cache/$d/${d}_$s.ts"; [ -f "$f" ] || { echo "MISSING $f"; exit 2; }
    printf '%s  %s\n' "$(md5sum "$f" | cut -d' ' -f1)" "$d/${d}_$s.ts" >> "$B/md5_uea_ts.txt"
  done
done
git rev-parse HEAD > "$B/HEAD.txt"
ls -la "$B"
