#!/usr/bin/env bash
# WP17-ACTS S3-N 第一阶段：服务器 CPU 计算 160 任务的 E_A（response_profile 25 维）→ ea.npz → upload/out_acts_s3n_ea.tar.gz。零 GPU。
#   NPART=24 setsid nohup bash .../scripts/pa_pod/acts_lab_run_s3n_ea.sh > /tmp/acts_s3n_out/logs/run.log 2>&1 < /dev/null &
set -uo pipefail
REPO="/home/b311/data/wangwenjie/kejunyu/AutoCLS"; T="/tmp/acts_s3n_out"; LOG="$T/logs"; UP="$REPO/upload"; OUT="results/acts/s3n"
NPART="${NPART:-24}"
cd "$REPO"; mkdir -p "$LOG" "$UP"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate pytorch
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
tar --no-same-owner -xf "$UP/code_head.tar" -C "$REPO" autocls scripts configs tests
find autocls scripts configs tests -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) -print0 | xargs -0 sed -i 's/\r$//'
tar --no-same-owner -xzf "$UP/acts_s3n_inputs.tar.gz" -C "$REPO"
echo "== HEAD $(cat "$UP/HEAD.txt")"
mkdir -p "$OUT" "$T/ea"; [ -L "$OUT/ea" ] || { rm -rf "$OUT/ea"; ln -s "$T/ea" "$OUT/ea"; }
python -m pytest tests/test_acts_s3n.py -q 2>&1 | tail -1 | tee "$LOG/pytest.log"; grep -q passed "$LOG/pytest.log" || exit 7
echo "== [$(date -u +%H:%M:%S)] E_A $NPART 份"
pids=(); for ((p=0; p<NPART; p++)); do ( nice -n 10 python scripts/101_acts_s3n.py --compute-ea --parts $NPART --part $p --outdir $OUT > "$LOG/ea_p$p.log" 2>&1 ) & pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
echo "== [$(date -u +%H:%M:%S)] E_A 完：$(cat $OUT/ea/part*.jsonl | wc -l) / 160"
python scripts/101_acts_s3n.py --merge-ea --outdir $OUT 2>&1 | tee "$LOG/merge.log"
tar -chzf "$UP/out_acts_s3n_ea.tar.gz" $OUT/ea.npz $OUT/ea "$LOG"/*.log 2>/dev/null
ls -la "$UP/out_acts_s3n_ea.tar.gz"
echo "== [$(date -u +%H:%M:%S)] acts s3n ea done"
