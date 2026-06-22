#!/usr/bin/env bash
# Agentic trace-replay bench against the 1P1D PROXY (:9100). Usage: g05_pd_bench.sh <CONC> [DURATION]
set -uo pipefail
CONC="${1:-32}"; DURATION="${2:-300}"; PORT="${PORT:-9100}"
IMG="${IMG:-kimi-lmc-mc-rocm:latest}"
IX="${IX:-/home/billhe12@amd.com/yiczhu/InferenceX}"
RUNDIR="${RUNDIR:-/home/thshan@amd.com/mc_lmc/run}"
DATASET="semianalysisai/cc-traces-weka-with-subagents-060826-256k"
RES="/run_logs/mc_pd_bench_c${CONC}"
NAME="kimi-mc-pd-bench-c${CONC}"
mkdir -p "$RUNDIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host \
  -v "$IX":/workspace -v /it-share/hf_cache:/models -v "$RUNDIR":/run_logs \
  --entrypoint bash "$IMG" -lc "
set -x
{
  export HF_HUB_CACHE=/models HF_HOME=/models HF_HUB_OFFLINE=0
  export INFMAX_CONTAINER_WORKSPACE=/workspace
  cd /workspace/benchmarks
  source ./benchmark_lib.sh
  export CONC=${CONC} DURATION=${DURATION} PORT=${PORT}
  export MODEL=Kimi-K2.5-MXFP4
  TOK=/models/models--amd--Kimi-K2.5-MXFP4/snapshots/419004c8716cf22c929aa15d39b85e09a8a2091a
  export TRACE_SOURCE_FLAG='--hf-dataset ${DATASET}'
  export RESULT_DIR=${RES} AGENTIC_OUTPUT_DIR=${RES}
  mkdir -p ${RES}
  install_agentic_deps
  build_replay_cmd ${RES}
  REPLAY_CMD=\"\$REPLAY_CMD --tokenizer \$TOK\"
  echo '=== REPLAY_CMD ==='; echo \"\$REPLAY_CMD\"
  eval \"\$REPLAY_CMD\"
  echo MC_PD_BENCH_DONE_rc=\$?
} > /run_logs/mc_pd_bench_c${CONC}.log 2>&1
"
echo "launched $NAME -> $RUNDIR/mc_pd_bench_c${CONC}.log (PORT=$PORT)"
sleep 5
docker ps --filter name=$NAME --format '{{.Names}} {{.Status}}'
