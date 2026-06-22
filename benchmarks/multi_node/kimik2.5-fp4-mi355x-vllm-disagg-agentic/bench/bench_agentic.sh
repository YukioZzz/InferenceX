#!/usr/bin/env bash
# Agentic trace-replay against the 1P1D proxy (:9100). Usage: bench_agentic.sh <CONC> [DURATION_S]
# Run on the prefiller node (g06); replays the SemiAnalysis Weka agentic trace.
set -uo pipefail
CONC="${1:-16}"; DURATION="${2:-1800}"; PORT="${PORT:-9100}"
IMG="${IMG:-kimi-lmc-mc-rocm:dmabuf}"
IX="${IX:-/home/billhe12@amd.com/yiczhu/InferenceX}"        # InferenceX checkout (has benchmarks/)
RUNDIR="${RUNDIR:-/home/thshan@amd.com/mc_lmc/run_pd}"
DATASET="${DATASET:-semianalysisai/cc-traces-weka-with-subagents-060826-256k}"
TOK="${TOK:-/models/models--amd--Kimi-K2.5-MXFP4/snapshots/419004c8716cf22c929aa15d39b85e09a8a2091a}"
RES="/run_logs/agentic_c${CONC}"; NAME="kimi-agentic-c${CONC}"
mkdir -p "$RUNDIR"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host \
  -v "$IX":/workspace -v /it-share/hf_cache:/models -v "$RUNDIR":/run_logs \
  --entrypoint bash "$IMG" -lc "
set -x
{
  export HF_HUB_CACHE=/models HF_HOME=/models HF_HUB_OFFLINE=0 INFMAX_CONTAINER_WORKSPACE=/workspace
  cd /workspace/benchmarks; source ./benchmark_lib.sh
  export CONC=${CONC} DURATION=${DURATION} PORT=${PORT} MODEL=Kimi-K2.5-MXFP4
  export TRACE_SOURCE_FLAG='--hf-dataset ${DATASET}'
  export RESULT_DIR=${RES} AGENTIC_OUTPUT_DIR=${RES}; mkdir -p ${RES}
  install_agentic_deps
  build_replay_cmd ${RES}
  REPLAY_CMD=\"\$REPLAY_CMD --tokenizer ${TOK}\"
  echo '=== REPLAY_CMD ==='; echo \"\$REPLAY_CMD\"; eval \"\$REPLAY_CMD\"
  echo BENCH_DONE_rc=\$?
} > /run_logs/agentic_c${CONC}.log 2>&1
"
echo "launched $NAME -> $RUNDIR/agentic_c${CONC}.log (proxy :$PORT, conc=$CONC, dur=${DURATION}s)"
