#!/usr/bin/env bash
# Drive agentic aiperf trace-replay against the Mooncake+LMCache PD proxy (port 10001) on DO-6.
# Usage: do_pd_bench.sh <CONC> [DURATION]
set -uo pipefail
CONC="${1:?usage: do_pd_bench.sh <CONC> [DURATION]}"
DURATION="${2:-300}"
IMG="kimi-lmc-mc-rocm:latest"
RUNDIR=/root/run_logs
RES="/run_logs/pd_mclmc_conc${CONC}"
NAME="kimi-pd-bench-c${CONC}"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" --network host \
  -v /root/InferenceX:/workspace \
  -v /root/models:/models \
  -v "$RUNDIR":/run_logs \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash "$IMG" -lc "
set -x
{
  cd /workspace/benchmarks
  source ./benchmark_lib.sh
  export INFMAX_CONTAINER_WORKSPACE=/workspace
  export AIPERF_DIR=/workspace/utils/aiperf
  export MODEL=/models/Kimi-K2.5-MXFP4
  export CONC=${CONC}
  export DURATION=${DURATION}
  export PORT=10001
  export MAX_MODEL_LEN=262144
  export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_060826_256k
  export RESULT_DIR=${RES}
  export AGENTIC_OUTPUT_DIR=${RES}
  mkdir -p ${RES}
  resolve_trace_source
  install_agentic_deps
  build_replay_cmd ${RES}
  echo '=== REPLAY_CMD ==='; echo \"\$REPLAY_CMD\"
  eval \"\$REPLAY_CMD\"
  echo PD_BENCH_DONE_rc=\$?
} > /run_logs/pd_bench_c${CONC}.log 2>&1
"
echo "launched $NAME -> /run_logs/pd_bench_c${CONC}.log"
sleep 5
docker ps --filter "name=$NAME" --format '{{.Names}} {{.Status}}'
tail -n 6 "$RUNDIR/pd_bench_c${CONC}.log" 2>/dev/null | tr '\r' '\n'
