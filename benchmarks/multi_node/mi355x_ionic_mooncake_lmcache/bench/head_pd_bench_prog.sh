#!/usr/bin/env bash
C="${1:-16}"
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 mia1-p01-g06 "bash -s $C" <<'EOF'
C="$1"
docker ps -a --filter name=kimi-mc-pd-bench-c$C --format '{{.Names}} {{.Status}}'
tail -n 40 /home/thshan@amd.com/mc_lmc/run_v1pd/mc_pd_bench_c${C}.log 2>/dev/null | tr '\r' '\n' | grep -aiE "Loading traces|dataset|warmup|Records|Completed:|Total Requests|TTFT|Throughput|error|Traceback|MC_PD_BENCH_DONE|fail|No such|500" | tail -14 | cut -c1-150
EOF
