#!/usr/bin/env bash
# Run agentic trace-replay against the 1P1D proxy (:9100) on g06. args: CONC DUR
CONC="${1:?conc}"; DUR="${2:-1800}"
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 mia1-p01-g06 "bash -s $CONC $DUR" <<'EOF'
CONC="$1"; DUR="$2"
curl -s -m6 -o /dev/null -w 'proxy=%{http_code}\n' http://127.0.0.1:9100/health 2>&1 || true
RUNDIR=/home/thshan@amd.com/mc_lmc/run_v1pd IX=/home/billhe12@amd.com/yiczhu/InferenceX IMG=kimi-lmc-mc-rocm:dmabuf PORT=9100 \
  bash /home/thshan@amd.com/mc_lmc/g05_pd_bench.sh "$CONC" "$DUR"
EOF
