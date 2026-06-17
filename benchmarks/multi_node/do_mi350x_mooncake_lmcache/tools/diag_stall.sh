#!/usr/bin/env bash
set -uo pipefail
echo "==== PREFILL (DO-6) tail ===="
tail -n 6 /root/run_logs/mclmc_prefill.log 2>/dev/null | tr '\r' '\n' | grep -aE 'Running|Waiting|Engine|EngineCore|Error|error|Timeout|timeout|Aborted|finished|GPU KV' | tail -6 | cut -c1-160
echo "==== PROXY tail ===="
tail -n 8 /root/run_logs/mclmc_proxy.log 2>/dev/null | cut -c1-160
echo "==== DECODE (DO-7) tail via ssh ===="
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no do7 "tail -n 6 /root/run_logs/mclmc_decode.log 2>/dev/null | tr '\r' '\n' | grep -aE 'Running|Waiting|Engine|Error|error|Timeout|Aborted|finished|recv|KV' | tail -6 | cut -c1-160" 2>&1
echo "==== prefill GPU util ===="
rocm-smi --showuse 2>/dev/null | grep -E 'GPU\[|use' | head -8 || true
