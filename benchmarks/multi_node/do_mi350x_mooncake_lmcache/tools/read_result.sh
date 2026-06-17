#!/usr/bin/env bash
set -uo pipefail
C="${1:?conc}"
R="/root/run_logs/pd_mclmc_conc${C}"
echo "== bench container =="; docker ps -a --filter name=kimi-pd-bench-c${C} --format '{{.Status}}'
echo "== aiperf console summary (key metrics) =="
sed -n '1,260p' "$R/aiperf_artifacts/profile_export_console.txt" 2>/dev/null | grep -aiE "Prefix Cache Hit|Request Count|Output Token Throughput|Input Token Throughput|Time to First Token|Request Latency|Input Sequence|Request Throughput" | cut -c1-115
echo "== LMCache MP server (prefill DO-6) =="
docker exec kimi-mclmc-prefill bash -lc 'curl -s -m8 http://127.0.0.1:8080/metrics 2>/dev/null | grep -iE "l1_write_chunks_total|l1_read_chunks_total|lookup_requested_tokens|lookup_hit_tokens" | grep -v "^#"' 2>/dev/null || echo "(prefill lmc n/a)"
echo "== vLLM external prefix hit (prefill) =="
docker exec kimi-mclmc-prefill bash -lc 'curl -s -m8 http://127.0.0.1:2584/metrics 2>/dev/null | grep -E "external_prefix_cache_(queries|hits)_total|prefix_cache_(queries|hits)_total" | grep -v created' 2>/dev/null || echo "(n/a)"
