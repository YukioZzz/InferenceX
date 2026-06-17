#!/usr/bin/env bash
set -uo pipefail
echo "== prefill health =="
curl -sf -m5 http://127.0.0.1:2584/health >/dev/null 2>&1 && echo "prefill READY" || echo "prefill not-ready-yet"
echo "== effective flags (vllm_command / startup) =="
grep -aE "enforce_eager|kv_cache_dtype|max_num_seqs|block_size|cuda_graph|enforce eager" /root/run_logs/mclmc_prefill.log 2>/dev/null | tr '\r' '\n' | grep -aivE "throughput" | head -6 | cut -c1-150
echo "== KV cache sizing (fp8 should ~2x tokens vs before) =="
grep -aE "GPU KV cache size|Maximum concurrency|kv cache dtype|Using kv-cache-dtype" /root/run_logs/mclmc_prefill.log 2>/dev/null | tr '\r' '\n' | grep -aivE "throughput" | tail -4 | cut -c1-170
echo "== LMCache server pool =="
grep -aE "l1.?size|read.?ttl|L1|GB" /root/run_logs/lmcache_prefill.log 2>/dev/null | tr '\r' '\n' | head -4 | cut -c1-150
