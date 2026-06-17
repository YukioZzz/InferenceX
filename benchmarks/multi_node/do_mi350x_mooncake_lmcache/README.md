# Kimi-K2.5 MXFP4 — 2-node vLLM PD disagg on DigitalOcean MI350X (Mooncake + LMCache)

Agentic trace-replay benchmark for **Kimi-K2.5-MXFP4** on a **2× DigitalOcean MI350X**
(SR-IOV **virtual-function**) cluster, using vLLM disaggregated prefill/decode with a
**MultiConnector = MooncakeConnector (cross-node KV transfer) + LMCacheMPConnector
(cross-request L2 host-DRAM cache)**.

This is the DigitalOcean-VF, Docker-based analogue of the single-node recipe
`benchmarks/single_node/agentic/kimik2.5_fp4_mi355x.sh` and the SLURM multi-node
recipe `benchmarks/multi_node/kimik2.5_fp4_mi355x_vllm-disagg.sh`. The engine and
aiperf parameters are deliberately kept **identical** to the single-node recipe so
results are comparable; only the topology and KV transport differ.

## Why this exists / hardware constraint

DigitalOcean MI350X instances are **SR-IOV virtual functions**, which do **not** expose
working **GPUDirect RDMA**. Direct GPU↔GPU RDMA KV transfer returns zeroed/corrupted
data. We evaluated several KV-transfer backends before settling on Mooncake host-staged
TCP + LMCache:

| Connector | Result on VF |
|---|---|
| MoRIIO (RDMA) | fails — GPUDirect RDMA unavailable on VF; `tcp` backend unimplemented |
| P2pNcclConnector | NCCL socket transfer hangs cross-node |
| NixlConnector (UCX) | host-buffer transfer produced degenerate output |
| **MooncakeConnector (tcp) + LMCacheMP** | **works** — host-staged TCP KV transfer, 0 errors |

Both LMCache and Mooncake are **built from source for ROCm/gfx950** (see `build/`),
because the default pip wheels pull CUDA libraries.

## Layout

```
launch_pd.sh              # primary launcher: <proxy|prefill|decode>  (Mooncake + LMCache MultiConnector)
bench_agentic.sh          # aiperf agentic trace-replay runner:  bench_agentic.sh <CONC> <DURATION_s>
toy_proxy_server.py       # PD proxy (prefill->decode routing); patched to stream text/event-stream (SSE)
connectors/               # alternative KV-transfer backends evaluated (reference only)
  mooncake_only.sh        #   MooncakeConnector without LMCache
  nixl.sh                 #   NixlConnector (UCX)         — degenerate output on VF
  p2p_nccl.sh             #   P2pNcclConnector            — hangs cross-node on VF
build/
  build_lmcache_rocm.sh   # build LMCache from source (CXX=hipcc BUILD_WITH_HIP=1)
  build_mooncake_rocm.sh  # build Mooncake transfer engine from source for ROCm
tools/
  read_result.sh          # dump aiperf console summary + LMCache/vLLM prefix-hit metrics
  verify_aligned.sh       # confirm effective engine flags (fp8 KV, no eager, max-num-seqs, KV-cache sizing)
  diag_stall.sh           # diagnose prefill/decode/proxy state under high-concurrency stall
  probe_vllm.sh           # check image vLLM version + kv_offload / connector module availability
```

Node IPs are hard-coded in `launch_pd.sh` (`P_IP=192.168.0.6`, `D_IP=192.168.0.7`);
prefill+proxy run on DO-6, decode on DO-7.

## How to run

```bash
# 1. on each node, build the ROCm image once (LMCache + Mooncake from source)
bash build/build_lmcache_rocm.sh ; bash build/build_mooncake_rocm.sh

# 2. start engines (CONC sets vLLM --max-num-seqs)
CONC=32 bash launch_pd.sh prefill     # DO-6
CONC=32 bash launch_pd.sh decode      # DO-7
bash launch_pd.sh proxy               # DO-6

# 3. run the agentic trace replay (CONC, duration seconds)
bash bench_agentic.sh 32 900

# 4. read results + cache hit rates
bash tools/read_result.sh 32
```

## Engine config — aligned to the single-node recipe

| Param | Value | Note |
|---|---|---|
| `--tensor-parallel-size` | 8 | |
| `--kv-cache-dtype` | **fp8** | halves KV → ~2× concurrency (10.77×→21.30× at 262 k ctx) |
| `--max-num-seqs` | `$CONC` | engine batches up to benchmark concurrency |
| `--block-size` | 1 | intentional for the MXFP4/MLA + connector path (matches SN recipe) |
| `--gpu-memory-utilization` | 0.90 | |
| `--mm-encoder-tp-mode` | data | |
| `--max-model-len` | 262144 | = Kimi native (SN recipe leaves unset → same) |
| HIP graphs | **on** (no `--enforce-eager`) | eager mode was the #1 cause of slow decode |
| prefix caching / hybrid mgr | enabled / disabled | |
| ROCm env | `VLLM_ROCM_USE_AITER=1`, `…_RMSNORM=1` (TP8), `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `PYTHONNOUSERSITE=1` | |

LMCache MP server: `--l1-size-gb 1200` (DO VF nodes have ~2 TB host DRAM, vs the SN
recipe's 3000 GB on 2.7 TB bare-metal), `--l1-read-ttl-seconds 7200`, `--chunk-size 256`,
`--max-workers 8`, `--eviction-policy LRU`.

The image ships **vLLM 0.20.2** (newer than the SN recipe's 0.18), which already includes
the upstream `v1/kv_offload` subsystem (demand-pinned host allocator, chunked GPU↔CPU KV
loading) and MLA `block_size=1` support that the recipe's flag combo relies on.

## Results (aligned engine, valid 900 s runs)

| Metric | conc32 | conc64 |
|---|---|---|
| Request throughput | **0.52 rps** | 0.48 rps |
| Output token throughput | **443 tok/s** | 375 tok/s |
| Input token throughput | 49,215 tok/s | 41,676 tok/s |
| TTFT (avg) | **5.0 s** | 17.8 s |
| Request latency (avg) | **43.4 s** | 104.0 s |
| Per-user output | **24.3 tok/s** | 10.4 tok/s |
| Requests / errors | 481 / 0 | 447 / 0 |
| GPU radix prefix hit | 89.5 % | 71.7 % |
| LMCache L2 lookup hit | 77.9 % | 78.1 % |
| Theoretical hit ceiling | 95.0 % | 94.9 % |

**Takeaways**
- Removing `--enforce-eager` + adding `--kv-cache-dtype fp8` + `--max-num-seqs $CONC`
  gave ~4.6× output throughput and ~2.7× faster TTFT at conc32 vs the unaligned config.
- The 1P1D topology is **throughput-saturated at conc32**; conc64 does not raise aggregate
  throughput (slightly lower) and only inflates latency — i.e. conc64 is past the
  saturation knee. Higher concurrency needs more prefill capacity (2P1D), not more load.
- Mooncake carries cross-node PD KV transfer with **0 errors**; LMCache L2 contributes a
  stable ~78 % lookup-token hit and absorbs more traffic as GPU-radix evicts under
  contention (read chunks 150 k→296 k from conc32→conc64).
