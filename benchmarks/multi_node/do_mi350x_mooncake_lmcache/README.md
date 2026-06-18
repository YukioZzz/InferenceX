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
                          #   env: CONC (=> --max-num-seqs), EP=1 (=> --enable-expert-parallel, EP size = TP = 8)
bench_agentic.sh          # aiperf agentic trace-replay runner:  bench_agentic.sh <CONC> <DURATION_s>
eval_gsm8k.sh             # GSM8K correctness check through the proxy:  eval_gsm8k.sh <chat|comp> [LIMIT]
toy_proxy_server.py       # PD proxy (prefill->decode routing); honors the OpenAI `stream` flag
                          #   (JSON when stream=false for lm-eval, SSE when stream=true for aiperf)
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

Node IPs are hard-coded in `launch_pd.sh` (`P_IP`/`D_IP`, the `192.168.0.x` fabric);
prefill+proxy run on the P node, decode on the D node.

## How to run

```bash
# 1. on each node, build the ROCm image once (LMCache + Mooncake from source)
bash build/build_lmcache_rocm.sh ; bash build/build_mooncake_rocm.sh

# 2. start engines (CONC sets vLLM --max-num-seqs; EP=1 adds --enable-expert-parallel)
CONC=128 bash launch_pd.sh prefill     # P node
CONC=128 bash launch_pd.sh decode      # D node
bash launch_pd.sh proxy                # P node

# 3a. run the agentic trace replay (CONC, duration seconds; >=900 is a canonical point)
bash bench_agentic.sh 32 1800
bash tools/read_result.sh 32

# 3b. or run a GSM8K correctness check through the PD path (both endpoints)
bash eval_gsm8k.sh chat 50     # /v1/chat/completions (apply_chat_template) — the correct path for this model
bash eval_gsm8k.sh comp 50     # /v1/completions (plain)
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

## Throughput — concurrency sweep (aligned engine, 30-min/1800 s points, no-EP)

| conc | req tput | out tok/s | in tok/s | TTFT p50 | per-user out (p50) | reqs | theo hit |
|---|---|---|---|---|---|---|---|
| 8   | 0.24 rps | 188 | 22.5k | 2.1 s | 58.0 | 427 | 95.7 % |
| 16  | 0.42 rps | 349 | 39.1k | 1.9 s | 42.9 | 765 | 95.8 % |
| **32** | **0.49 rps** | **442 (peak)** | 44.2k | 2.7 s | 22.8 | 894 | 94.3 % |
| 64  | 0.49 rps | 374 | 42.1k | 14.6 s | 9.0 | 894 | 94.7 % |
| 128 | 0.38 rps | 384 | 38.2k | 200.9 s | 9.9 | 702 | 95.3 % |

**Saturation knee = conc32** (442 tok/s peak). Beyond it aggregate throughput plateaus
(374–384) while TTFT p50 explodes (2.7 s → 14.6 s → 201 s at conc128 = request-queue
collapse). Usable operating range is conc 16–32. Removing `--enforce-eager` + `fp8` KV +
`--max-num-seqs` gave ~4.6× output / ~2.7× TTFT vs the unaligned config. Higher concurrency
needs more prefill capacity (2P1D), not more load.

### Expert parallelism (EP) at high concurrency — net negative here

`EP=1` (`--enable-expert-parallel`, EP size = TP = 8) on **both** prefill and decode, vs no-EP:

| conc | out tok/s no-EP → EP | req tput no-EP → EP |
|---|---|---|
| 64  | 374 → **341** (-9 %) | 0.49 → 0.44 |
| 128 | 384 → **357** (-7 %) | 0.38 → 0.36 |

EP **slightly hurts** at conc64/128: the 1P1D is already prefill-compute-saturated at
conc32, so the per-step MoE token batch never gets large enough for EP's all-to-all
dispatch to beat TP's all-reduce — the extra dispatch overhead is pure loss. EP only pays
off with a much larger per-step batch (non-saturated prefill, i.e. 2P1D+ and higher batch).

## Correctness — GSM8K through the PD path (5-shot, 50 samples)

| endpoint | exact_match (flexible / strict) | verdict |
|---|---|---|
| `/v1/chat/completions` (apply_chat_template) | **0.98 / 0.98** | ✅ correct — matches the ~90 %+ expected for Kimi-K2.5-MXFP4 |
| `/v1/completions` (plain) | 0.02 / 0.00 | benign artifact, **not** a serving bug — see note |

The chat path scoring **98 %** confirms the Mooncake cross-node KV transfer + LMCache PD
produces **correct tokens** (unlike the MoRIIO+LMCache path elsewhere, which scored ~30 %
on chat from a connector KV-block bug — that bug does **not** reproduce here). The plain
`/v1/completions` near-zero is a stop-token/extraction artifact: Kimi is a reasoning model
and without a chat template it emits the correct answer (`#### 18`, `#### 3`, …) but then
keeps generating hallucinated follow-up Q&A (no matching stop token), so GSM8K's strict
extractor grabs the wrong number. The generated tokens are coherent and correct — chat is
the intended eval path for this model.

**Cache behavior**: Mooncake carries cross-node PD KV transfer with **0 errors**; LMCache
L2 holds a stable ~78 % lookup-token hit and absorbs more traffic as the GPU radix cache
evicts under contention (L2 read chunks 150 k→296 k from conc32→conc64).
