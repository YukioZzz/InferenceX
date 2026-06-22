# Kimi-K2.5-MXFP4 — 1P1D Mooncake-TCP + LMCacheConnectorV1 (MI355X / ionic)

Agentic-leaderboard recipe: cross-node 1P1D (prefiller g06 + decoder g05, TP8) with a single
`LMCacheConnectorV1` doing KV reuse across **all three tiers**, and a **Mooncake-TCP** L3
distributed store for the cross-node KV handoff.

```
L1 = vLLM GPU prefix cache   (HBM; --enable-prefix-caching)            ~1.46 TB / 5.58M tok on this box
L2 = LMCache local_cpu       (host DRAM; node-local agentic reuse)     ~1200 GB/node (SA recipe)
L3 = mooncakestore over TCP  (cross-node shared pool; host-DRAM)       256 GiB/engine (tunable)
```

## Why TCP for L3 (not RDMA) on ionic
The MI355X **ionic** NIC caps **host** memory RDMA registration (`ibv_reg_mr`): single MR
**≤ 64 MiB**, **~3.8 GiB total/node**. mooncake's RDMA L3 is a host-DRAM segment registered for
one-sided RDMA, so on ionic it can only be ~2 GiB — far too small for 256k-token agentic
contexts. **TCP transport does not register MRs**, so the host L3 pool can be large (hundreds of
GB), and the cache-hit win dwarfs TCP's per-byte bandwidth cost. (GPU dmabuf RDMA is unaffected,
but HBM is scarce.) Full root-cause analysis (ionic host-MR ceiling, the HipTransport
segment-protocol clobber, and the RDMA/TCP/big-L2 comparison) is kept on the process-record
branch `yichaozhu/mi355x-mooncake-lmcache-process` (`benchmarks/multi_node/mi355x_ionic_mooncake_lmcache/PROCESS.md`).

## Tier / config knobs
| Knob | Value | Why |
|---|---|---|
| `--enable-prefix-caching` | on (L1) | free GPU-resident reuse (HBM holds ~5.58M tok here) |
| `local_cpu` / `max_local_cpu_size` | True / **150 GB per worker** (~1200 GB/node) | L2 host CPU reuse, per SA single-node agent recipe (`l1-size-gb 1200`) |
| `protocol` | `tcp` | ionic host RDMA reg capped; TCP host L3 has no NIC reg limit |
| `global_segment_size` | 256 GiB/engine | L3 pool size (host-DRAM bound; node ~3 TB) |
| `chunk_size` | 256 | KV chunk / hash granularity (LMCache default; matches SA) |
| `save_chunk_meta` | True | keeps LMCache's big CPU cache off the NIC registration path |
| env `MC_FORCE_TCP=1` | — | force TCP transport (HCAs present would default to rdma) |
| env `MC_MAX_MR_SIZE=128 GiB` | — | `CheckRegisterMemoryParams` applies even for TCP; must exceed buffers |
| env `PYTHONHASHSEED=0` | — | cross-process pre-caching hash consistency (else all-miss) |

## Required one-time build — patched mooncake `.so`
The stock `USE_HIP` build has `HipTransport::install()` overwrite the node's segment descriptor
with `protocol="hip"`, which makes remote peers fail to open our host (`cpu:*`) segment
(`Corrupted segment descriptor`) — breaking cross-node read on **both** TCP and RDMA. Patch + rebuild:
```bash
docker run --rm -v <deploy>:/deploy kimi-lmc-mc-rocm:dmabuf bash /deploy/patch_mooncake_hipseg.sh
# -> writes patched engine.so + store.so into <deploy>/dmabuf_so (bind-mounted by the engines)
```
(`build/patch_mooncake_hipseg.sh`)

## Run
```bash
# 0) (once) build the patched .so (above), and copy configs/proxy into <deploy>
# 1) launch master + prefiller(g06) + decoder(g05) + proxy(:9100)
bash scripts/launch_pd.sh
# 2) agentic trace replay against the proxy
bash bench/bench_agentic.sh 16 1800        # conc16, 30 min
```

## Validated behaviour (conc16 / 5 min; L2-isolated TCP L3 = 64 GiB; L1 off)
| Metric | RDMA 2 GiB L3 (capped) | **TCP 64 GiB L3** |
|---|---|---|
| External (L3) hit | 0–8 % | **63–70 %** |
| TTFT avg / p95 | 26.9 / 54.6 s | **7.0 / 16 s** |
| Input throughput | 11,205 tok/s | **24,896 tok/s** |
| corruption / open-fail | 0 / 0 | 0 / 0 |

This leaderboard config additionally turns on **L1** (GPU APC) and **L2** (1200 GB host CPU) and
enlarges **L3** to 256 GiB, i.e. all three reuse tiers stacked. gsm8k correctness on the
LMCacheConnectorV1 path was 0.96–0.98 exact-match (vs 0.30 for the MultiConnector path).

> The all-tiers-on (L1+L2+256 GiB-L3) combination is assembled from individually validated parts;
> re-run `bench/bench_agentic.sh` (30 min) to record the stacked-tier numbers before publishing.

## Notes
- `kv_role:"kv_both"`, `kv_load_failure_policy:"recompute"` (partial loads fall back, no 500s).
- Single store client per engine (`Clients: 2` at the master); `global_segment_size` is per engine.
- abort-guard patch to `vllm_v1_adapter.py:request_finished` applied at container start (keeps
  EngineCore alive through the bench's end-of-run cancel).
