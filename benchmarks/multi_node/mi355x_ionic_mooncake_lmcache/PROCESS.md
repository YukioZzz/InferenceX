# Cross-node 1P1D — Mooncake (RDMA/TCP) + LMCacheConnectorV1 on MI355X (ionic) — process record

Kimi-K2.5-MXFP4, 2 nodes (g06 prefiller + g05 decoder), TP8, agentic Weka trace replay.
This file is the **full debugging/process record**. The clean, leaderboard-ready
recipe is the sibling `mi355x_mooncake_tcp_lmcache/` directory.

> Supersedes the earlier `mc_lmc_deploy/INFERENCEX_RECIPE.md` conclusion that
> "cross-node Mooncake does not work on ionic" — it **does**, once the two real
> root causes below are fixed.

---

## TL;DR

Goal: a stable cross-node 1P1D where KV reuse flows through **LMCacheConnectorV1 +
mooncakestore** (the distributed store path, not MultiConnector/MooncakeConnector).

Two real root causes were blocking it on the MI355X **ionic** (AMD/Pensando AINIC) NICs,
plus one TCP-mode interaction:

1. **ionic host-memory MR registration ceiling.** `ibv_reg_mr` on **host** memory fails
   with `EINVAL[22]` for any MR **> ~64 MiB**, and the per-node total registrable host
   memory is **~3.8 GiB**. mooncake's L3 store is a 64 GiB host-DRAM segment registered
   for one-sided RDMA → it never fully registered → every cross-node Put/Get failed
   (`PutRevoke`, `Get=0`, `TRANSFER_FAIL`, `Corrupted segment descriptor`). External
   prefix-cache hit was ~0% the entire time → the sweep was effectively "no-cache PD".
   (GPU memory is unaffected — it registers via `ibv_reg_dmabuf_mr`, a different path.)

2. **HIP transport clobbers the published segment protocol to `"hip"`.** This is a
   `USE_HIP` build; the engine installs the rdma (or tcp) transport first (segment
   protocol `rdma`/`tcp`), then `HipTransport::install()` calls
   `addLocalSegment(LOCAL_SEGMENT_ID, …, protocol="hip")` and **overwrites** it. Remote
   peers then parse our host (`cpu:*`) segment under the `hip` protocol, which requires a
   `shm_name` the host buffer doesn't have → `Corrupted segment descriptor` → cross-node
   `open segment` fails. (`hip_transport.cpp:447`.)

3. **`MC_MAX_MR_SIZE` is enforced even for TCP.** mooncake's `CheckRegisterMemoryParams`
   rejects `length > max_mr_size` regardless of protocol ("Tcp is not limited by
   max_mr_size, but we ignore it for now"). So the 64 MiB cap we set for the RDMA fix also
   rejected the TCP store's large buffers → store didn't connect (`Clients: 0`).

Fixes (config/env + one 1-line `.so` patch; **no model change**):

| # | Fix | Where |
|---|---|---|
| 1 | `MC_MAX_MR_SIZE=64MiB` + `local_buffer_size=64MiB` + `global_segment_size ≤ ~3.8 GiB` (used 2 GiB) so the RDMA host segment registers in ionic-legal 64 MiB MRs | `rdma_engine.sh`, `mooncake-rdma-sa-*.yaml` |
| 2 | Patch `HipTransport::install()` to not overwrite the local segment descriptor; rebuild `.so` | `build/mc_patch_hipseg.sh` |
| 3 | For TCP: raise `MC_MAX_MR_SIZE` (e.g. 128 GiB) + `MC_FORCE_TCP=1`; then `global_segment_size` can be large (no NIC reg) | `head_v1pd_launch.sh` (EXTRA_ENV), TCP yamls |

Result (conc16, 5 min, same trace, **L1 GPU prefix-cache off** in all three):

| Config | External hit | TTFT avg | TTFT p95/max | Input tok/s | Output tok/s | reqs(5min) | errors |
|---|---|---|---|---|---|---|---|
| RDMA L3 = 2 GiB (host, capped) | 0–8 % | 26.9 s | 54.6 / 59.2 | 11,205 | 115 | 74 | 0 |
| **Big L2** = 320 GB host (L3 idle) | 50–63 % | 10.9 s | 31 / 33 | 19,003 | 198 | 123 | 0 |
| **TCP L3 = 64 GiB** (L2 off) | **63–70 %** | **7.0 s** | **16 / 20** | **24,896** | **298** | **158** | 0 |

Key takeaways:
- The hierarchy on this box is **inverted**: L1(GPU KV) ≈ 1.46 TB / 5.58 M tokens ≫
  L2(host) ≫ **L3(ionic-RDMA host) ≈ 2 GiB** — the cross-node tier is forced smallest by
  the ionic ceiling.
- Put the bulk reuse where capacity is cheap: **big L2 host CPU** (not RDMA-registered →
  no ionic limit) for same-prefiller agentic reuse, and a cross-node L3 for the PD handoff.
- **TCP L3 is the best practical cross-node store on ionic**: no registration limit → big
  pool, and the cache-hit win dwarfs TCP's per-byte bandwidth penalty (you ship cached KV
  instead of recomputing prefill).

---

## Architecture

```
proxy(:9100)  ──prompt(max_tokens=1)──▶  prefiller g06 (kv_both)
      │                                      │  writes KV → store
      └────────── decode ───────────▶  decoder  g05 (kv_both)  reads KV ← store
                                                 ▲
      Mooncake master (g06:50051) + HTTP metadata (g06:8080)  ◀── both engines connect

LMCacheConnectorV1 tiers (per engine):
  L1 = vLLM GPU prefix cache  (HBM; --enable-prefix-caching; OFF in these process runs)
  L2 = LMCache local_cpu       (host DRAM; node-local; NOT NIC-registered)
  L3 = mooncakestore           (remote shared pool; RDMA host = capped / TCP host = big)
```

## Root-cause detail & evidence

### Bug 1 — ionic host MR ceiling (the main blocker)
Decoder log at store mount:
```
rdma_context.cpp:171] RDMA context setup failed: fork compatibility: Invalid argument[22]   # benign (ibv_fork_init only logs)
rdma_context.cpp:536] Failed to register memory 0x..: Invalid argument[22]                   # ibv_reg_mr(host) fails
client_service.cpp:488] length 5368709120 is larger than max_mr_size: 2147483648
real_client.cpp:877]   Failed to mount segment: INVALID_PARAMS
```
Unit test (`tools/mc_xnode_client_test.py`, store client only, no model) pinned the ceiling:

| Case | MC_MAX_MR_SIZE | seg | local_buf | result |
|---|---|---|---|---|
| A | default(2 GiB) | 64 GiB | 256 MiB | `register_memory size=2 GiB` **EINVAL** |
| B | 256 MiB | 64 GiB | 256 MiB | `size=256 MiB` **EINVAL** |
| C | 256 MiB + relaxed-ordering OFF | 64 GiB | 256 MiB | **EINVAL** (so relaxed-ordering ≠ cause) |
| F | 128 MiB | 64 GiB | 128 MiB | `size=128 MiB` **EINVAL** |
| **G** | **64 MiB** | 64 GiB | **64 MiB** | **registered ~3.8 GiB then EINVAL** (61×64 MiB) → ceiling ~3.8 GiB/node |
| clean-node 2 GiB seg | 64 MiB | 2 GiB | 64 MiB | **full mount + XNODE_OK** |
| clean-node 3 GiB seg | 64 MiB | 3 GiB | 64 MiB | **full mount + XNODE_OK** |

So: single host MR must be **≤ 64 MiB**; per-node total host registration **≈ 3.8 GiB**.
GPU dmabuf (`ibv_reg_dmabuf_mr`) is unaffected (proven separately; that's why MI350/DO
worked and MI355X/ionic didn't).

> Note: earlier "256 MiB works" / "XNODE_OK" were **false positives** — the UT only
> `put` ~1 KB, which fits the local buffer and never exercised the large segment.

### Bug 2 — HIP segment-protocol clobber
After Bug 1 fixed (2 GiB RDMA segment mounts), Put committed (`PutEnd>0`, Keys grew) but
cross-node read still failed:
```
transfer_metadata.cpp:770] Corrupted segment descriptor, name <peer> protocol hip ... cpu:0 ...
transfer_task.cpp:1178]   Failed to open segment for endpoint='<peer>'
```
`hip_transport.cpp:447 HipTransport::install()` sets `desc->protocol="hip"` and
`addLocalSegment(LOCAL_SEGMENT_ID,…)`, overwriting the rdma/tcp segment (HIP is installed
after rdma/tcp). Patch (`build/mc_patch_hipseg.sh`) early-returns before that
`addLocalSegment`, leaving the rdma/tcp segment intact. After rebuild+redeploy:
`Corrupted=0`, `open-fail=0`, external hit `0% → 7–9%` (RDMA 2 GiB), proving the
cross-node store path is functional.

### Bug 3 — TCP path also gated by MC_MAX_MR_SIZE
Switching to `protocol:tcp` + `MC_FORCE_TCP=1` with a 64 GiB segment first failed with
`Failed to register local memory: INVALID_PARAMS` / `-600` and `Clients:0`, because the
64 MiB `MC_MAX_MR_SIZE` (set for the RDMA fix) is still applied by
`CheckRegisterMemoryParams` even for TCP. Raising `MC_MAX_MR_SIZE` to 128 GiB → 64 GiB
segment mounts fully, `Clients:2`.

## Tier sizes measured on this box (per node, TP8)
- **L1** GPU KV: `Available KV cache memory: 182.58 GiB/GPU` → ~1.46 TB, **5,579,599 tokens**
  (21× a full 262 k context). Currently unused for reuse (`--no-enable-prefix-caching`).
- **L2** host CPU: `max_local_cpu_size` GB **per worker** (×8). 40 → ~320 GB.
- **L3** mooncake pool: `global_segment_size` per engine (1 store client/engine, `Clients:2`).
  RDMA host → ≤ ~3.8 GiB; TCP host → host-DRAM bound (node has ~3 TB).

## Other fixes carried over
- `mooncake_master_server_addr` config key (LMCache forwards `mooncake_`-prefixed keys
  with priority; plain `master_server_address` was ignored → defaulted to 127.0.0.1 →
  broke cross-node). Both keys set in the yamls.
- abort-guard patch to `lmcache/.../vllm_v1_adapter.py:request_finished` (upstream asserts
  `lmcache_engine is not None` on the scheduler-side connector for aborted requests → kills
  EngineCore on the bench's end-of-run cancel). Applied at container start in the launchers.
- `kv_load_failure_policy:"recompute"` so partial loads fall back instead of 500.
- `PYTHONHASHSEED=0` (cross-process pre-caching hash consistency).

## Files
```
build/   mc_patch_hipseg.sh        # Bug-2 fix: HipTransport::install no-clobber + rebuild .so
         mc_patch_hiptransport.sh  # earlier host-skip patch (superseded by hipseg, kept for history)
         mc_rebuild_dmabuf.sh      # USE_HIP_DMABUF rebuild (GPU dmabuf RDMA enablement)
configs/ mooncake-rdma-sa-pref.yaml / -dec.yaml   # final TCP variant (protocol tcp, big L3)
         mooncake-tcp-agentic.yaml / -decoder.yaml # single-node + early cross-node TCP
         mooncake-rdma-verify.yaml                 # single-node RDMA verify
scripts/ rdma_engine.sh            # generic 1P1D engine launcher (MC_MAX_MR_SIZE, MC_FORCE_TCP via EXTRA_ENV)
         g05_mc_rdma_lmc.sh        # master + single-node serve
         g06_mc_decoder.sh / g05_rdma_decoder.sh   # decoder variants
         head_v1pd_launch.sh       # orchestrator: master+prefiller(g06) + decoder(g05)
proxy/   mc_pd_proxy.py            # minimal 1P1D proxy (prefill max_tokens=1 → decode)
         g05_pd_proxy.sh
bench/   head_pd_bench_run.sh / head_pd_bench_prog.sh / g05_pd_bench.sh   # agentic trace replay
         do_sn_launch.sh           # SA single-node agent recipe reference (L1 on, CPU cache 1200 GB)
tools/   mc_xnode_client_test.py   # store-client unit test used to pin the ionic MR ceiling
```

## Reproduce (RDMA path)
```bash
# 1) rebuild .so with the hip-clobber fix (and USE_HIP_DMABUF already in :dmabuf image)
ssh g06 'docker run --rm -v /home/.../mc_lmc:/deploy kimi-lmc-mc-rocm:dmabuf bash /deploy/mc_patch_hipseg.sh'
# 2) launch master+prefiller(g06) + decoder(g05); engines mount patched .so (MOUNT_SO=1)
bash scripts/head_v1pd_launch.sh
# 3) proxy + agentic sweep
bash proxy/g05_pd_proxy.sh ; bash bench/head_pd_bench_run.sh 16 300
```
For TCP: set `protocol:tcp` in the yamls, `EXTRA_ENV='export MC_FORCE_TCP=1; export MC_MAX_MR_SIZE=137438953472'`, bump `global_segment_size`.
