# In-tree sglang overlays for DSv4 FP8 1P1D MoRI PD-disagg (MI355X)

This directory carries the full set of sglang source overlays that make
DeepSeek-V4 (DSv4) FP8 run under 1-prefill / 1-decode disaggregation on
the AMD MoRI RDMA backend. They are bind-mounted over the upstream
sglang files inside the docker container at runtime — there is **no
runtime monkey-patching**; every file here is the real, patched module
source.

The overlay tree mirrors the in-container sglang package layout, so the
bind-mount mapping is 1:1:

```
patches/dsv4/srt/<rel>  ->  /sgl-workspace/sglang/python/sglang/srt/<rel>
```

## Image

Forked from the files shipped in
`rocm/sgl-dev:rocm720-mi35x-a8410de-20260502-DSv4`. The edits are pinned
to that image's sglang tree; re-derive them (diff against a fresh
container) before reusing on a different image.

## The headline fix — `srt/disaggregation/mori/conn.py`

`Engine batch write error 11 message length out of range` fired
8000+ times per run, at any concurrency > 1, and dropped most requests.

**Root cause.** DSv4 registers two per-layer KV MR sizes — full-attention
layers register full-pool-capacity MRs (~28 MB), SWA-attention layers
register the smaller SWA-pool MRs (~1.3 / 6.3 MB). Upstream
`MoriKVManager._issue_layer_transfers` passed full-pool slot ids straight
to MoRI as `slot * kv_item_len`. Once the prefill allocator handed out
slot ids past the SWA pool capacity (slot 10000+ observed at conc=4),
every WR into a SWA-layer MR exceeded `MR.length` and tripped MoRI's
pre-flight bounds check (`common.cpp:334`).

**Fix.** In `_issue_layer_transfers`, classify each layer at runtime by
comparing `src_desc.size` against the largest layer-MR size cached on
the first call. SWA-sized layers wrap slot ids on **both** sides with
`(slot % min(src_cap, dst_cap)) * kv_item_len` and emit singleton WRs;
full-attention layers keep the upstream fast path unchanged. Writer
(prefill) and reader (decode) use the same wrap modulus, so any aliasing
applies identically and intra-request KV stays consistent. A permanent
pre-flight bounds assertion (`DSV4_LAYER_PREFLIGHT_STRICT`, default 1)
mirrors MoRI's C++ check just before `batch_write`, so any future
regression aborts loudly in Python instead of flooding MoRI `[io][error]`.

The same file also carries the sparse, page-indexed **state-buffer
transfer** (`_transfer_state_buffers` + `TransferInfo.dst_state_indices`
plumbing): DSv4's compress / indexer-compress state pools are copied
per request using SWA-page indices, with optional per-WR chunking
(`DSV4_STATE_CHUNK_CAP_BYTES`, default 512 KiB) to stay under MoRI's
per-WR cap, and single-call aggregation (`DSV4_STATE_AGGREGATE_BATCH`).

## Supporting overlays

| Overlay | Edit | Why |
|---|---|---|
| `srt/disaggregation/utils.py` | `is_mla_backend` also accepts `DeepSeekV4TokenToKVPool` | disagg KV registration recognizes the DSv4 pool |
| `srt/model_executor/forward_batch_info.py` | `ForwardMode.is_prefill(include_draft_extend_v2=False)` forwards the kwarg | DSv4 radix backend calls `is_prefill(include_draft_extend_v2=True)`; upstream signature TypeErrors |
| `srt/layers/attention/nsa/index_buf_accessor.py` | relax `loc.dtype` assertion to `int32`/`int64` | disagg prefill passes int32 alloc indices; author already noted "can be int32" |
| `srt/layers/attention/deepseek_v4_backend_radix.py` | `_create_flashmla_metadata` is import-tolerant (returns `None` if `flash_mla` absent) | the ROCm DSv4 image ships no `flash_mla` (CUDA-only); HIP routes to tilelang and ignores the metadata |
| `srt/mem_cache/deepseekv4_memory_pool.py` | seed an **identity** `full_to_swa_index_mapping` at the end of `_init_paged_compress_states` | DSv4 uses `PagedTokenToKVPoolAllocator` (no separate SWA allocator), so the mapping is otherwise never populated; identity is correct because `swa_kv_pool` is addressed by the same slot id as the full pool |
| `srt/disaggregation/prefill.py` | `state_type = "dsv4"` for the DSv4 pool; compute SWA-page `state_indices` via `translate_loc_from_full_to_swa` (allocator-exposed pool) | tells the transfer backend state buffers exist and supplies the page indices |
| `srt/disaggregation/decode.py` | same `state_type` + `state_indices` (pool exposed directly on decode) | receiver side of the state-buffer transfer |

## How to enable

The overlay tree must be present on **every** node that runs a
prefill/decode container, then bind-mounted over the sglang package.

The direct-SSH launcher `local_scripts/run_dsv4_pd_disagg.sh` does this
automatically — point `DSV4_OVERLAY_DIR` at this directory (or a synced
copy) on the GPU nodes and it builds the eight `-v` mounts. To wire it
into another driver (e.g. the `EXTRA_DOCKER_MOUNTS` convention used by
`../README.md`):

```bash
SGL=/sgl-workspace/sglang/python/sglang
OV=$DI_REPO_DIR/benchmarks/multi_node/amd_utils/patches/dsv4
for rel in \
  srt/disaggregation/utils.py \
  srt/disaggregation/prefill.py \
  srt/disaggregation/decode.py \
  srt/disaggregation/mori/conn.py \
  srt/model_executor/forward_batch_info.py \
  srt/mem_cache/deepseekv4_memory_pool.py \
  srt/layers/attention/deepseek_v4_backend_radix.py \
  srt/layers/attention/nsa/index_buf_accessor.py ; do
    EXTRA_DOCKER_MOUNTS+=" -v $OV/$rel:$SGL/$rel:ro"
done
export EXTRA_DOCKER_MOUNTS
```

When the overlays are not mounted, container behavior is byte-identical
to the unpatched image.

## Tunable knobs (read by the overlays)

| Env var | Default | Effect |
|---|---|---|
| `DSV4_LAYER_SWA_WRAP` | `1` | enable the SWA-layer slot-id wrap (the fix) |
| `DSV4_LAYER_PREFLIGHT_STRICT` | `1` | raise on a pre-flight bounds violation (0 = warn only) |
| `DSV4_STATE_CHUNK_CAP_BYTES` | `524288` | split state WRs larger than this into sub-WRs |
| `DSV4_STATE_AGGREGATE_BATCH` | `1` | issue one `batch_write` per request instead of per segment |
| `DSV4_STATE_MAX_ITEM_LEN` | `0` | if > 0, skip state segments whose per-page size exceeds it (A/B only) |
| `DSV4_STATE_SERIALIZE` | `0` | serialize state `batch_write` across MoRI threads (diagnostic) |
| `DSV4_{LAYER,STATE}_DIAG_*` | varies | per-call / per-fail diagnostic logging (failures only by default) |

## Verification

1P1D on `GPU3D78` (prefill) + `GPU74C0` (decode), random ISL=OSL=1024,
range_ratio 0.8:

| conc | completed | mean TPOT | output tok/s | MoRI len-OOR errors |
|---|---|---|---|---|
| 1/10 | 10/10 | 203 ms | 4.91 | 0 |
| 4/40 | 40/40 | 207 ms | 18.13 | 0 |
| 10/40 | 40/40 | 206 ms | 45.33 | 0 |

Completion smoke (T=0, 4 prompts) returned coherent on-topic text,
confirming end-to-end KV consistency.

## Provenance / stop-gap

This is a stop-gap pinned to the DSv4 dev image. The proper home for the
`_issue_layer_transfers` SWA-wrap and the DSv4 state-buffer transfer is
upstream sglang. When a published image carries the fix, retire the
corresponding overlay; unmounted files always fall back to the image's
own source.
