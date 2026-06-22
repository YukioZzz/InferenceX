#!/usr/bin/env bash
# 1P1D decoder on g06: LMCacheConnectorV1 (kv_both) pointing at the SHARED Mooncake
# master on g05. Retrieves prompt-prefix KV the g05 prefiller stored into the pool.
# Run ON g06.
set -uo pipefail
IMG="${IMG:-kimi-lmc-mc-rocm:latest}"
DEPLOY="${DEPLOY:-/home/thshan@amd.com/mc_lmc}"
RUNDIR="${RUNDIR:-/home/thshan@amd.com/mc_lmc/run}"
MODEL="${MODEL:-/models/models--amd--Kimi-K2.5-MXFP4/snapshots/419004c8716cf22c929aa15d39b85e09a8a2091a}"
CFG="${CFG:-mooncake-tcp-decoder.yaml}"
HOSTIP="${HOSTIP:-10.24.112.182}"
SERVE_PORT="${SERVE_PORT:-8000}"
CONC="${CONC:-128}"
mkdir -p "$RUNDIR"

common_docker() {
  echo --name "$1" --hostname "$1" --init --stop-timeout 10 \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined --privileged \
    --shm-size 64G -v /sys:/sys -v /it-share/hf_cache:/models -v "$RUNDIR":/run_logs \
    -v /opt/amd/ainic/deb-repo:/ainic-repo:ro -v "$DEPLOY":/deploy
}

cat > "$RUNDIR/run_mc_decoder.sh" <<EOF
#!/usr/bin/env bash
set -x
dpkg -i /ainic-repo/ionic-common*.deb /ainic-repo/libionic1*.deb 2>/dev/null || true
# abort-guard patch (same as serve): LMCache asserts lmcache_engine!=None on the
# scheduler-side connector for ABORTED requests -> kills EngineCore on bench cancel.
python3 - <<'PYAB' 2>&1 | tail -2 || true
import importlib.util as u
s=u.find_spec("lmcache.integration.vllm.vllm_v1_adapter")
f=s.origin; src=open(f).read()
if "[PATCHED] abort guard" in src:
    print("[patch] abort guard already applied")
else:
    old='''            assert self.lmcache_engine is not None
            sm = self.lmcache_engine.storage_manager
            if sm is not None:
                sm.cancel_request(request.request_id)

            if self.async_loading:
                # Cancel any ongoing async lookup and prefetch tasks on workers
                lookup_id = request.request_id
                assert self.lookup_client is not None
                self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[attr-defined]'''
    new='''            if self.lmcache_engine is not None:
                sm = self.lmcache_engine.storage_manager
                if sm is not None:
                    sm.cancel_request(request.request_id)
            if self.async_loading and getattr(self, "lookup_client", None) is not None:
                self.lookup_client.cancel_lookup(request.request_id)  # type: ignore[attr-defined]
            # [PATCHED] abort guard'''
    if old in src:
        open(f,"w").write(src.replace(old,new,1)); print("[patch] applied abort guard")
    else:
        print("[patch] WARN: abort pattern not found")
PYAB
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE=/deploy/${CFG}
export LMCACHE_USE_EXPERIMENTAL=True
export VLLM_USE_V1=1
export VLLM_HOST_IP=${HOSTIP}
export NCCL_IB_DISABLE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
export VLLM_ROCM_USE_AITER_RMSNORM=1
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
exec vllm serve ${MODEL} \\
  --served-model-name Kimi-K2.5-MXFP4 \\
  --host 0.0.0.0 --port ${SERVE_PORT} \\
  --tensor-parallel-size 8 --trust-remote-code \\
  --max-model-len 262144 --gpu-memory-utilization 0.90 --block-size 1 --mm-encoder-tp-mode data \\
  --kv-cache-dtype fp8 --max-num-seqs ${CONC} \\
  --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \\
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
EOF
chmod +x "$RUNDIR/run_mc_decoder.sh"
docker rm -f kimi-mc-decoder >/dev/null 2>&1 || true
docker run -d $(common_docker kimi-mc-decoder) --entrypoint "" "$IMG" \
  bash -lc "bash /run_logs/run_mc_decoder.sh > /run_logs/mc_decoder.log 2>&1"
echo "launched kimi-mc-decoder"; sleep 4
docker ps --filter name=kimi-mc-decoder --format '{{.Names}} {{.Status}}'
