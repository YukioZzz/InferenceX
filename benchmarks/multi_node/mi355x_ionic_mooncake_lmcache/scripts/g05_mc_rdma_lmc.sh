#!/usr/bin/env bash
# Minimal verify: Mooncake-RDMA store + LMCacheConnectorV1 (kv_both) on a single
# MI355 node (g05). Roles: master | serve.
#   g05: bash g05_mc_rdma_lmc.sh master
#   g05: bash g05_mc_rdma_lmc.sh serve
set -uo pipefail
ROLE="${1:?usage: <master|serve>}"
IMG="${IMG:-kimi-lmc-mc-rocm:latest}"
DEPLOY="${DEPLOY:-/home/thshan@amd.com/mc_lmc}"
RUNDIR="${RUNDIR:-/home/thshan@amd.com/mc_lmc/run}"
MODEL="${MODEL:-/models/models--amd--Kimi-K2.5-MXFP4/snapshots/419004c8716cf22c929aa15d39b85e09a8a2091a}"
CFG="${CFG:-mooncake-rdma-verify.yaml}"
HOSTIP="${HOSTIP:-10.24.112.181}"
SERVE_PORT=8000
CONC="${CONC:-32}"
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

if [ "$ROLE" = "master" ]; then
  docker rm -f kimi-mc-master >/dev/null 2>&1 || true
  docker run -d $(common_docker kimi-mc-master) --entrypoint bash "$IMG" -lc \
    "dpkg -i /ainic-repo/ionic-common*.deb /ainic-repo/libionic1*.deb 2>/dev/null || true; \
     mooncake_master --enable_http_metadata_server=1 \
       --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=8080 \
       --rpc_address=0.0.0.0 -v=1 > /run_logs/mc_master.log 2>&1"
  echo "launched mooncake master"; sleep 8
  docker exec kimi-mc-master tail -n 15 /run_logs/mc_master.log 2>/dev/null | tr '\r' '\n' | tail -12
  echo "== metadata server check =="
  curl -s -m5 -o /dev/null -w 'meta_http=%{http_code}\n' http://127.0.0.1:8080/metadata 2>&1 || true
  exit 0
fi

# ------------------------------- SERVE -------------------------------
cat > "$RUNDIR/run_mc_serve.sh" <<EOF
#!/usr/bin/env bash
set -x
dpkg -i /ainic-repo/ionic-common*.deb /ainic-repo/libionic1*.deb 2>/dev/null || true
echo "[run] ibv_devices:"; ibv_devices 2>/dev/null | head
# Patch: LMCache abort path asserts lmcache_engine!=None (None on scheduler side)
# -> aborted requests crash EngineCore. Make it guarded/non-fatal.
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
    new='''            # [PATCHED] abort guard: lmcache_engine/lookup_client are None on the
            # scheduler-side connector; do not assert (would kill EngineCore on abort).
            if self.lmcache_engine is not None:
                sm = self.lmcache_engine.storage_manager
                if sm is not None:
                    sm.cancel_request(request.request_id)
            if self.async_loading and getattr(self, "lookup_client", None) is not None:
                self.lookup_client.cancel_lookup(request.request_id)  # type: ignore[attr-defined]'''
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
export IBDEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export NCCL_IB_DISABLE=1
export UCX_IB_GID_INDEX=1
export MORI_RDMA_TC=104
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
chmod +x "$RUNDIR/run_mc_serve.sh"
docker rm -f kimi-mc-serve >/dev/null 2>&1 || true
docker run -d $(common_docker kimi-mc-serve) --entrypoint "" "$IMG" \
  bash -lc "bash /run_logs/run_mc_serve.sh > /run_logs/mc_serve.log 2>&1"
echo "launched kimi-mc-serve"; sleep 4
docker ps --filter name=kimi-mc-serve --format '{{.Names}} {{.Status}}'
