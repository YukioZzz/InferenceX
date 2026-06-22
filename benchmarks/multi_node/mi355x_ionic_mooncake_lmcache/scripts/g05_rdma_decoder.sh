#!/usr/bin/env bash
# Cross-node RDMA decoder on g05: :latest image + dmabuf mooncake .so bind-mounted,
# LMCacheConnectorV1 kv_both, protocol rdma, pointing at g06's shared master.
set -uo pipefail
IMG="${IMG:-kimi-lmc-mc-rocm:latest}"
DEPLOY="${DEPLOY:-/home/thshan@amd.com/mc_lmc}"
SO="${SO:-/home/thshan@amd.com/mc_lmc/dmabuf_so}"
RUNDIR="${RUNDIR:-/home/thshan@amd.com/mc_lmc/run_g05dec}"
MODEL="${MODEL:-/models/models--amd--Kimi-K2.5-MXFP4/snapshots/419004c8716cf22c929aa15d39b85e09a8a2091a}"
CFG="${CFG:-mooncake-rdma-decoder-g05.yaml}"
HOSTIP="${HOSTIP:-10.24.112.181}"
SERVE_PORT="${SERVE_PORT:-8000}"
CONC="${CONC:-128}"
EXTRA_ENV="${EXTRA_ENV:-}"
M=/usr/local/lib/python3.12/dist-packages/mooncake
mkdir -p "$RUNDIR"

docker rm -f kimi-mc-decoder >/dev/null 2>&1 || true
cat > "$RUNDIR/run_rdma_decoder.sh" <<EOF
#!/usr/bin/env bash
set -x
dpkg -i /ainic-repo/ionic-common*.deb /ainic-repo/libionic1*.deb 2>/dev/null || true
python3 - <<'PYAB' 2>&1 | tail -2 || true
import importlib.util as u
s=u.find_spec("lmcache.integration.vllm.vllm_v1_adapter"); f=s.origin; src=open(f).read()
if "[PATCHED] abort guard" in src: print("[patch] already")
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
    print("[patch] applied" if old in src else "[patch] WARN pattern miss"); 
    open(f,"w").write(src.replace(old,new,1)) if old in src else None
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
${EXTRA_ENV}
exec vllm serve ${MODEL} --served-model-name Kimi-K2.5-MXFP4 \\
  --host 0.0.0.0 --port ${SERVE_PORT} \\
  --tensor-parallel-size 8 --trust-remote-code \\
  --max-model-len 262144 --gpu-memory-utilization 0.90 --block-size 1 --mm-encoder-tp-mode data \\
  --kv-cache-dtype fp8 --max-num-seqs ${CONC} \\
  --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \\
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
EOF
chmod +x "$RUNDIR/run_rdma_decoder.sh"

docker run -d --name kimi-mc-decoder --hostname kimi-mc-decoder --init --stop-timeout 10 \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 \
  --network host --ipc host --group-add video \
  --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined --privileged \
  --shm-size 64G -v /sys:/sys -v /it-share/hf_cache:/models -v "$RUNDIR":/run_logs \
  -v /opt/amd/ainic/deb-repo:/ainic-repo:ro -v "$DEPLOY":/deploy \
  -v "$SO/engine.cpython-312-x86_64-linux-gnu.so:$M/engine.cpython-312-x86_64-linux-gnu.so:ro" \
  -v "$SO/store.cpython-312-x86_64-linux-gnu.so:$M/store.cpython-312-x86_64-linux-gnu.so:ro" \
  --entrypoint "" "$IMG" bash -lc "bash /run_logs/run_rdma_decoder.sh > /run_logs/mc_decoder.log 2>&1"
echo "launched kimi-mc-decoder (g05, :latest+dmabuf.so, rdma, master=g06)"; sleep 4
docker ps --filter name=kimi-mc-decoder --format '{{.Names}} {{.Status}}'
