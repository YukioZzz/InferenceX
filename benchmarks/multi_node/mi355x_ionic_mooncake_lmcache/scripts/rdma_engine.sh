#!/usr/bin/env bash
# Generic 1P1D RDMA engine launcher (Mooncake RDMA + LMCacheConnectorV1).
# Usage: rdma_engine.sh <prefiller|decoder>
#   env: IMG, MOUNT_SO(0/1), CFG, HOSTIP, CONC, RUNDIR, SERVE_PORT
set -uo pipefail
ROLE="${1:?prefiller|decoder}"
IMG="${IMG:-kimi-lmc-mc-rocm:latest}"
MOUNT_SO="${MOUNT_SO:-1}"
DEPLOY="${DEPLOY:-/home/thshan@amd.com/mc_lmc}"
SO="${SO:-/home/thshan@amd.com/mc_lmc/dmabuf_so}"
MODEL="${MODEL:-/models/models--amd--Kimi-K2.5-MXFP4/snapshots/419004c8716cf22c929aa15d39b85e09a8a2091a}"
HOSTIP="${HOSTIP:-10.24.112.181}"
SERVE_PORT="${SERVE_PORT:-8000}"
CONC="${CONC:-128}"
M=/usr/local/lib/python3.12/dist-packages/mooncake
case "$ROLE" in
  prefiller) CN=kimi-mc-serve;   CFG="${CFG:-mooncake-rdma-sa-pref.yaml}"; RUNDIR="${RUNDIR:-/home/thshan@amd.com/mc_lmc/run_g06}";;
  decoder)   CN=kimi-mc-decoder; CFG="${CFG:-mooncake-rdma-sa-dec.yaml}";  RUNDIR="${RUNDIR:-/home/thshan@amd.com/mc_lmc/run_g05dec}";;
  *) echo bad role; exit 1;;
esac
mkdir -p "$RUNDIR"
docker rm -f "$CN" >/dev/null 2>&1 || true

cat > "$RUNDIR/run_rdma_${ROLE}.sh" <<EOF
#!/usr/bin/env bash
set -x
dpkg -i /ainic-repo/ionic-common*.deb /ainic-repo/libionic1*.deb 2>/dev/null || true
python3 - <<'PYAB' 2>&1 | tail -1 || true
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
    open(f,"w").write(src.replace(old,new,1)) if old in src else print("[patch] miss")
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
# --- SA single-node agent mooncake knobs ---
export MC_SLICE_SIZE=1048576
export MC_WORKERS_PER_CTX=4
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
# --- ionic host-MR ceiling fix ---
# ionic rejects host ibv_reg_mr >~64MiB with EINVAL[22]; the 64GiB host-DRAM
# segment must be registered in <=64MiB MRs. Validated via xnode UT (XNODE_OK
# at 64MiB, EINVAL at 128MiB+). Parallelize the ~1024 chunk registrations.
export MC_MAX_MR_SIZE=67108864
export MC_ENABLE_PARALLEL_REG_MR=1
${EXTRA_ENV:-}
exec vllm serve ${MODEL} --served-model-name Kimi-K2.5-MXFP4 \\
  --host 0.0.0.0 --port ${SERVE_PORT} \\
  --tensor-parallel-size 8 --trust-remote-code \\
  --max-model-len 262144 --gpu-memory-utilization 0.90 --block-size 1 --mm-encoder-tp-mode data \\
  --kv-cache-dtype fp8 --max-num-seqs ${CONC} \\
  --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \\
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'
EOF
chmod +x "$RUNDIR/run_rdma_${ROLE}.sh"

MOUNTS=(-v /sys:/sys -v /it-share/hf_cache:/models -v "$RUNDIR":/run_logs -v /opt/amd/ainic/deb-repo:/ainic-repo:ro -v "$DEPLOY":/deploy)
if [ "$MOUNT_SO" = "1" ]; then
  MOUNTS+=(-v "$SO/engine.cpython-312-x86_64-linux-gnu.so:$M/engine.cpython-312-x86_64-linux-gnu.so:ro")
  MOUNTS+=(-v "$SO/store.cpython-312-x86_64-linux-gnu.so:$M/store.cpython-312-x86_64-linux-gnu.so:ro")
fi
docker run -d --name "$CN" --hostname "$CN" --init --stop-timeout 10 \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 \
  --network host --ipc host --group-add video \
  --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined --privileged \
  --shm-size 64G "${MOUNTS[@]}" \
  --entrypoint "" "$IMG" bash -lc "bash /run_logs/run_rdma_${ROLE}.sh > /run_logs/mc_${ROLE}.log 2>&1"
echo "launched $CN (role=$ROLE img=$IMG mount_so=$MOUNT_SO cfg=$CFG)"; sleep 4
docker ps --filter name="$CN" --format '{{.Names}} {{.Status}}'
