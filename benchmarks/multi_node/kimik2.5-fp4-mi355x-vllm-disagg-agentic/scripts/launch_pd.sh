#!/usr/bin/env bash
# 1P1D leaderboard launcher — MI355X (ionic), Kimi-K2.5-MXFP4.
# Mooncake-TCP L3 + LMCacheConnectorV1 with ALL tiers on (L1 GPU APC + L2 host CPU + L3 TCP).
#
# Topology:  prefiller + master + proxy on g06 (10.24.112.182), decoder on g05 (10.24.112.181).
# Prereq:    patched mooncake .so in $SO (run build/patch_mooncake_hipseg.sh once; it fixes the
#            HipTransport::install segment-protocol clobber so the TCP/RDMA segment isn't
#            re-published as "hip" and remote peers can open it).
#
# Run from the SLURM head node (it ssh'es to g05/g06). Adjust IPs/paths/image as needed.
set -uo pipefail

IMG="${IMG:-kimi-lmc-mc-rocm:dmabuf}"
DEPLOY="${DEPLOY:-/home/thshan@amd.com/mc_lmc}"          # holds the yamls, proxy, patched .so
SO="${SO:-$DEPLOY/dmabuf_so}"                            # patched engine.so + store.so
MODEL="${MODEL:-/models/models--amd--Kimi-K2.5-MXFP4/snapshots/419004c8716cf22c929aa15d39b85e09a8a2091a}"
P_NODE="${P_NODE:-mia1-p01-g06}"; P_IP="${P_IP:-10.24.112.182}"
D_NODE="${D_NODE:-mia1-p01-g05}"; D_IP="${D_IP:-10.24.112.181}"
CONC="${CONC:-128}"; SERVE_PORT=8000; PROXY_PORT="${PROXY_PORT:-9100}"
M=/usr/local/lib/python3.12/dist-packages/mooncake

# ---- engine run-script generator (shared by prefiller/decoder) -----------------------------
emit_run() {  # $1=role(prefiller|decoder) $2=cfg $3=hostip
  local ROLE="$1" CFG="$2" HOSTIP="$3"
  cat <<EOF
#!/usr/bin/env bash
set -x
dpkg -i /ainic-repo/ionic-common*.deb /ainic-repo/libionic1*.deb 2>/dev/null || true
# abort-guard: upstream asserts lmcache_engine!=None on scheduler-side connector for ABORTED
# requests (None there) -> kills EngineCore on the bench's end-of-run cancel. Make it non-fatal.
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
# --- Mooncake TCP L3 on ionic (host RDMA reg is capped; TCP has no NIC reg limit) ---
export MC_FORCE_TCP=1                  # force the TCP transport (HCAs present would default to rdma)
export MC_MAX_MR_SIZE=137438953472     # 128GiB: CheckRegisterMemoryParams applies even to TCP; must exceed buffers
export MC_SLICE_SIZE=1048576
export MC_WORKERS_PER_CTX=4
exec vllm serve ${MODEL} --served-model-name Kimi-K2.5-MXFP4 \\
  --host 0.0.0.0 --port ${SERVE_PORT} \\
  --tensor-parallel-size 8 --trust-remote-code \\
  --max-model-len 262144 --gpu-memory-utilization 0.90 --block-size 1 --mm-encoder-tp-mode data \\
  --kv-cache-dtype fp8 --max-num-seqs ${CONC} \\
  --enable-prefix-caching --disable-hybrid-kv-cache-manager \\
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'
EOF
}

DOCK="--init --stop-timeout 10 --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 --network host --ipc host --group-add video \
  --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined --privileged --shm-size 64G"
SOMNT="-v $SO/engine.cpython-312-x86_64-linux-gnu.so:$M/engine.cpython-312-x86_64-linux-gnu.so:ro \
       -v $SO/store.cpython-312-x86_64-linux-gnu.so:$M/store.cpython-312-x86_64-linux-gnu.so:ro"

# ---- g06: master + prefiller ----------------------------------------------------------------
ssh -o StrictHostKeyChecking=no "$P_NODE" "bash -s" <<EOF
set -e
RUN=$DEPLOY/run_pd; mkdir -p \$RUN
docker rm -f kimi-mc-master kimi-mc-prefill kimi-mc-pd-proxy >/dev/null 2>&1 || true
# Mooncake master + HTTP metadata
docker run -d --name kimi-mc-master --network host --privileged -v /opt/amd/ainic/deb-repo:/ainic-repo:ro -v \$RUN:/run_logs \
  --entrypoint bash $IMG -lc 'dpkg -i /ainic-repo/ionic-common*.deb /ainic-repo/libionic1*.deb 2>/dev/null||true; \
    mooncake_master --enable_http_metadata_server=1 --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=8080 --rpc_address=0.0.0.0 -v=1 > /run_logs/mc_master.log 2>&1'
sleep 8
cat > \$RUN/run_prefiller.sh <<'RUNEOF'
$(emit_run prefiller lmcache-tcp-prefiller.yaml "$P_IP")
RUNEOF
docker run -d --name kimi-mc-prefill --hostname kimi-mc-prefill $DOCK \
  -v /sys:/sys -v /it-share/hf_cache:/models -v \$RUN:/run_logs -v /opt/amd/ainic/deb-repo:/ainic-repo:ro -v $DEPLOY:/deploy $SOMNT \
  --entrypoint "" $IMG bash -lc "bash /run_logs/run_prefiller.sh > /run_logs/prefiller.log 2>&1"
echo "[g06] master+prefiller launched"
EOF

# ---- g05: decoder ---------------------------------------------------------------------------
ssh -o StrictHostKeyChecking=no "$D_NODE" "bash -s" <<EOF
set -e
RUN=$DEPLOY/run_pddec; mkdir -p \$RUN
docker rm -f kimi-mc-decoder >/dev/null 2>&1 || true
cat > \$RUN/run_decoder.sh <<'RUNEOF'
$(emit_run decoder lmcache-tcp-decoder.yaml "$D_IP")
RUNEOF
docker run -d --name kimi-mc-decoder --hostname kimi-mc-decoder $DOCK \
  -v /sys:/sys -v /it-share/hf_cache:/models -v \$RUN:/run_logs -v /opt/amd/ainic/deb-repo:/ainic-repo:ro -v $DEPLOY:/deploy $SOMNT \
  --entrypoint "" $IMG bash -lc "bash /run_logs/run_decoder.sh > /run_logs/decoder.log 2>&1"
echo "[g05] decoder launched"
EOF

# ---- g06: 1P1D proxy ------------------------------------------------------------------------
ssh -o StrictHostKeyChecking=no "$P_NODE" "bash -s" <<EOF
docker run -d --name kimi-mc-pd-proxy --network host -v $DEPLOY:/deploy -v $DEPLOY/run_pd:/run_logs \
  --entrypoint bash $IMG -lc "python3 -c 'import httpx,fastapi,uvicorn' 2>/dev/null || pip install -q httpx fastapi uvicorn; \
    python3 /deploy/mc_pd_proxy.py --host 0.0.0.0 --port $PROXY_PORT --prefiller-host 127.0.0.1 --prefiller-port $SERVE_PORT --decoder-host $D_IP --decoder-port $SERVE_PORT > /run_logs/proxy.log 2>&1"
sleep 6; curl -s -m5 -o /dev/null -w 'proxy_health=%{http_code}\n' http://127.0.0.1:$PROXY_PORT/health || true
EOF
echo "launched 1P1D (Mooncake-TCP L3 + LMCacheConnectorV1, L1+L2+L3 on). Proxy: $P_IP:$PROXY_PORT"
