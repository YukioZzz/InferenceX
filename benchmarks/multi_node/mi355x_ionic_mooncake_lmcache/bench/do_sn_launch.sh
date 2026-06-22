#!/usr/bin/env bash
# Single-node Kimi-K2.5 MXFP4 engine on DO-6: TP8 + LMCache L2 (no cross-node PD).
# Params aligned to the PD experiment / SA single-node recipe.
# Env: CONC (max-num-seqs, default 128), EP (1 => --enable-expert-parallel).
set -uo pipefail
IMG="kimi-lmc-mc-rocm:latest"
RUNDIR=/root/run_logs
mkdir -p "$RUNDIR"
SERVER_PORT=10001
CONC="${CONC:-128}"
EP="${EP:-0}"
LMC_L1_SIZE_GB="${LMC_L1_SIZE_GB:-1200}"
LMC_READ_TTL="${LMC_READ_TTL:-7200}"
HOSTIP="192.168.0.6"
CN="kimi-sn-engine"

# free port 10001: stop any leftover PD proxy / prefill / old SN engine
docker rm -f kimi-mclmc-proxy kimi-mclmc-prefill "$CN" >/dev/null 2>&1 || true

EP_ARG=""
[ "$EP" = "1" ] && EP_ARG="--enable-expert-parallel"

LMC_CONN='{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":5555}}'

cat > "$RUNDIR/run_sn.sh" <<EOF
#!/usr/bin/env bash
set -x
export PATH=/opt/rocm/bin:\$PATH
export VLLM_USE_V1=1
export VLLM_HOST_IP=${HOSTIP}
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_USE_AITER_TRITON_SILU_MUL=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export PYTHONNOUSERSITE=1

echo "[run] starting LMCache MP server..."
lmcache server --host 127.0.0.1 --port 5555 --http-host 127.0.0.1 --http-port 8080 \\
  --l1-size-gb ${LMC_L1_SIZE_GB} --l1-init-size-gb 20 --l1-read-ttl-seconds ${LMC_READ_TTL} \\
  --chunk-size 256 --max-workers 8 --eviction-policy LRU > /run_logs/lmcache_sn.log 2>&1 &
for i in \$(seq 1 90); do
  curl -sf --max-time 3 http://127.0.0.1:8080/healthcheck >/dev/null 2>&1 && { echo "[run] LMCache MP healthy"; break; }
  sleep 2
done

exec vllm serve /models/Kimi-K2.5-MXFP4 \\
  --host 0.0.0.0 --port ${SERVER_PORT} \\
  --tensor-parallel-size 8 --trust-remote-code ${EP_ARG} \\
  --max-model-len 262144 --gpu-memory-utilization 0.90 --block-size 1 --mm-encoder-tp-mode data \\
  --kv-cache-dtype fp8 --max-num-seqs ${CONC} \\
  --enable-prefix-caching --disable-hybrid-kv-cache-manager \\
  --kv-transfer-config '${LMC_CONN}'
EOF
chmod +x "$RUNDIR/run_sn.sh"

docker run -d --name "$CN" --hostname "$CN" --init --stop-timeout 10 \
  --device /dev/dri --device /dev/kfd \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 \
  --network host --ipc host --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  --shm-size 128G -v /sys:/sys -v /root/models:/models -v "$RUNDIR":/run_logs \
  --entrypoint "" "$IMG" bash -lc "bash /run_logs/run_sn.sh > /run_logs/sn_engine.log 2>&1"
echo "launched $CN (EP=$EP, max-num-seqs=$CONC)"; sleep 4
docker ps --filter "name=$CN" --format '{{.Names}} {{.Status}}'
