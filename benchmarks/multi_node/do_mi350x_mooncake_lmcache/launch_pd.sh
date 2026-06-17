#!/usr/bin/env bash
# Mooncake(tcp PD transfer) + LMCache L2  via MultiConnector, on DO MI350X VF.
# Usage: do_mc_lmc_launch.sh <proxy|prefill|decode>
set -uo pipefail
ROLE="${1:?usage: <proxy|prefill|decode>}"
IMG="kimi-lmc-mc-rocm:latest"
RUNDIR=/root/run_logs
mkdir -p "$RUNDIR"
SERVER_PORT=2584
P_IP="192.168.0.6"; D_IP="192.168.0.7"; PROXY_PORT=10001
# Engine max batch = benchmark concurrency (aligns with SN recipe --max-num-seqs $CONC).
CONC="${CONC:-32}"
# LMCache L1 host-DRAM pool. DO MI350X VF nodes have ~2 TB host DRAM (not the
# MI355X bare-metal ~2.7 TB), so we size to 1200 GB instead of the SN recipe's
# 3000 GB to keep ~700 GB headroom for vLLM worker RSS + page cache.
LMC_L1_SIZE_GB="${LMC_L1_SIZE_GB:-1200}"
LMC_READ_TTL="${LMC_READ_TTL:-7200}"

common_docker() {  # $1=name $2=hostname
  echo --name "$1" --hostname "$2" --init --stop-timeout 10 \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --device=/dev/infiniband/rdma_cm \
    --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/uverbs1 \
    --device=/dev/infiniband/uverbs2 --device=/dev/infiniband/uverbs3 \
    --device=/dev/infiniband/uverbs4 --device=/dev/infiniband/uverbs5 \
    --device=/dev/infiniband/uverbs6 --device=/dev/infiniband/uverbs7 \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    --shm-size 128G -v /sys:/sys -v /root/models:/models -v "$RUNDIR":/run_logs
}

if [ "$ROLE" = "proxy" ]; then
  NAME=kimi-mclmc-proxy
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d $(common_docker "$NAME" kimi-mclmc-proxy) --entrypoint bash "$IMG" -lc \
    "python3 /run_logs/toy_proxy_server.py --host 0.0.0.0 --port ${PROXY_PORT} --prefiller-hosts ${P_IP} --prefiller-ports ${SERVER_PORT} --decoder-hosts ${D_IP} --decoder-ports ${SERVER_PORT} > /run_logs/mclmc_proxy.log 2>&1"
  echo "launched $NAME"; sleep 6
  docker ps --filter "name=$NAME" --format '{{.Names}} {{.Status}}'; tail -n 5 "$RUNDIR/mclmc_proxy.log" 2>/dev/null | tr '\r' '\n'
  exit 0
fi

case "$ROLE" in
  prefill) HOSTIP="$P_IP"; KVROLE="kv_producer"; CN="kimi-mclmc-prefill";;
  decode)  HOSTIP="$D_IP"; KVROLE="kv_consumer"; CN="kimi-mclmc-decode";;
  *) echo "bad role"; exit 1;;
esac

MC_CONN="{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"${KVROLE}\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"tcp\"}}"
LMC_CONN="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":5555}}"
KVCFG="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"${KVROLE}\",\"kv_connector_extra_config\":{\"connectors\":[${MC_CONN},${LMC_CONN}]}}"

cat > "$RUNDIR/run_mclmc_${ROLE}.sh" <<EOF
#!/usr/bin/env bash
set -x
export PATH=/opt/rocm/bin:\$PATH
export VLLM_USE_V1=1
export VLLM_HOST_IP=${HOSTIP}
export VLLM_NIXL_SIDE_CHANNEL_HOST=${HOSTIP}
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600
export GLOO_SOCKET_IFNAME=eth2
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_USE_AITER_TRITON_SILU_MUL=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# SN-recipe ROCm alignment
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export PYTHONNOUSERSITE=1

echo "[run] starting LMCache MP server..."
lmcache server --host 127.0.0.1 --port 5555 --http-host 127.0.0.1 --http-port 8080 \\
  --l1-size-gb ${LMC_L1_SIZE_GB} --l1-init-size-gb 20 --l1-read-ttl-seconds ${LMC_READ_TTL} \\
  --chunk-size 256 --max-workers 8 --eviction-policy LRU > /run_logs/lmcache_${ROLE}.log 2>&1 &
for i in \$(seq 1 90); do
  curl -sf --max-time 3 http://127.0.0.1:8080/healthcheck >/dev/null 2>&1 && { echo "[run] LMCache MP healthy"; break; }
  sleep 2
done

exec vllm serve /models/Kimi-K2.5-MXFP4 \\
  --host 0.0.0.0 --port ${SERVER_PORT} \\
  --tensor-parallel-size 8 --trust-remote-code \\
  --max-model-len 262144 --gpu-memory-utilization 0.90 --block-size 1 --mm-encoder-tp-mode data \\
  --kv-cache-dtype fp8 --max-num-seqs ${CONC} \\
  --enable-prefix-caching --disable-hybrid-kv-cache-manager \\
  --kv-transfer-config '${KVCFG}'
EOF
chmod +x "$RUNDIR/run_mclmc_${ROLE}.sh"

docker rm -f "$CN" >/dev/null 2>&1 || true
docker run -d $(common_docker "$CN" "$CN") --entrypoint "" "$IMG" \
  bash -lc "bash /run_logs/run_mclmc_${ROLE}.sh > /run_logs/mclmc_${ROLE}.log 2>&1"
echo "launched $CN (role=$ROLE)"; sleep 4
docker ps --filter "name=$CN" --format '{{.Names}} {{.Status}}'
