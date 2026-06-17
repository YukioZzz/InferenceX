#!/usr/bin/env bash
# NixlConnector PD on DO MI350X. kv_buffer_device=cpu => host-staged (bypasses GPUDirect, VF-safe).
# Usage: do_nixl_launch.sh <proxy|prefill|decode>
set -uo pipefail
ROLE="${1:?usage: do_nixl_launch.sh <proxy|prefill|decode>}"
IMG="aigmkt/vllm-openai-rocm:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036"
RUNDIR=/root/run_logs
mkdir -p "$RUNDIR"
SERVER_PORT=2584
P_IP="192.168.0.6"; D_IP="192.168.0.7"; PROXY_PORT=10001

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
  NAME=kimi-nixl-proxy
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d $(common_docker "$NAME" kimi-nixl-proxy) --entrypoint bash "$IMG" -lc \
    "python3 /run_logs/toy_proxy_server.py --host 0.0.0.0 --port ${PROXY_PORT} --prefiller-hosts ${P_IP} --prefiller-ports ${SERVER_PORT} --decoder-hosts ${D_IP} --decoder-ports ${SERVER_PORT} > /run_logs/nixl_proxy.log 2>&1"
  echo "launched $NAME"; sleep 6
  docker ps --filter "name=$NAME" --format '{{.Names}} {{.Status}}'
  tail -n 8 "$RUNDIR/nixl_proxy.log" 2>/dev/null | tr '\r' '\n'
  exit 0
fi

case "$ROLE" in
  prefill) HOSTIP="$P_IP"; KVROLE="kv_producer"; CN="kimi-nixl-prefill";;
  decode)  HOSTIP="$D_IP"; KVROLE="kv_consumer"; CN="kimi-nixl-decode";;
  *) echo "bad role"; exit 1;;
esac

KVCFG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"${KVROLE}\",\"kv_buffer_device\":\"cpu\"}"

cat > "$RUNDIR/run_nixl_${ROLE}.sh" <<EOF
#!/usr/bin/env bash
set -x
export VLLM_USE_V1=1
export VLLM_NIXL_SIDE_CHANNEL_HOST=${HOSTIP}
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600
export UCX_TLS=tcp,sm,self
export NCCL_SOCKET_IFNAME=eth2
export GLOO_SOCKET_IFNAME=eth2
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_USE_AITER_TRITON_SILU_MUL=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
exec vllm serve /models/Kimi-K2.5-MXFP4 \\
  --host 0.0.0.0 --port ${SERVER_PORT} \\
  --tensor-parallel-size 8 --trust-remote-code --enforce-eager \\
  --max-model-len 32768 --gpu-memory-utilization 0.85 --mm-encoder-tp-mode data \\
  --kv-transfer-config '${KVCFG}'
EOF
chmod +x "$RUNDIR/run_nixl_${ROLE}.sh"

docker rm -f "$CN" >/dev/null 2>&1 || true
docker run -d $(common_docker "$CN" "$CN") --entrypoint "" "$IMG" \
  bash -lc "bash /run_logs/run_nixl_${ROLE}.sh > /run_logs/nixl_${ROLE}.log 2>&1"
echo "launched $CN (role=$ROLE, hostip=$HOSTIP)"; sleep 4
docker ps --filter "name=$CN" --format '{{.Names}} {{.Status}}'
