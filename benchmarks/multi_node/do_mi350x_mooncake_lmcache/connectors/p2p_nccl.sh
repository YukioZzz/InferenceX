#!/usr/bin/env bash
# P2pNccl PD on DO MI350X (NCCL forced to host-staged socket; GPUDirect dead on VF).
# Usage: do_p2p_launch.sh <proxy|prefill|decode>
set -uo pipefail
ROLE="${1:?usage: do_p2p_launch.sh <proxy|prefill|decode>}"
IMG="${P2P_IMG:-aigmkt/vllm-openai-rocm:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036}"
RUNDIR=/root/run_logs
mkdir -p "$RUNDIR"
PROXY_IP="192.168.0.6"; PROXY_PORT=30001; SERVER_PORT=2584
MODEL="/models/Kimi-K2.5-MXFP4"

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
  NAME=kimi-p2p-proxy
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d $(common_docker "$NAME" kimi-proxy) --entrypoint bash "$IMG" -lc \
    "python3 -c 'import quart,aiohttp,msgpack,zmq' 2>/dev/null || pip install -q quart aiohttp msgpack pyzmq; python3 /run_logs/p2p_proxy.py > /run_logs/p2p_proxy.log 2>&1"
  echo "launched $NAME"; sleep 6
  docker ps --filter "name=$NAME" --format '{{.Names}} {{.Status}}'
  tail -n 8 "$RUNDIR/p2p_proxy.log" 2>/dev/null | tr '\r' '\n'
  exit 0
fi

case "$ROLE" in
  prefill) HOSTIP="192.168.0.6"; KVROLE="kv_producer"; KVPORT="21001"; KVBUF="1e1"; CN="kimi-p2p-prefill";;
  decode)  HOSTIP="192.168.0.7"; KVROLE="kv_consumer"; KVPORT="22001"; KVBUF="8e9"; CN="kimi-p2p-decode";;
  *) echo "bad role"; exit 1;;
esac

KVCFG="{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"${KVROLE}\",\"kv_buffer_size\":\"${KVBUF}\",\"kv_port\":\"${KVPORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${PROXY_IP}\",\"proxy_port\":\"${PROXY_PORT}\",\"http_port\":\"${SERVER_PORT}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}"

cat > "$RUNDIR/run_p2p_${ROLE}.sh" <<EOF
#!/usr/bin/env bash
set -x
export VLLM_USE_V1=1
export VLLM_HOST_IP=${HOSTIP}
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth2
export NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME=eth2
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_USE_AITER_TRITON_SILU_MUL=0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
exec vllm serve ${MODEL} \\
  --host 0.0.0.0 --port ${SERVER_PORT} \\
  --tensor-parallel-size 8 --trust-remote-code --enforce-eager \\
  --max-model-len 32768 --gpu-memory-utilization 0.85 --mm-encoder-tp-mode data \\
  --kv-transfer-config '${KVCFG}'
EOF
chmod +x "$RUNDIR/run_p2p_${ROLE}.sh"

docker rm -f "$CN" >/dev/null 2>&1 || true
docker run -d $(common_docker "$CN" "$CN") --entrypoint "" "$IMG" \
  bash -lc "bash /run_logs/run_p2p_${ROLE}.sh > /run_logs/p2p_${ROLE}.log 2>&1"
echo "launched $CN (role=$ROLE, hostip=$HOSTIP)"; sleep 4
docker ps --filter "name=$CN" --format '{{.Names}} {{.Status}}'
