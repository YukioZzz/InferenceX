#!/usr/bin/env bash
# MooncakeConnector PD (mooncake_protocol=tcp, P2PHANDSHAKE) on DO MI350X VF.
# Usage: do_mc_launch.sh <proxy|prefill|decode>
set -uo pipefail
ROLE="${1:?usage: do_mc_launch.sh <proxy|prefill|decode>}"
IMG="kimi-lmc-mc-rocm:latest"
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
  NAME=kimi-mc-proxy
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d $(common_docker "$NAME" kimi-mc-proxy) --entrypoint bash "$IMG" -lc \
    "python3 /run_logs/toy_proxy_server.py --host 0.0.0.0 --port ${PROXY_PORT} --prefiller-hosts ${P_IP} --prefiller-ports ${SERVER_PORT} --decoder-hosts ${D_IP} --decoder-ports ${SERVER_PORT} > /run_logs/mc_proxy.log 2>&1"
  echo "launched $NAME"; sleep 6
  docker ps --filter "name=$NAME" --format '{{.Names}} {{.Status}}'
  tail -n 6 "$RUNDIR/mc_proxy.log" 2>/dev/null | tr '\r' '\n'
  exit 0
fi

case "$ROLE" in
  prefill) HOSTIP="$P_IP"; KVROLE="kv_producer"; CN="kimi-mc-prefill";;
  decode)  HOSTIP="$D_IP"; KVROLE="kv_consumer"; CN="kimi-mc-decode";;
  *) echo "bad role"; exit 1;;
esac

KVCFG="{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"${KVROLE}\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"tcp\"}}"

cat > "$RUNDIR/run_mc_${ROLE}.sh" <<EOF
#!/usr/bin/env bash
set -x
export PATH=/opt/rocm/bin:\$PATH
export VLLM_USE_V1=1
export VLLM_HOST_IP=${HOSTIP}
export VLLM_NIXL_SIDE_CHANNEL_HOST=${HOSTIP}
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600
export MC_TCP_INTERFACE=eth2
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
chmod +x "$RUNDIR/run_mc_${ROLE}.sh"

docker rm -f "$CN" >/dev/null 2>&1 || true
docker run -d $(common_docker "$CN" "$CN") --entrypoint "" "$IMG" \
  bash -lc "bash /run_logs/run_mc_${ROLE}.sh > /run_logs/mc_${ROLE}.log 2>&1"
echo "launched $CN (role=$ROLE, hostip=$HOSTIP)"; sleep 4
docker ps --filter "name=$CN" --format '{{.Names}} {{.Status}}'
