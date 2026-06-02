#!/usr/bin/env bash
# DSv4 FP8 1P1D disaggregated SGLang serving on MI355X (MoRI backend).
#
# The fix for the MoRI batch_write "length out of range" failures ships
# as in-tree sglang source overlays under
#   benchmarks/multi_node/amd_utils/patches/dsv4/
# (see that directory's README.md). The overlays are bind-mounted over
# the upstream sglang files inside the container — no runtime monkey-
# patching. The headline edit is the per-layer SWA-wrap in
# MoriKVManager._issue_layer_transfers (mori/conn.py): full-attention
# layers keep upstream behavior; SWA-sized layers wrap slot ids modulo
# the SWA pool capacity on BOTH sides and emit singleton WRs so every
# WR stays within the registered MR bounds.
#
# The overlay tree must exist on BOTH NODE0 and NODE1 (point
# DSV4_OVERLAY_DIR at the InferenceX checkout's patches/dsv4 dir, or at
# a synced copy). Tunable knobs are passed via ENV_DSV4 below.
#
# Usage:
#   NODE0=GPU3D78 NODE1=GPU74C0 bash run_dsv4_pd_option3.sh           # launch + idle
#   NODE0=GPU3D78 NODE1=GPU74C0 RUN_BENCH=1 bash run_dsv4_pd_option3.sh

set -euo pipefail

BENCH_ONLY="${1:-}"

NODE0="${NODE0:?set NODE0 (e.g. GPU3D78)}"
NODE1="${NODE1:?set NODE1 (e.g. GPU74C0)}"

IMAGE="${IMAGE:-rocm/sgl-dev:rocm720-mi35x-a8410de-20260502-DSv4}"
MODEL_DIR="${MODEL_DIR:-/nfsdata/hf_hub_cache}"

# ---- sglang source overlays (the fix) ----------------------------------
# Host dir holding the patches/dsv4 overlay tree (srt/...). Must exist on
# BOTH NODE0 and NODE1. Point this at the InferenceX checkout's
# benchmarks/multi_node/amd_utils/patches/dsv4 directory (or a synced copy).
DSV4_OVERLAY_DIR="${DSV4_OVERLAY_DIR:-/home/amd/zhuyc/patches_dsv4}"
# In-container sglang package root the overlays are mounted over.
SGL_PKG="${SGL_PKG:-/sgl-workspace/sglang/python/sglang}"
DSV4_OVERLAY_RELPATHS=(
    srt/disaggregation/utils.py
    srt/disaggregation/prefill.py
    srt/disaggregation/decode.py
    srt/disaggregation/mori/conn.py
    srt/model_executor/forward_batch_info.py
    srt/mem_cache/deepseekv4_memory_pool.py
    srt/layers/attention/deepseek_v4_backend_radix.py
    srt/layers/attention/nsa/index_buf_accessor.py
)
DSV4_OVERLAY_MOUNTS=""
for _rel in "${DSV4_OVERLAY_RELPATHS[@]}"; do
    DSV4_OVERLAY_MOUNTS+=" -v ${DSV4_OVERLAY_DIR}/${_rel}:${SGL_PKG}/${_rel}:ro"
done

MODEL="${MODEL:-/models/sgl-project--DeepSeek-V4-Pro-FP8}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/amd/zhuyc/dsv4_pd}"

PREFILL_TP="${PREFILL_TP:-8}"
DECODE_TP="${DECODE_TP:-8}"
PREFILL_VISIBLE="${PREFILL_VISIBLE:-0,1,2,3,4,5,6,7}"
DECODE_VISIBLE="${DECODE_VISIBLE:-0,1,2,3,4,5,6,7}"
PREFILL_PORT=8000
DECODE_PORT=8000
ROUTER_PORT=30000

ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
CONCURRENCIES=(${CONCURRENCIES:-4})
RUN_BENCH="${RUN_BENCH:-0}"

CTR_PREFIX="${CTR_PREFIX:-zhuyc-dsv4disagg}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR_HOST="${PROJECT_ROOT}/logs_${TIMESTAMP}"
RESULT_DIR="/workspace/logs"

SSH_OPTS="${SSH_OPTS:--i ${HOME}/.ssh/id_rsa -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

ENV_MORI="\
    -e SGLANG_MORI_DISPATCH_DTYPE=auto \
    -e SGLANG_MORI_FP8_COMB=true \
    -e SGLANG_MORI_QP_PER_TRANSFER=4 \
    -e SGLANG_MORI_NUM_WORKERS=4 \
    -e MORI_SHMEM_MODE=ISOLATION \
    -e MORI_MAX_DISPATCH_TOKENS_PREFILL=8192 \
    -e MORI_MAX_DISPATCH_TOKENS_DECODE=512"

# DSv4 disagg knobs consumed by the sglang overlays (patches/dsv4).
# The overlays bake in identity full_to_swa mapping + sparse state
# transfer + SWA-wrap unconditionally; these env vars only tune the
# behavior/diagnostics of that code (defaults match the validated runs).
ENV_DSV4="\
    -e SGLANG_OPT_DPSK_V4_RADIX=1 \
    -e DSV4_LAYER_SWA_WRAP=${DSV4_LAYER_SWA_WRAP:-1} \
    -e DSV4_LAYER_PREFLIGHT_STRICT=${DSV4_LAYER_PREFLIGHT_STRICT:-1} \
    -e DSV4_STATE_MAX_ITEM_LEN=${DSV4_STATE_MAX_ITEM_LEN:-0} \
    -e DSV4_STATE_CHUNK_CAP_BYTES=${DSV4_STATE_CHUNK_CAP_BYTES:-524288} \
    -e DSV4_STATE_AGGREGATE_BATCH=${DSV4_STATE_AGGREGATE_BATCH:-1} \
    -e DSV4_STATE_SERIALIZE=${DSV4_STATE_SERIALIZE:-0} \
    -e DSV4_STATE_DIAG_EVERY=${DSV4_STATE_DIAG_EVERY:-10} \
    -e DSV4_STATE_DIAG_FAILED=${DSV4_STATE_DIAG_FAILED:-1} \
    -e DSV4_STATE_DIAG_BATCHWRITE=${DSV4_STATE_DIAG_BATCHWRITE:-1} \
    -e DSV4_LAYER_DIAG_EVERY=${DSV4_LAYER_DIAG_EVERY:-100} \
    -e DSV4_LAYER_DIAG_FAILED=${DSV4_LAYER_DIAG_FAILED:-1} \
    -e DSV4_LAYER_DIAG_BATCHWRITE=${DSV4_LAYER_DIAG_BATCHWRITE:-1} \
    -e SGLANG_USE_AITER=1 \
    -e SGLANG_REASONING_EFFORT=max \
    -e SGLANG_OPT_USE_FUSED_COMPRESS=true \
    -e SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false \
    -e SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false \
    -e SGLANG_OPT_USE_FUSED_HASH_TOPK=false \
    -e SGLANG_HACK_FLASHMLA_BACKEND=tilelang \
    -e SGLANG_OPT_USE_TILELANG_INDEXER=true \
    -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=false \
    -e SGLANG_OPT_USE_TILELANG_MHC_PRE=false \
    -e SGLANG_OPT_USE_TILELANG_MHC_POST=false \
    -e SGLANG_ENABLE_THINKING=1 \
    -e SGLANG_USE_ROCM700A=1 \
    -e SGLANG_TOPK_TRANSFORM_512_TORCH=0 \
    -e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 \
    -e SGLANG_DSV4_FP4_EXPERTS=false \
    -e SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false \
    -e SGLANG_OPT_USE_FUSED_STORE_CACHE=false \
    -e SGLANG_FORCE_TRITON_MOE_FP8=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=/sgl-workspace/sglang/python \
    -e SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=10000000 \
    -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=7200 \
    -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=7200 \
    -e SGLANG_LOG_MS=true"

log "Resolving node IPs..."
NODE0_IP=$(ssh $SSH_OPTS "$NODE0" "ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print \$7; exit}'")
NODE1_IP=$(ssh $SSH_OPTS "$NODE1" "ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print \$7; exit}'")
log "Node 0: $NODE0 -> $NODE0_IP"
log "Node 1: $NODE1 -> $NODE1_IP"

ssh $SSH_OPTS "$NODE0" "test -d '${MODEL_DIR}/sgl-project--DeepSeek-V4-Pro-FP8' || { echo 'ERROR: DSv4 weights missing'; exit 1; }"

log "Verifying sglang overlay tree on both nodes (${DSV4_OVERLAY_DIR})..."
for _node in "$NODE0" "$NODE1"; do
    for _rel in "${DSV4_OVERLAY_RELPATHS[@]}"; do
        ssh $SSH_OPTS "$_node" "test -f '${DSV4_OVERLAY_DIR}/${_rel}'" \
            || { echo "ERROR: overlay missing on ${_node}: ${DSV4_OVERLAY_DIR}/${_rel}"; exit 1; }
    done
done
log "Overlay tree present on ${NODE0} and ${NODE1}."

cleanup() {
    log "Tearing down ${CTR_PREFIX}-* containers..."
    ssh $SSH_OPTS "$NODE0" "docker rm -f ${CTR_PREFIX}-prefill ${CTR_PREFIX}-router 2>/dev/null" || true
    ssh $SSH_OPTS "$NODE1" "docker rm -f ${CTR_PREFIX}-decode 2>/dev/null" || true
    log "Cleanup done."
}
trap cleanup EXIT INT TERM

DOCKER_BASE="docker run -d --init \
    --device /dev/dri --device /dev/kfd --device=/dev/infiniband \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v ${MODEL_DIR}:/models \
    -v ${LOG_DIR_HOST}:/workspace/logs \
    ${DSV4_OVERLAY_MOUNTS} \
    --shm-size 128G"

wait_for_server() {
    local node=$1 port=$2 label=$3 timeout=${4:-3600}
    local start=$(date +%s)
    log "Waiting for ${label} (${node}:${port})..."
    while true; do
        if ssh $SSH_OPTS "$node" "curl -sf http://localhost:${port}/health >/dev/null 2>&1"; then
            log "${label} is UP"
            return 0
        fi
        if (( $(date +%s) - start >= timeout )); then
            log "TIMEOUT: ${label}"
            return 1
        fi
        sleep 10
    done
}

if [[ "${BENCH_ONLY}" == "bench" ]]; then
    :
else
    cleanup 2>/dev/null || true
    ssh $SSH_OPTS "$NODE0" "mkdir -p ${LOG_DIR_HOST}" &
    ssh $SSH_OPTS "$NODE1" "mkdir -p ${LOG_DIR_HOST}" &
    wait

    log "Starting PREFILL (Option3 — identity mapping + sparse state copy) on ${NODE0}..."
    ssh $SSH_OPTS "$NODE0" "${DOCKER_BASE} \
        --name ${CTR_PREFIX}-prefill \
        -e HIP_VISIBLE_DEVICES=${PREFILL_VISIBLE} \
        -e ROCR_VISIBLE_DEVICES=${PREFILL_VISIBLE} \
        ${ENV_MORI} ${ENV_DSV4} \
        ${IMAGE} \
        bash -c 'python3 -m sglang.launch_server \
            --model-path ${MODEL} \
            --host 0.0.0.0 --port ${PREFILL_PORT} \
            --disaggregation-mode prefill \
            --disaggregation-transfer-backend mori \
            --trust-remote-code --skip-server-warmup \
            --disable-radix-cache --disable-cuda-graph \
            --tp-size ${PREFILL_TP} --dp-size 1 \
            --attention-backend compressed --page-size 256 \
            --chunked-prefill-size 8192 --max-running-requests 256 \
            --mem-fraction-static 0.85 \
            --cuda-graph-bs 1 2 3 \
            --disable-shared-experts-fusion \
            --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
            --watchdog-timeout 1800 \
            2>&1 | tee ${RESULT_DIR}/pd_prefill_${TIMESTAMP}.log'"

    log "Starting DECODE (Option3 — identity mapping + sparse state copy) on ${NODE1}..."
    ssh $SSH_OPTS "$NODE1" "${DOCKER_BASE} \
        --name ${CTR_PREFIX}-decode \
        -e HIP_VISIBLE_DEVICES=${DECODE_VISIBLE} \
        -e ROCR_VISIBLE_DEVICES=${DECODE_VISIBLE} \
        ${ENV_MORI} ${ENV_DSV4} \
        ${IMAGE} \
        bash -c 'python3 -m sglang.launch_server \
            --model-path ${MODEL} \
            --host 0.0.0.0 --port ${DECODE_PORT} \
            --disaggregation-mode decode \
            --disaggregation-transfer-backend mori \
            --trust-remote-code --skip-server-warmup \
            --disable-radix-cache --disable-cuda-graph \
            --tp-size ${DECODE_TP} --dp-size 1 \
            --attention-backend compressed --page-size 256 \
            --chunked-prefill-size 4096 --max-running-requests 4096 \
            --mem-fraction-static 0.88 \
            --disable-shared-experts-fusion \
            --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
            --decode-log-interval 1000 \
            --watchdog-timeout 1800 \
            2>&1 | tee ${RESULT_DIR}/pd_decode_${TIMESTAMP}.log'"

    wait_for_server "$NODE0" $PREFILL_PORT "Prefill" 7200 &
    wait_for_server "$NODE1" $DECODE_PORT "Decode" 7200 &
    wait

    log "Starting ROUTER on ${NODE0}..."
    ssh $SSH_OPTS "$NODE0" "${DOCKER_BASE} \
        --name ${CTR_PREFIX}-router \
        ${IMAGE} \
        bash -c 'python3 -m sglang_router.launch_router \
            --pd-disaggregation --host 0.0.0.0 --port ${ROUTER_PORT} \
            --tokenizer-path ${MODEL} --mini-lb \
            --prefill http://${NODE0_IP}:${PREFILL_PORT} \
            --decode  http://${NODE1_IP}:${DECODE_PORT} \
            2>&1 | tee ${RESULT_DIR}/pd_router_${TIMESTAMP}.log'"

    wait_for_server "$NODE0" $ROUTER_PORT "Router" 600

    echo ""
    echo "=========================================="
    echo " DSv4 FP8 1P1D Option-3 ready"
    echo " Router: http://${NODE0_IP}:${ROUTER_PORT}"
    echo " Logs:   ${LOG_DIR_HOST}"
    echo "=========================================="
fi

if [[ "${RUN_BENCH}" == "1" || "${BENCH_ONLY}" == "bench" ]]; then
    BENCH_URL="http://${NODE0_IP}:${ROUTER_PORT}"
    for CONC in "${CONCURRENCIES[@]}"; do
        PROMPTS=$((CONC * 10))
        (( PROMPTS < 10 )) && PROMPTS=10
        log "bench conc=${CONC} prompts=${PROMPTS}"
        ssh $SSH_OPTS "$NODE0" "docker exec ${CTR_PREFIX}-prefill bash -c '\
            python3 -m sglang.bench_serving \
                --model ${MODEL} --backend sglang-oai --base-url ${BENCH_URL} \
                --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \
                --random-range-ratio ${RANDOM_RANGE_RATIO} \
                --num-prompts ${PROMPTS} --max-concurrency ${CONC} \
                --request-rate inf --warmup-requests 0 --seed 2026 \
                --output-file ${RESULT_DIR}/dsv4_pd_conc${CONC}_${TIMESTAMP}.json'"
    done
    log "Benchmark done."
fi

if [[ "${RUN_BENCH}" != "1" && "${BENCH_ONLY}" != "bench" ]]; then
    log "Launch-only mode (RUN_BENCH=0). Cluster left running; Ctrl+C tears down."
fi

while true; do sleep 3600; done
