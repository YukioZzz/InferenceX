#!/bin/bash
# Dual-Engine Disaggregated Benchmark Runner
#
# ENGINE=sglang (default): SGLang benchmark
# ENGINE=vllm:             vLLM benchmark
#
# Produces JSON result files via benchmark_serving.py so that the CI pipeline
# can collect and process results.
#
# Usage: bash bench.sh <n_prefill> <n_decode> <prefill_gpus> <decode_gpus> \
#            <model_dir> <model_name> <log_path> <isl> <osl> \
#            <concurrency_list> <req_rate> <random_range_ratio> <num_prompts_multiplier>

ENGINE="${ENGINE:-sglang-disagg}"

n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
model_path=$5
model_name=$6
MODEL_PATH="${MODEL_PATH:-${model_path}/${model_name}}"
# vllm-disagg uses --served-model-name MODEL_NAME; sglang defaults to MODEL_PATH
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    BENCH_MODEL="${MODEL_NAME:-${MODEL_PATH}}"
else
    BENCH_MODEL="${MODEL_PATH}"
fi
log_path=$7

chosen_isl=${8:-1024}
chosen_osl=${9:-1024}
concurrency_list=${10:-"512x1"}
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    chosen_req_rate=${11:-inf}
else
    chosen_req_rate=${11:-1}
fi
random_range_ratio=${12:-0.8}
num_prompts_multiplier=${13:-10}

IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"

ROUTER_PORT="${ROUTER_PORT:-30000}"

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[0]}; ${chosen_req_rate}"

profile_folder="${log_path}/${ENGINE}_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p "$profile_folder"

source "$(dirname "$0")/../../benchmark_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

# =============================================================================
# Agentic trace-replay path (IS_AGENTIC=1)
# -----------------------------------------------------------------------------
# Replays the SemiAnalysis WEKA agentic-coding corpus against the disagg proxy
# (ROUTER_PORT) via aiperf, once per concurrency, reusing the same
# benchmark_lib.sh helpers as the single-node agentic path. Writes
# ${RESULT_FILENAME}_conc<N>.json at the workspace root for the multinode
# workflow's result collector. Gated so the default fixed-seq path is untouched.
# =============================================================================
if [[ "${IS_AGENTIC:-0}" == "1" ]]; then
    export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-$REPO_ROOT}"
    export MODEL="${MODEL:-$BENCH_MODEL}"
    export PORT="$ROUTER_PORT"
    export DURATION="${DURATION:-1800}"
    # 052726-256k is supported by resolve_trace_source on main; override per recipe.
    export WEKA_LOADER_OVERRIDE="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_with_subagents_256k}"
    # Write per-conc result JSONs straight into the host-mounted logs dir so the
    # runner-side collector (launch_mi355x-amds.sh) can stage them to the workspace.
    AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-/benchmark_logs}"
    export AGENTIC_OUTPUT_DIR
    RESULT_FILENAME_BASE="${RESULT_FILENAME:-agentic_bench}"

    echo "[BENCH] IS_AGENTIC=1 -> agentic trace replay (model=$MODEL port=$PORT dur=${DURATION}s conc=${concurrency_list})"
    resolve_trace_source
    install_agentic_deps

    # The multinode workflow runs ONE conc per matrix job and its RESULT_FILENAME
    # already encodes the conc (..._conc8_..._08). So for a single conc, write
    # exactly ${RESULT_FILENAME}.json (no extra _conc suffix, which broke the
    # workflow's success gate). Only suffix when this script sweeps >1 conc.
    n_concs=${#chosen_concurrencies[@]}
    any_failed=0
    for max_concurrency in "${chosen_concurrencies[@]}"; do
        echo "=========================================="
        echo "Agentic replay: conc=$max_concurrency"
        echo "=========================================="
        CONC_RESULT_DIR="${profile_folder}/conc${max_concurrency}"
        mkdir -p "$CONC_RESULT_DIR"
        export CONC="$max_concurrency" USERS="$max_concurrency"
        if [[ "$n_concs" -gt 1 ]]; then
            PER_RF="${RESULT_FILENAME_BASE}_conc${max_concurrency}"
        else
            PER_RF="${RESULT_FILENAME_BASE}"
        fi
        build_replay_cmd "$CONC_RESULT_DIR"
        RESULT_DIR="$CONC_RESULT_DIR" \
        AGENTIC_OUTPUT_DIR="$AGENTIC_OUTPUT_DIR" \
        RESULT_FILENAME="$PER_RF" \
            run_agentic_replay_and_write_outputs "$CONC_RESULT_DIR" || any_failed=1
        echo "-----------------------------------------"
        sleep 10
    done
    if [[ "$any_failed" -ne 0 ]]; then
        echo "WARNING: at least one agentic conc exited non-zero. Dumping proxy/engine logs for diagnosis:" >&2
        for _l in "${log_path}"/mc_pd_proxy.log "${log_path}"/prefill_*.log "${log_path}"/mc_master.log; do
            [ -f "$_l" ] || continue
            echo "===== tail $_l =====" >&2
            tail -n 40 "$_l" 2>/dev/null | tr '\r' '\n' | grep -aiE "proxy|error|exception|traceback|500|400|mooncake|lmcache|fail|refused" | tail -25 >&2
        done
    fi
    exit 0
fi

for max_concurrency in "${chosen_concurrencies[@]}"; do

    export_file="${profile_folder}/concurrency_${max_concurrency}_req_rate_${chosen_req_rate}_gpus_$((prefill_gpus+decode_gpus))_ctx_${prefill_gpus}_gen_${decode_gpus}"

    num_prompts=$(( max_concurrency * num_prompts_multiplier ))
    if [[ "$num_prompts" -lt 16 ]]; then
        num_prompts=16
    fi

    echo "profile_folder: $profile_folder"
    echo "max_concurrency: $max_concurrency"
    echo "chosen_req_rate: $chosen_req_rate"
    echo "MODEL_PATH: $MODEL_PATH"
    echo "ROUTER_PORT: $ROUTER_PORT"
    echo "chosen_isl: $chosen_isl"
    echo "chosen_osl: $chosen_osl"
    echo "num_prompts: $num_prompts"
    echo "export_file: $export_file"

    # Engine-specific extra flags
    extra_flags=""
    if [[ "$ENGINE" == "vllm-disagg" ]]; then
        extra_flags="--trust-remote-code --tokenizer $MODEL_PATH"
    else
        if [ "$IS_MTP" = "true" ]; then
            extra_flags="--use-chat-template"
        fi
    fi

    run_benchmark_serving \
        --bench-serving-dir "$REPO_ROOT" \
        --model "$BENCH_MODEL" \
        --port "$ROUTER_PORT" \
        --backend openai \
        --input-len "$chosen_isl" \
        --output-len "$chosen_osl" \
        --random-range-ratio "$random_range_ratio" \
        --num-prompts "$num_prompts" \
        --max-concurrency "$max_concurrency" \
        --result-filename "$export_file" \
        --result-dir /workspace/ \
        $extra_flags

    echo "-----------------------------------------"

    # vLLM: cooldown between rounds for idle KV block reaper
    if [[ "$ENGINE" == "vllm-disagg" ]]; then
        echo "[BENCH] Cooldown: waiting 10s for idle KV block reaper..."
        sleep 10
    fi
done
