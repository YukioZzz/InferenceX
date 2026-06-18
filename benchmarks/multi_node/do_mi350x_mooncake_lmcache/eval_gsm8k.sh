#!/usr/bin/env bash
# GSM8K correctness check through the 1P1D PD proxy (port 10001).
# Usage: do_gsm8k.sh <chat|comp> [LIMIT]
#   chat -> local-chat-completions --apply_chat_template  -> /v1/chat/completions
#   comp -> local-completions (no chat template)          -> /v1/completions
set -uo pipefail
MODE="${1:?usage: do_gsm8k.sh <chat|comp> [LIMIT]}"
LIMIT="${2:-50}"
IMG="kimi-lmc-mc-rocm:latest"
RUNDIR=/root/run_logs
NAME="kimi-gsm8k-${MODE}"
RES="/run_logs/gsm8k_${MODE}"

if [ "$MODE" = "chat" ]; then
  LM_MODEL="local-chat-completions"
  CHAT_FLAG="--apply_chat_template"
  EP_PATH="/v1/chat/completions"
else
  LM_MODEL="local-completions"
  CHAT_FLAG=""
  EP_PATH="/v1/completions"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host \
  -v /root/InferenceX:/workspace \
  -v /root/models:/models \
  -v "$RUNDIR":/run_logs \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash "$IMG" -lc "
set -x
{
  export HF_HUB_CACHE=/root/.cache/huggingface HF_HOME=/root/.cache/huggingface
  export OPENAI_API_KEY=EMPTY
  cd /workspace/benchmarks
  source ./benchmark_lib.sh
  _install_lm_eval_deps
  _patch_lm_eval
  mkdir -p ${RES}
  python3 -m lm_eval --model ${LM_MODEL} ${CHAT_FLAG} \
    --tasks /workspace/utils/evals/gsm8k.yaml \
    --output_path ${RES} --log_samples --limit ${LIMIT} \
    --model_args \"model=/models/Kimi-K2.5-MXFP4,base_url=http://127.0.0.1:10001${EP_PATH},api_key=EMPTY,num_concurrent=16,timeout=1800,max_retries=3,tokenized_requests=False,max_length=16384,trust_remote_code=True,tokenizer=/models/Kimi-K2.5-MXFP4\" \
    --gen_kwargs \"max_tokens=4096,temperature=0,top_p=1\"
  echo GSM8K_${MODE}_DONE_rc=\$?
} > /run_logs/gsm8k_${MODE}.log 2>&1
"
echo "launched $NAME -> /run_logs/gsm8k_${MODE}.log"
