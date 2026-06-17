#!/usr/bin/env bash
set -uo pipefail
docker exec kimi-mclmc-prefill bash -lc '
python3 - <<PY
import vllm, importlib.util as u
print("vllm_version", vllm.__version__)
for m in ["vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
          "vllm.v1.kv_offload",
          "lmcache.integration.vllm.lmcache_mp_connector"]:
    print(m, "PRESENT" if u.find_spec(m) else "MISSING")
PY
' 2>&1 | tail -8
