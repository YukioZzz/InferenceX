#!/usr/bin/env bash
# Build LMCache from source for ROCm (gfx950) inside the aigmkt image, then commit a new image.
set -uo pipefail
IMG="aigmkt/vllm-openai-rocm:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036"
RUNDIR=/root/run_logs
mkdir -p "$RUNDIR"
docker rm -f lmc-build >/dev/null 2>&1 || true

docker run -d --name lmc-build --network host \
  -v "$RUNDIR":/run_logs \
  --entrypoint bash "$IMG" -lc '
set -x
{
  command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }
  cd /root && rm -rf lmcache-src
  git clone --depth 1 https://github.com/LMCache/LMCache.git lmcache-src
  cd lmcache-src
  echo "=== lmcache source version ==="; cat lmcache/_version.py 2>/dev/null || git -C . describe --tags 2>/dev/null || true
  python3 -m pip uninstall -y lmcache >/dev/null 2>&1 || true
  python3 -m pip install -r requirements/build.txt
  PYTORCH_ROCM_ARCH=gfx950 TORCH_DONT_CHECK_COMPILER_ABI=1 CXX=hipcc BUILD_WITH_HIP=1 \
    python3 -m pip install -e . --no-build-isolation
  echo "=== verify ==="
  python3 -c "import lmcache; print(\"lmcache\", getattr(lmcache,\"__version__\",\"?\"))"
  python3 -c "import lmcache.c_ops; print(\"c_ops import OK (HIP)\")"
  python3 -c "import lmcache.integration.vllm.lmcache_mp_connector as m; print(\"LMCacheMPConnector OK\")"
  echo BUILD_DONE_OK
} > /run_logs/lmc_build.log 2>&1
'
echo "build container launched"
sleep 4
docker ps --filter name=lmc-build --format '{{.Names}} {{.Status}}'
tail -n 5 "$RUNDIR/lmc_build.log" 2>/dev/null | tr '\r' '\n'
