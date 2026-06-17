#!/usr/bin/env bash
# Build Mooncake from source for ROCm (USE_HIP=ON) on top of the lmcache image.
set -uo pipefail
IMG="kimi-lmcache-rocm:latest"
RUNDIR=/root/run_logs
mkdir -p "$RUNDIR"
docker rm -f mc-build >/dev/null 2>&1 || true

docker run -d --name mc-build --network host \
  -v "$RUNDIR":/run_logs \
  --entrypoint bash "$IMG" -lc '
set -x
{
  echo "=== env ==="; which hipcc hipify-perl cmake; ls /opt/rocm/bin/hipify-perl 2>/dev/null
  export PATH=/opt/rocm/bin:$PATH
  command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }
  cd /root && rm -rf Mooncake
  git clone --depth 1 https://github.com/kvcache-ai/Mooncake.git
  cd Mooncake
  echo "=== git describe ==="; git -C . log --oneline -1
  echo "=== dependencies.sh ==="
  if [ -f dependencies.sh ]; then bash dependencies.sh -y 2>&1 | tail -25 || bash dependencies.sh 2>&1 | tail -25; fi
  echo "=== cmake USE_HIP ==="
  mkdir -p build && cd build
  cmake .. -DUSE_HIP=ON -DUSE_CUDA=OFF -DUSE_ETCD=OFF -DWITH_STORE=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -30
  echo "=== make ==="
  make -j"$(nproc)" 2>&1 | tail -40
  echo "=== make install ==="
  make install 2>&1 | tail -10
  echo "=== pip install python pkg ==="
  cd /root/Mooncake
  python3 -m pip install . --no-build-isolation 2>&1 | tail -10 || pip install ./mooncake-wheel/dist/*.whl 2>&1 | tail -10 || true
  echo "=== verify ==="
  PATH=/opt/rocm/bin:$PATH python3 -c "from mooncake.engine import TransferEngine; print(\"TransferEngine HIP import OK\")" 2>&1 | tail -3
  echo MC_BUILD_DONE
} > /run_logs/mc_build.log 2>&1
'
echo "mooncake build container launched"
sleep 5
docker ps --filter name=mc-build --format "{{.Names}} {{.Status}}"
tail -n 8 "$RUNDIR/mc_build.log" 2>/dev/null | tr '\r' '\n'
