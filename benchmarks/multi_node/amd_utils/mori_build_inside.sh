#!/usr/bin/env bash
# Build ROCm/mori #341 (IOEngine.wait_all) into the current engine container.
# This runs only while preparing a derived image; serving never compiles mori.
set -euo pipefail

SRC=${SRC:-/mori-src}
MORI_VERSION=${MORI_VERSION:-1.0.1+mori341.f7e6ac68}

echo "[mori-build] staging source off shared storage"
rm -rf /tmp/mori
cp -a "$SRC" /tmp/mori
cd /tmp/mori
git config --global --add safe.directory '*' || true
echo "[mori-build] HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)"

# Match SGLang's BUILD_UMBP=ON build instead of disabling a subsystem.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libpci-dev libgrpc++-dev protobuf-compiler-grpc
test -s /usr/include/pci/pci.h
test -s /usr/include/grpcpp/grpcpp.h

python3 -m pip install -q --break-system-packages -r requirements-build.txt
export SETUPTOOLS_SCM_PRETEND_VERSION="$MORI_VERSION"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_MORI="$MORI_VERSION"
export MORI_GPU_ARCHS=${MORI_GPU_ARCHS:-gfx950}
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

echo "[mori-build] building arch=$MORI_GPU_ARCHS jobs=$CMAKE_BUILD_PARALLEL_LEVEL"
python3 -m pip install --break-system-packages --no-build-isolation \
    --force-reinstall .

cd /
python3 - <<'PY'
import mori
import mori.io as io

print("[mori-build] mori=", getattr(mori, "__version__", "?"), mori.__file__)
print("[mori-build] wait_all=", hasattr(io.IOEngine, "wait_all"))
assert hasattr(io.IOEngine, "wait_all")
assert hasattr(io, "StatusCode")
PY
echo "MORI_BUILD_OK"
