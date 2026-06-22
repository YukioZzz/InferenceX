#!/usr/bin/env bash
# Patch mooncake HIP transport to SKIP host/managed memory in registerLocalMemory
# (it currently returns -1 "Unsupported memory type", failing the whole multi-
# transport registration of the store's host staging buffer). The cross-node
# RDMA/TCP transport registers host memory fine via ibv_reg_mr. Then rebuild +
# install + re-extract the .so.
set -x
{
  F=/root/Mooncake/mooncake-transfer-engine/src/transport/hip_transport/hip_transport.cpp
  python3 - "$F" <<'PY'
import sys
f=sys.argv[1]; s=open(f).read()
if "[MCPATCH host-skip]" in s:
    print("[patch] already applied"); sys.exit(0)
old='''        if (attr.type != hipMemoryTypeDevice) {
            LOG(ERROR) << "Unsupported memory type, " << addr << " "
                       << attr.type;
            return -1;
        }'''
new='''        if (attr.type != hipMemoryTypeDevice) {
            // [MCPATCH host-skip] host/managed memory is served by the cross-node
            // RDMA/TCP transport; skip it in the intra-node HIP transport instead
            // of failing the whole multi-transport local-memory registration.
            return 0;
        }'''
n=s.count(old)
print("[patch] occurrences:", n)
if n==0:
    print("[patch] WARN: pattern not found"); sys.exit(2)
open(f,"w").write(s.replace(old,new))
print("[patch] applied to", n, "site(s)")
PY
  echo "=== rebuild (incremental) ==="
  cd /root/Mooncake/build || exit 2
  make -j"$(nproc)" 2>&1 | tail -15
  echo "=== install ==="
  make install 2>&1 | tail -6
  echo "=== verify patch string in .so + dmabuf still present ==="
  INST=$(python3 -c "import mooncake,inspect,os;print(os.path.dirname(inspect.getfile(mooncake)))")
  strings "$INST"/store*.so "$INST"/engine*.so 2>/dev/null | grep -aiE "MOONCAKE_DISABLE_HIP_DMABUF" | head -1 && echo "dmabuf-ok"
  python3 -c "from mooncake.store import MooncakeDistributedStore; from mooncake.engine import TransferEngine; print('import OK')"
  echo PATCH_REBUILD_DONE
} 2>&1
