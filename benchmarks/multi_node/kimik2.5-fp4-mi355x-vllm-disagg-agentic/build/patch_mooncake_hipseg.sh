#!/usr/bin/env bash
# Patch HipTransport::install() so it does NOT overwrite the node's LOCAL segment
# descriptor with protocol="hip". In a USE_HIP build the engine installs rdma
# FIRST (segment protocol "rdma") then hip, whose install() clobbers the local
# segment to "hip" -> remote peers see a hip descriptor for a host (cpu) buffer
# with empty shm_name -> "Corrupted segment descriptor" -> cross-node open fails.
# We keep the HIP transport object (for any intra-node use) but stop it from
# publishing/overwriting the local segment, leaving rdma's segment intact.
# Then rebuild + extract the .so into dmabuf_so (bind-mounted by both engines).
set -x
F=/root/Mooncake/mooncake-transfer-engine/src/transport/hip_transport/hip_transport.cpp
python3 - "$F" <<'PY'
import sys
f=sys.argv[1]; s=open(f).read()
if "[MCPATCH hipseg-skip]" in s:
    print("[patch] hipseg already applied"); sys.exit(0)
old='''    desc->name = local_server_name_;
    desc->protocol = "hip";

    metadata_->addLocalSegment(LOCAL_SEGMENT_ID, local_server_name_,
                               std::move(desc));
    return 0;
}'''
new='''    desc->name = local_server_name_;
    desc->protocol = "hip";

    // [MCPATCH hipseg-skip] Do NOT overwrite the node's local segment with a
    // "hip" descriptor. rdma transport (installed first) already published the
    // segment as protocol "rdma"; clobbering it to "hip" makes remote peers
    // fail to open our host (cpu) segment (empty shm_name -> corrupted desc).
    // We only use rdma for cross-node store transfers, so skip this.
    (void)desc;
    return 0;

    metadata_->addLocalSegment(LOCAL_SEGMENT_ID, local_server_name_,
                               std::move(desc));
    return 0;
}'''
n=s.count(old)
print("[patch] occurrences:", n)
if n!=1:
    print("[patch] WARN: expected 1 occurrence, got", n); sys.exit(2)
open(f,"w").write(s.replace(old,new))
print("[patch] hipseg applied")
PY
echo "=== rebuild (incremental) ==="
cd /root/Mooncake/build || exit 2
make -j"$(nproc)" 2>&1 | tail -12
echo "=== copy fresh .so to /deploy/dmabuf_so (bind-mounted) ==="
INST=$(python3 -c "import mooncake,inspect,os;print(os.path.dirname(inspect.getfile(mooncake)))")
echo "INST=$INST"
make install 2>&1 | tail -4
mkdir -p /deploy/dmabuf_so
cp -v "$INST"/engine.cpython-312-x86_64-linux-gnu.so /deploy/dmabuf_so/ 2>&1
cp -v "$INST"/store.cpython-312-x86_64-linux-gnu.so /deploy/dmabuf_so/ 2>&1
echo "=== verify patch baked into .so ==="
strings /deploy/dmabuf_so/engine*.so | grep -a "MCPATCH hipseg-skip" | head -1 && echo "HIPSEG_PATCH_OK"
python3 -c "from mooncake.store import MooncakeDistributedStore; from mooncake.engine import TransferEngine; print('import OK')"
echo HIPSEG_REBUILD_DONE
