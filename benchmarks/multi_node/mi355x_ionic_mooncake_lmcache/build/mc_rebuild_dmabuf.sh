#!/usr/bin/env bash
# Rebuild Mooncake with USE_HIP_DMABUF + reinstall, so RDMA GPU registration uses
# ibv_reg_dmabuf_mr (dmabuf GPUDirect) instead of legacy ibv_reg_mr (-600 on ionic).
set -x
{
  export PATH=/opt/rocm/bin:$PATH
  cd /root/Mooncake/build || exit 2
  echo "=== reconfigure: USE_HIP_DMABUF GLOBALLY (rdma_context.cpp is in the"
  echo "    rdma_transport OBJECT lib; transfer_engine's PRIVATE define doesn't reach it) ==="
  cmake .. -DUSE_HIP=ON -DUSE_CUDA=OFF -DUSE_HIP_DMABUF=ON -DUSE_ETCD=OFF \
        -DWITH_STORE=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-DUSE_HIP_DMABUF -I/opt/rocm/include -I/opt/rocm/include/hsa" 2>&1 \
        | grep -iE "dmabuf|hsa-runtime|HIP " 
  echo "=== force-recompile rdma_transport objects (stale .o lacks the define) ==="
  find . -path '*rdma_transport*' -name '*.cpp.o' -delete 2>/dev/null
  find . -name 'rdma_context.cpp.o' -delete 2>/dev/null
  echo "=== make (transfer_engine + store) ==="
  make -j"$(nproc)" 2>&1 | tail -40
  echo "=== make install ==="
  make install 2>&1 | tail -12
  echo "=== verify dmabuf compiled into installed .so ==="
  INST=$(python3 -c "import mooncake,inspect,os;print(os.path.dirname(inspect.getfile(mooncake)))")
  echo "installed: $INST"
  strings "$INST"/engine*.so "$INST"/store*.so 2>/dev/null | grep -aiE "dmabuf|CONFIG_PCI_P2PDMA|reg_dmabuf|HIP dmabuf" | sort -u | head
  python3 -c "from mooncake.engine import TransferEngine; print('TransferEngine import OK')"
  echo REBUILD_DONE
} 2>&1
