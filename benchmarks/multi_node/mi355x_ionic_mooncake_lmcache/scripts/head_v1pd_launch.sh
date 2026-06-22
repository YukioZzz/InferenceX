#!/usr/bin/env bash
echo "===== g06: tear down old pd; launch mooncake master (0.0.0.0:50051) + prefiller (LMCacheV1, :dmabuf) ====="
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 mia1-p01-g06 'bash -s' <<'EOF'
docker rm -f kimi-pd-prefill kimi-pd-proxy kimi-mc-serve kimi-mc-master >/dev/null 2>&1
export RUNDIR=/home/thshan@amd.com/mc_lmc/run_v1pd
mkdir -p "$RUNDIR"
IMG=kimi-lmc-mc-rocm:dmabuf RUNDIR=$RUNDIR bash /home/thshan@amd.com/mc_lmc/g05_mc_rdma_lmc.sh master 2>&1 | tail -3
  EXTRA_ENV='export MC_FORCE_TCP=1; export MC_MAX_MR_SIZE=137438953472' IMG=kimi-lmc-mc-rocm:dmabuf MOUNT_SO=1 RUNDIR=$RUNDIR CONC=128 CFG=mooncake-rdma-sa-pref.yaml HOSTIP=10.24.112.182 \
  bash /home/thshan@amd.com/mc_lmc/rdma_engine.sh prefiller 2>&1 | tail -3
EOF
echo "===== g05: launch decoder (LMCacheV1, :latest+.so) -> g06 master ====="
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 mia1-p01-g05 'bash -s' <<'EOF'
docker rm -f kimi-pd-decode kimi-mc-decoder >/dev/null 2>&1
export RUNDIR=/home/thshan@amd.com/mc_lmc/run_v1pddec
mkdir -p "$RUNDIR"
  EXTRA_ENV='export MC_FORCE_TCP=1; export MC_MAX_MR_SIZE=137438953472' IMG=kimi-lmc-mc-rocm:latest MOUNT_SO=1 RUNDIR=$RUNDIR CONC=128 CFG=mooncake-rdma-sa-dec.yaml HOSTIP=10.24.112.181 \
  bash /home/thshan@amd.com/mc_lmc/rdma_engine.sh decoder 2>&1 | tail -3
EOF
