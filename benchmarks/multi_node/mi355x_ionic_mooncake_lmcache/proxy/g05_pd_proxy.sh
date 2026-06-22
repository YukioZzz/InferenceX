#!/usr/bin/env bash
# 1P1D proxy on g05: prefiller=g05:8000 (local), decoder=g06:8000. HTTP on :9100.
set -uo pipefail
IMG="${IMG:-kimi-lmc-mc-rocm:latest}"
DEPLOY="${DEPLOY:-/home/thshan@amd.com/mc_lmc}"
RUNDIR="${RUNDIR:-/home/thshan@amd.com/mc_lmc/run}"
P_IP="${P_IP:-127.0.0.1}"; P_PORT="${P_PORT:-8000}"
D_IP="${D_IP:-10.24.112.182}"; D_PORT="${D_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-9100}"
mkdir -p "$RUNDIR"
docker rm -f kimi-mc-pd-proxy >/dev/null 2>&1 || true
docker run -d --name kimi-mc-pd-proxy --network host \
  -v "$DEPLOY":/deploy -v "$RUNDIR":/run_logs \
  --entrypoint bash "$IMG" -lc \
  "python3 -c 'import httpx,fastapi,uvicorn' 2>/dev/null || pip install -q httpx fastapi uvicorn; \
   python3 /deploy/mc_pd_proxy.py --host 0.0.0.0 --port ${PROXY_PORT} \
     --prefiller-host ${P_IP} --prefiller-port ${P_PORT} \
     --decoder-host ${D_IP} --decoder-port ${D_PORT} > /run_logs/mc_pd_proxy.log 2>&1"
echo "launched kimi-mc-pd-proxy (http ${PROXY_PORT}; P=${P_IP}:${P_PORT} D=${D_IP}:${D_PORT})"
sleep 8
docker ps --filter name=kimi-mc-pd-proxy --format '{{.Names}} {{.Status}}'
tail -n 8 "$RUNDIR/mc_pd_proxy.log" 2>/dev/null | tr '\r' '\n'
curl -s -m5 -o /dev/null -w 'proxy_health=%{http_code}\n' http://127.0.0.1:${PROXY_PORT}/health 2>&1 || true
