# SPDX-License-Identifier: Apache-2.0
# Minimal 1P1D proxy for LMCacheConnectorV1 + shared Mooncake pool.
#
# Mechanism (shared-STORE PD, not P2P): for each request the proxy
#   1. sends the prompt to the PREFILLER with max_tokens=1 (stream=False) and
#      AWAITS completion -> prefiller computes prompt KV and stores it into the
#      shared Mooncake pool;
#   2. forwards the ORIGINAL request to the DECODER, which retrieves the prompt
#      KV from the pool (cross-node over Mooncake TCP) and only computes the
#      decode steps.
# No telemetry wait: LMCacheConnectorV1 (unlike the MP connector) does not emit
# request_store_finished. Correctness never depends on the handoff (a decoder
# miss just recomputes locally); the agentic shared-prefix reuse is what we
# measure across requests.
from contextlib import asynccontextmanager
import argparse
import itertools
import os
import sys
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx


def csv_strs(s):
    return [x.strip() for x in s.split(",")]


def csv_ints(s):
    return [int(x) for x in s.split(",")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--prefiller-host", type=csv_strs, default=["127.0.0.1"])
    p.add_argument("--prefiller-port", type=csv_ints, default=[8000])
    p.add_argument("--decoder-host", type=csv_strs, default=["127.0.0.1"])
    p.add_argument("--decoder-port", type=csv_ints, default=[8000])
    return p.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pf = list(zip(global_args.prefiller_host, global_args.prefiller_port * len(global_args.prefiller_host)))
    df = list(zip(global_args.decoder_host, global_args.decoder_port * len(global_args.decoder_host)))
    app.state.prefill = [httpx.AsyncClient(timeout=None, base_url=f"http://{h}:{p}") for h, p in pf]
    app.state.decode = [httpx.AsyncClient(timeout=None, base_url=f"http://{h}:{p}") for h, p in df]
    yield
    for c in app.state.prefill + app.state.decode:
        await c.aclose()


app = FastAPI(lifespan=lifespan)
_rr = itertools.count()


def _pick():
    i = next(_rr)
    return (app.state.prefill[i % len(app.state.prefill)],
            app.state.decode[i % len(app.state.decode)])


def _headers():
    h = {"Content-Type": "application/json"}
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if k:
        h["Authorization"] = f"Bearer {k}"
    return h


async def _handle(request: Request, endpoint: str):
    try:
        req = await request.json()
        pc, dc = _pick()

        # 1) prefill (max_tokens=1) -> populate Mooncake pool with prompt KV
        pre = dict(req)
        pre["max_tokens"] = 1
        if "max_completion_tokens" in pre:
            pre["max_completion_tokens"] = 1
        pre["stream"] = False
        pre.pop("stream_options", None)
        try:
            r = await pc.post(endpoint, json=pre, headers=_headers())
            r.raise_for_status()
        except Exception as e:  # prefill failure must not abort decode
            print(f"[proxy] prefill warn: {e}", flush=True)

        # 2) decode -> original request (retrieves prompt KV from pool)
        is_stream = bool(req.get("stream", False))
        if is_stream:
            async def gen():
                async with dc.stream("POST", endpoint, json=req, headers=_headers()) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            return StreamingResponse(gen(), media_type="text/event-stream")
        else:
            r = await dc.post(endpoint, json=req, headers=_headers())
            r.raise_for_status()
            return JSONResponse(content=r.json())
    except Exception as e:
        print(f"[proxy] error {endpoint}: {e}", flush=True)
        print("".join(traceback.format_exception(*sys.exc_info())), flush=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle(request, "/v1/completions")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle(request, "/v1/chat/completions")


@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})


@app.get("/v1/models")
async def models():
    try:
        r = await app.state.prefill[0].get("/v1/models", headers=_headers())
        return JSONResponse(content=r.json())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
