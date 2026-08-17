#!/usr/bin/env bash
# Apply the Kimi-K3 MoRIIO connector patch inside the engine container.
#
# Two patches live in patches/ and exactly one is applied per run:
#
#   k3_moriio_51052.patch   vLLM #51052 alone: hybrid mamba/KDA state transfer,
#                           spec decode on the hybrid decode side. This is the
#                           plain 1P1D arm.
#   k3_tpdcp_hetero.patch   a superset: #51052 + the AITER DCP work (#51705) +
#                           the TP-prefill -> DCP-decode relayout. Required by
#                           any arm that sets DECODE_DCP > 1.
#
# Both are generated against the exact vLLM commit this image ships
# (ac7509e2b), so they apply to site-packages with -p1 and no fuzz. Select with
# K3_MORIIO_PATCH; the DCP arm sets it from the matrix.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${K3_MORIIO_PATCH:-}" ]]; then
    PATCH_FILE="$K3_MORIIO_PATCH"
    # A bare filename from the matrix resolves against patches/.
    [[ -f "$PATCH_FILE" ]] || PATCH_FILE="$HERE/patches/$K3_MORIIO_PATCH"
elif [[ "${DECODE_DCP:-1}" -gt 1 ]]; then
    # Asking for DCP without the DCP patch would start an engine that rejects
    # --decode-context-parallel-size, 40 minutes into a weight load. Pick it.
    PATCH_FILE="$HERE/patches/k3_tpdcp_hetero.patch"
    echo "[k3-moriio] DECODE_DCP=${DECODE_DCP} -> selecting the TP-DCP superset patch"
else
    PATCH_FILE="$HERE/patches/k3_moriio_51052.patch"
fi
ROOT="$(python3 -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"

if [[ -z "$ROOT" || ! -d "$ROOT/vllm" ]]; then
    echo "[k3-moriio] ERROR: could not resolve vLLM root (ROOT='$ROOT')" >&2
    exit 1
fi
if [[ ! -f "$PATCH_FILE" ]]; then
    echo "[k3-moriio] ERROR: patch file missing: $PATCH_FILE" >&2
    exit 1
fi
echo "[k3-moriio] patch: $PATCH_FILE"

MORIIO_DIR="$ROOT/vllm/distributed/kv_transfer/kv_connector/v1/moriio"

# What must be true afterwards. The #51052 markers are common to both patches,
# so on a DCP run they are not enough: the 51052-only patch also satisfies them,
# and it would then be reported as "already applied" while the engine has no DCP
# relayout at all. Check for a relayout symbol too when that is what was asked
# for, and fail loudly rather than serve the wrong code.
want_dcp=0
grep -q 'build_dcp_token_pairing' "$PATCH_FILE" && want_dcp=1

have_51052=0
grep -RqsE '_draft_only_layers|as_attn_mamba' "$MORIIO_DIR" && have_51052=1
have_dcp=0
grep -Rqs 'build_dcp_token_pairing' "$MORIIO_DIR" && have_dcp=1

if [[ "$have_51052" == 1 ]]; then
    if [[ "$want_dcp" == 1 && "$have_dcp" == 0 ]]; then
        echo "[k3-moriio] ERROR: #51052 is already applied without the DCP relayout," >&2
        echo "[k3-moriio]        so the DCP superset can no longer be applied cleanly." >&2
        echo "[k3-moriio]        This image was patched by a previous, non-DCP run." >&2
        exit 1
    fi
    echo "[k3-moriio] already applied (dcp=$have_dcp)"
    exit 0
fi

if (cd "$ROOT" && git apply -p1 "$PATCH_FILE"); then
    echo "[k3-moriio] applied with git apply"
elif patch -p1 -d "$ROOT" --forward --no-backup-if-mismatch < "$PATCH_FILE"; then
    echo "[k3-moriio] applied with patch"
else
    echo "[k3-moriio] ERROR: failed to apply $PATCH_FILE" >&2
    exit 1
fi

# Import by name. A bare `import vllm` would pass even if the patch had landed a
# moriio package whose entry points no longer resolve.
if [[ "$want_dcp" == 1 ]]; then
    python3 - <<'PY' || { echo "[k3-moriio] ERROR: DCP relayout not importable after patch" >&2; exit 1; }
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout import (
    build_dcp_block_pairing,
    build_dcp_token_pairing,
    dcp_relayout_granularity,
    validate_moriio_heterogeneous_dcp,
)
print("[k3-moriio] DCP relayout entry points OK")
PY
fi
