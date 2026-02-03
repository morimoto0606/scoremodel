#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspaces/scoremodel"
EDM="$ROOT/upstream/edm"

# --- Patch 1: CPU-friendly distributed backend (nccl -> gloo when no CUDA)
python - <<'PY'
from pathlib import Path
import re

p = Path("/workspaces/scoremodel/upstream/edm/torch_utils/distributed.py")
s = p.read_text()

# backend selection: 'gloo' if no cuda, else 'nccl' (and keep windows rule)
s = re.sub(
    r"backend\s*=\s*'gloo'\s*if\s*os\.name\s*==\s*'nt'\s*else\s*'nccl'",
    "backend = 'nccl' if (torch.cuda.is_available() and os.name != 'nt') else 'gloo'",
    s
)

# guard cuda.set_device (only when cuda available)
# replace the direct call with guarded block (idempotent-ish)
if "torch.cuda.set_device" in s and "if torch.cuda.is_available()" not in s:
    s = s.replace(
        "torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))",
        "if torch.cuda.is_available():\n        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))"
    )

p.write_text(s)
print("[patch_edm_cpu] patched distributed.py")
PY

# --- Patch 2: generate.py default device auto cuda/cpu
python - <<'PY'
from pathlib import Path
import re

p = Path("/workspaces/scoremodel/upstream/edm/generate.py")
s = p.read_text()

s2 = re.sub(
    r"device=torch\.device\('cuda'\)",
    "device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))",
    s
)

p.write_text(s2)
print("[patch_edm_cpu] patched generate.py")
PY

echo "[patch_edm_cpu] done"