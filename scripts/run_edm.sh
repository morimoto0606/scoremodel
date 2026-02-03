#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspaces/scoremodel
EDM=$ROOT/upstream/edm
OUT=$ROOT/results/edm/cifar10/steps18_heun_seeds0-63
NET="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"

mkdir -p "$OUT"

# apply CPU patches (keeps submodule clean if you restored first)
"$ROOT/scripts/patch_edm_cpu.sh"

python "$EDM/generate.py" \
  --outdir="$OUT" \
  --seeds=0-63 \
  --batch=8 \
  --steps=18 \
  --solver=heun \
  --network="$NET"

python - <<PY
import json, subprocess, pathlib, time
out = pathlib.Path("$OUT")
meta = {
  "time": time.strftime("%Y-%m-%d %H:%M:%S"),
  "cmd": "generate.py --outdir ... --seeds=0-63 --batch=8 --steps=18 --solver=heun --network ...",
  "git": {
    "scoremodel": subprocess.check_output(["git","-C","$ROOT","rev-parse","HEAD"]).decode().strip(),
    "edm": subprocess.check_output(["git","-C","$EDM","rev-parse","HEAD"]).decode().strip(),
  },
}
(out/"manifest.json").write_text(json.dumps(meta, indent=2))
print("wrote", out/"manifest.json")
PY
