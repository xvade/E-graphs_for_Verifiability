#!/bin/bash
# Recreate the alpha-beta-CROWN tree this project runs on, and apply our local edits.  Idempotent.
#   NNs/verifier_patches/apply.sh [<target dir, default: <repo>/alpha-beta-CROWN>]
# Upstream pins (verified identical to our copy on 2026-09-12 except for the patch below):
#   alpha-beta-CROWN e5c7e17bf0488843acb77b7519f59876717a49f4  (abcrown 0.7.0, "June 2026 release" + pillow downgrade)
#   auto_LiRPA       5a098e8f9fb5786a428a024981d833d303921f2d  (auto_LiRPA 0.7.2, the submodule pin of that commit)
# Local edits:
#   auto_LiRPA_softmax_gradsafe.patch — gradient-safe denominators in _softmax_lse_lower/_softmax_lse_upper
#     (the unselected torch.where branch is 0/0 when diff == 0: forward is masked, backward gives 0*NaN = NaN, which
#     kills every gauge learner that back-propagates through CROWN's softmax bounds; forward values unchanged).
# The exp_configs/*.yaml this project added are tracked under NNs/ (see git ls-files | grep exp_configs).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; T="${1:-$REPO/alpha-beta-CROWN}"
if [ ! -d "$T/.git" ] && [ ! -f "$T/pyproject.toml" ]; then
  git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git "$T"; (cd "$T" && git checkout -q e5c7e17bf0488843acb77b7519f59876717a49f4 && git submodule update --init --recursive)
fi
cd "$T"
# auto_LiRPA is installed by uv from the submodule; our copy has it vendored at complete_verifier/auto_LiRPA (a plain download).
# Apply to whichever layout is present.
for f in complete_verifier/auto_LiRPA/operators/softmax.py auto_LiRPA/auto_LiRPA/operators/softmax.py; do
  [ -f "$f" ] || continue
  if grep -q "safe_diff" "$f"; then echo "already patched: $f"; continue; fi
  sed "s|complete_verifier/auto_LiRPA/operators/softmax.py|$f|g" "$HERE/auto_LiRPA_softmax_gradsafe.patch" | patch -p1 --forward && echo "patched: $f"
done
echo "now: UV_CACHE_DIR=<scratch> uv sync   (torch 2.11+cu130 per uv.lock; check the target driver supports CUDA 13 first)"
