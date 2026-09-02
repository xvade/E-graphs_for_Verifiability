#!/bin/bash
# Run a python script in the WORKING taso+onnx environment.
#
# For two days taso and onnx couldn't co-exist in-repo (container had real taso but
# a broken py3.14 onnx; host taso_py had onnx but only a stub taso). Resolved
# 2026-09-02 by building taso's cython ext for taso_py's python 3.10 (reusing the
# existing CPU libtaso_runtime.so, compiled inside the container for ABI match):
#
#   $TP/bin/python3 -m pip install cython            # one-time, on a networked node
#   apptainer exec --no-mount bind-paths -B <toolchain> tensat.sif bash -c '
#     cd taso/python; CC=/usr/bin/gcc CXX=/usr/bin/g++ TASO_LIB_DIR=$PWD/../build \
#       PYTHONPATH=$SP $TP/bin/python3 setup.py build_ext --inplace'
#
# yielding taso/python/taso/core.cpython-310-*.so (gitignored). The env is then:
# taso_py's python 3.10, run INSIDE tensat.sif (for the container's libprotobuf that
# libtaso_runtime links), with the repo taso package + taso_py site-packages on
# PYTHONPATH. See MEMORY: taso-env-working / PROBLEMATIC.md #5.
#
# Usage:  bash NNs/tests/in_taso_env.sh <script.py> [args...]
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"
TP=/mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat/miniconda3/envs/taso_py
SP="$TP/lib/python3.10/site-packages"
EXT="$REPO/taso/python/taso/core.cpython-310-x86_64-linux-gnu.so"

if [ ! -f "$EXT" ]; then
  echo "ERROR: py3.10 taso ext not built ($EXT missing). Build it first (see header)." >&2
  exit 2
fi

# Resolve the script (first arg) to an absolute path before the container cd's away.
SCRIPT="$(realpath "$1")"; shift

exec apptainer exec --no-mount bind-paths -B "$TP" -B /mmfs1/gscratch/scrubbed/sgvtc/toolchain-tensat \
  "$REPO/tensat.sif" bash -c "
    export LD_LIBRARY_PATH='$REPO/taso/build:/opt/conda/lib:\${LD_LIBRARY_PATH:-}'
    export PYTHONPATH='$REPO/taso/python:$REPO/NNs:$SP'
    cd '$REPO/taso/python'
    '$TP/bin/python3' '$SCRIPT' \"\$@\"
  " _ "$@"
