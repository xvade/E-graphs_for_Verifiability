#!/bin/bash
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
cd "$REPO"; export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
echo "== patched learned-G (svd init)"; "$PY" NNs/vit_rewrite/vit_patch_onnx.py --model pgd_2_3_16 --gauge_file NNs/vit_rewrite/gauges/pgd_mix_svdinit.pt --name learnedG_patched 2>&1 | grep "^#"
echo "== patched learned-G (id init)";  "$PY" NNs/vit_rewrite/vit_patch_onnx.py --model pgd_2_3_16 --gauge_file NNs/vit_rewrite/gauges/pgd_mix_idinit.pt --name idinitG_patched 2>&1 | grep "^#"
echo "== patched identity control (should be byte-identical weights)"; "$PY" NNs/vit_rewrite/vit_patch_onnx.py --model pgd_2_3_16 --variant base --name base_patched 2>&1 | grep "^#"
echo "== base export via vit_export.py (export-path control)"; "$PY" NNs/vit_rewrite/vit_export.py --model pgd_2_3_16 --variant base --name base_export 2>&1 | grep "^#"
