#!/bin/bash
# CPU smoke: exactness gate with a RANDOM gauge, stock eval on the benchmark, and 3 debug learner steps.
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3; cd "$REPO/NNs/vit_rewrite"
"$PY" - <<'PYEOF'
import torch, sys; sys.argv=["x"]; sys.path.insert(0, "."); import genbab_gauge as g
net, sd, attn, H, dh, L = g.load_model("vit_2_3"); ids, boxes = g.bench_boxes("vit_2_3")
torch.manual_seed(0); Gq = torch.eye(dh, dtype=torch.float64).expand(L, H, dh, dh) + 0.3 * torch.randn(L, H, dh, dh, dtype=torch.float64); Ga = torch.eye(dh, dtype=torch.float64).expand(L, H, dh, dh) + 0.3 * torch.randn(L, H, dh, dh, dtype=torch.float64)
print(f"# RANDOM-gauge fp64 gate over 16 benchmark boxes: {g.fp64_gate('vit_2_3', Gq, Ga, boxes[:16]):.2e} (cond {torch.linalg.cond(Gq).max():.1f})")
I = torch.eye(dh, dtype=torch.float64).expand(L, H, dh, dh).clone(); print(f"# identity-gauge gate: {g.fp64_gate('vit_2_3', I, I, boxes[:4]):.2e}")
PYEOF
"$PY" genbab_gauge.py eval --name vit_2_3 --batch 36
"$PY" genbab_gauge.py learn --name vit_2_3 --steps 3 --batch 8 --n_train 64 --log_every 1 --debug 1 --out "$S/genbab_smoke_gauge.pt"
echo SMOKE_DONE
