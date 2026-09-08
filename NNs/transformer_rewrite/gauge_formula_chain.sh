#!/bin/bash
# Formula search for the attention gauge (gauge_formula.py validate): scores gauges of known CROWN quality + weight-only candidates
# on sst_bert_small_6.   gauge_formula_chain.sh smoke | full
set -u
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
W=$1; G="gauges"   # relative: the repo path contains spaces and $ARGS is word-split
ALL="$G/deept_small6_seed0.pt,$G/deept_small6_oodyelp_seed0.pt,$G/deept_small6_oodrandom_seed0.pt,$G/deept_small6_2w_seed0.pt,$G/pbvtrained_small6_inner_seed0.pt,$G/pbvtrained_small6_origin_seed0.pt"
mark() { echo "$1 formula_$W $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
case $W in
  smoke) ARGS="--n_sent 2 --pos 1 --n_sst 4 --l1_steps 20 --hi 0.05 --gauges $G/deept_small6_seed0.pt,$G/pbvtrained_small6_origin_seed0.pt"; OUT="$T/results/formula_small6_smoke.json" ;;
  full)  ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 400 --hi 0.1 --gauges $ALL"; OUT="$T/results/formula_small6_validate.json" ;;
  big3)  NAME=sst_bert_big_3;   ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 400 --hi 0.1 --gauges $G/deept_big3_seed0.pt";  OUT="$T/results/formula_big3_validate.json" ;;
  yelp3) NAME=yelp_bert_small_3; ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 400 --hi 0.1 --gauges $G/deept_yelp3_seed0.pt"; OUT="$T/results/formula_yelp3_validate.json" ;;
  yelp3_big) NAME=yelp_bert_small_3; ARGS="--n_sent 48 --pos 2 --n_sst 48 --l1_steps 100 --hi 0.1 --gauges $G/deept_yelp3_seed0.pt"; OUT="$T/results/formula_yelp3_validate_96box.json" ;;   # 4x the probes: does averaging remove the probe-seed dependence?
  # ---- improvement goal (2026-09-07 eve): localisation hybrids, probe-seed cross, unlabeled dev-text probes, held-out radius screen
  hyb_yelp3)  NAME=yelp_bert_small_3; ARGS="--n_sent 48 --pos 2 --n_sst 48 --l1_steps 0 --hi 0.1 --hybrids 1 --cross 1 --radius_names auto --n_dev 24 --dev_probes 48 --tag _h --gauges $G/deept_yelp3_seed0.pt";  OUT="$T/results/formula_yelp3_hyb.json" ;;
  hyb_small6) NAME=sst_bert_small_6;  ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 0 --hi 0.1 --hybrids 1 --cross 0 --radius_names auto --n_dev 24 --dev_probes 24 --tag _h --gauges $G/deept_small6_seed0.pt"; OUT="$T/results/formula_small6_hyb.json" ;;
  hyb_big3)   NAME=sst_bert_big_3;    ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 0 --hi 0.1 --hybrids 1 --cross 0 --radius_names auto --n_dev 24 --dev_probes 24 --tag _h --gauges $G/deept_big3_seed0.pt";   OUT="$T/results/formula_big3_hyb.json" ;;
  smoke_hyb)  NAME=yelp_bert_small_3; ARGS="--n_sent 2 --pos 1 --n_sst 4 --l1_steps 0 --hi 0.05 --hybrids 1 --cross 1 --radius_names auto --n_dev 2 --dev_probes 2 --tag _smoke --gauges $G/deept_yelp3_seed0.pt"; OUT="$T/results/formula_hyb_smoke.json" ;;
  # ---- round 2: sensitivity-weighted token blocks (--sens) and CROWN-inflation rescaling of M (--infl), held-out radius screen
  r2_yelp3)  NAME=yelp_bert_small_3; ARGS="--n_sent 48 --pos 2 --n_sst 48 --l1_steps 0 --hi 0.1 --sens 1 --infl 3 --radius_names auto --n_dev 24 --tag _r2 --gauges $G/deept_yelp3_seed0.pt";  OUT="$T/results/formula_yelp3_r2.json" ;;
  r2_small6) NAME=sst_bert_small_6;  ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 0 --hi 0.1 --sens 1 --infl 3 --radius_names auto --n_dev 24 --tag _r2 --gauges $G/deept_small6_seed0.pt"; OUT="$T/results/formula_small6_r2.json" ;;
  r2_big3)   NAME=sst_bert_big_3;    ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 0 --hi 0.1 --sens 1 --infl 3 --radius_names auto --n_dev 24 --tag _r2 --gauges $G/deept_big3_seed0.pt";   OUT="$T/results/formula_big3_r2.json" ;;
  smoke_r2)  NAME=yelp_bert_small_3; ARGS="--n_sent 2 --pos 1 --n_sst 4 --l1_steps 0 --hi 0.05 --sens 1 --infl 2 --radius_names auto --n_dev 2 --tag _smoke --gauges $G/deept_yelp3_seed0.pt"; OUT="$T/results/formula_r2_smoke.json" ;;
  # ---- round 3: the layer-0 width product is an l1 problem (one token block) -> l1 surrogate with N from the closed form (+ splices), held-out screen on SST test
  r3_yelp3)  NAME=yelp_bert_small_3; ARGS="--n_sent 48 --pos 2 --n_sst 48 --l1_steps 400 --l1N 1 --infl 1 --hi 0.1 --radius_names auto --n_dev 24 --tag _r3 --gauges $G/deept_yelp3_seed0.pt";  OUT="$T/results/formula_yelp3_r3.json" ;;
  r3_big3)   NAME=sst_bert_big_3;    ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 400 --l1N 1 --infl 1 --hi 0.1 --radius_names auto --n_dev 24 --dev_split test --tag _r3 --gauges $G/deept_big3_seed0.pt";   OUT="$T/results/formula_big3_r3.json" ;;
  r3_small6) NAME=sst_bert_small_6;  ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 400 --l1N 1 --infl 1 --hi 0.1 --radius_names auto --n_dev 24 --dev_split test --tag _r3 --gauges $G/deept_small6_seed0.pt"; OUT="$T/results/formula_small6_r3.json" ;;
  smoke_r3)  NAME=sst_bert_big_3;    ARGS="--n_sent 2 --pos 1 --n_sst 4 --l1_steps 20 --l1N 1 --infl 1 --hi 0.05 --radius_names auto --n_dev 2 --dev_split test --tag _smoke --gauges $G/deept_big3_seed0.pt"; OUT="$T/results/formula_r3_smoke.json" ;;
  # ---- round 4: longer, annealed l1N optimisation (1500 steps, cosine lr); explicit radius-screen list
  r4_yelp3)  NAME=yelp_bert_small_3; ARGS="--n_sent 48 --pos 2 --n_sst 48 --l1_steps 1500 --l1N 1 --hi 0.1 --radius_names identity,yelp3,cand:svd_jacN_all,cand:l1N,cand:l1N@L0 --n_dev 24 --tag _r4 --gauges $G/deept_yelp3_seed0.pt";  OUT="$T/results/formula_yelp3_r4.json" ;;
  r4_big3)   NAME=sst_bert_big_3;    ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 1500 --l1N 1 --hi 0.1 --radius_names identity,big3,cand:svd_jacN_all,cand:l1N,cand:l1N@L0 --n_dev 24 --dev_split test --tag _r4 --gauges $G/deept_big3_seed0.pt";   OUT="$T/results/formula_big3_r4.json" ;;
  r4_small6) NAME=sst_bert_small_6;  ARGS="--n_sent 12 --pos 2 --n_sst 48 --l1_steps 1500 --l1N 1 --hi 0.1 --radius_names identity,small6,cand:svd_jacN_all,cand:l1N,cand:l1N@L0 --n_dev 24 --dev_split test --tag _r4 --gauges $G/deept_small6_seed0.pt"; OUT="$T/results/formula_small6_r4.json" ;;
esac
mark START; "$PY" -u gauge_formula.py validate --name ${NAME:-sst_bert_small_6} $ARGS --seed 0 --out "$OUT" >> "$S/formula_$W.log" 2>&1; RC=$?; mark "DONE rc=$RC"; exit $RC
