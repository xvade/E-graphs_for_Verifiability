#!/bin/bash
# signed-rule variants (CPU): rotation-only (sgn_rot, sgn_rot_qk; tag _rot_lab<y>) and strong cond penalty 1e-2 (sgn, sgn_qk; tag _c2_lab<y>)
#   sbatch sgnopt_var.sbatch <small6|big3> <label> <rot|c2>
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""; cd "$T"; M="$1"; Y="$2"; V="$3"
case $M in small6) NAME=sst_bert_small_6; INIT=gauges/formula_sst_bert_small_6_l1N_qk_av0_r4.pt ;; big3) NAME=sst_bert_big_3; INIT=gauges/formula_sst_bert_big_3_l1N_qk_av0_r3.pt ;; esac
case $V in mix) VAR="--signed_variants sgn_mix,sgn_mix_qk"; TAG="_mix_lab$Y" ;; rot) VAR="--signed_variants sgn_rot,sgn_rot_qk"; TAG="_rot_lab$Y" ;; c2) VAR="--signed_variants sgn,sgn_qk --signed_cond 0.01"; TAG="_c2_lab$Y" ;; esac
mark() { echo "$1 sgnopt_${M}_${V}_lab$Y $(date) job=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0}" >> "$S/official_sequence.log"; }
mark START; [ -f results/formula_${M}_sgnopt${TAG}.json ] && { mark "already done, skipped"; exit 0; }
"$PY" -u gauge_formula.py validate --name $NAME --seed 0 --n_sent 24 --pos 2 --max_len 8 --cost_only 1 --signed 1 --signed_steps 400 --signed_init $INIT $VAR --l1_steps 0 --probe_label $Y --gauges gauges/deept_${M}_seed0.pt,gauges/deept_${M}_lab${Y}_seed0.pt --tag $TAG --out results/formula_${M}_sgnopt${TAG}.json > "$S/formula_${M}_sgnopt${TAG}.log" 2>&1; RC=$?
mark "DONE rc=$RC"; exit $RC
