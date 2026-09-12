#!/bin/bash
# sign-aware per-label rule: candidate_signed (400 Adam steps, CPU) from the unified rule on label-y random-token probes -> gauges/formula_<name>_sgn{,_qk}_lab<y>.pt
# skip-if-done (both gauge files exist); --requeue on ckpt.   sbatch sgnopt.sbatch <small6|big3> <label>
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""; cd "$T"; M="$1"; Y="$2"
case $M in small6) NAME=sst_bert_small_6; INIT=gauges/formula_sst_bert_small_6_l1N_qk_av0_r4.pt ;; big3) NAME=sst_bert_big_3; INIT=gauges/formula_sst_bert_big_3_l1N_qk_av0_r3.pt ;; esac
mark() { echo "$1 sgnopt_${M}_lab$Y $(date) job=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0}" >> "$S/official_sequence.log"; }
mark START; if [ -f gauges/formula_${NAME}_sgn_lab$Y.pt ] && [ -f gauges/formula_${NAME}_sgn_qk_lab$Y.pt ]; then mark "already done, skipped"; exit 0; fi
"$PY" -u gauge_formula.py validate --name $NAME --seed 0 --n_sent 24 --pos 2 --max_len 8 --cost_only 1 --signed 1 --signed_steps 400 --signed_init $INIT --l1_steps 0 --probe_label $Y --gauges gauges/deept_${M}_seed0.pt,gauges/deept_${M}_lab${Y}_seed0.pt --tag _lab$Y --out results/formula_${M}_sgnopt_lab$Y.json > "$S/formula_${M}_sgnopt_lab$Y.log" 2>&1; RC=$?
mark "DONE rc=$RC"; exit $RC
