#!/bin/bash
# learner warm-started from the manual unified rule (closed form + l1 QK all layers + l1 AV layer 0), single or per-label
#   sbatch learn_init.sbatch <small6|big3> <label: -1 (all) | 0 | 1>
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"; M="$1"; L="$2"
case $M in
  small6) NAME=sst_bert_small_6; NS=60; INIT=gauges/formula_sst_bert_small_6_l1N_qk_av0_r4.pt ;;
  big3)   NAME=sst_bert_big_3;   NS=40; INIT=gauges/formula_sst_bert_big_3_l1N_qk_av0_r3.pt ;;
esac
SUF=$([ "$L" -ge 0 ] && echo "lab$L" || echo ""); TAG="${M}_init${SUF}"
mark() { echo "$1 deept_${TAG}_learn $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
mark START
"$PY" -u deept_gauge.py learn --name $NAME --split dev --max_len 8 --n_sent $NS --pos_per_sent 3 --steps 120 --accum 4 --lr 0.01 --cond_pen 0.0001 --clip 1.0 --hi 0.1 --radius_iters 8 --n_eval 48 --seed 0 --log_every 10 --label $L --gauge "$INIT" --out "$T/gauges/deept_${TAG}_seed0.pt" > "$S/deept_learn_${TAG}.log" 2>&1; RC=$?
mark "DONE rc=$RC"; exit $RC
