#!/bin/bash
# usage: run_deept_eval_chain.sh <seed> <node> : wait for the learner's DONE marker, then paired eval (short test sentences, then a longer-sentence sample)
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; seed=$1; node=$2
while ! grep -q "^DONE_deept_learn_seed$seed " "$S/official_sequence.log"; do sleep 60; done
G="$REPO/NNs/transformer_rewrite/gauges/deept_small3_seed$seed.pt"; [ -f "$G" ] || { echo "no gauge file $G" >> "$S/official_sequence.log"; exit 1; }
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$REPO/NNs/transformer_rewrite"
echo "START_deept_eval_seed$seed $(date)" >> "$S/official_sequence.log"
srun --jobid=39619518 --overlap -N1 -n1 -w $node --cpus-per-task=4 --gres=gpu:l40s:1 "$PY" -u deept_gauge.py eval --name sst_bert_small_3 --gauge "$G" --max_len 12 --n_sent 40 --seed 0 --eps_list 0.01,0.02,0.03 --hi 0.1 --iters 10 --save_json "$REPO/NNs/transformer_rewrite/results/deept_small3_eval_short_seed$seed.json" > "$S/deept_eval_short_seed$seed.log" 2>&1
srun --jobid=39619518 --overlap -N1 -n1 -w $node --cpus-per-task=4 --gres=gpu:l40s:1 "$PY" -u deept_gauge.py eval --name sst_bert_small_3 --gauge "$G" --max_len 32 --n_sent 12 --seed 7 --eps_list 0.005,0.01,0.02 --hi 0.1 --iters 8 --save_json "$REPO/NNs/transformer_rewrite/results/deept_small3_eval_long_seed$seed.json" > "$S/deept_eval_long_seed$seed.log" 2>&1
echo "DONE_deept_eval_seed$seed $(date)" >> "$S/official_sequence.log"
