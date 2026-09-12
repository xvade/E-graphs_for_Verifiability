#!/bin/bash
# Two-word paired eval only (the gauge gauges/deept_small6_2w_seed0.pt is learned and committed; the learn+eval chain job was pre-empted
# twice and restarted from the top each time).  Resumes weight sets already saved in the JSON.  eps grid = the one stored in the JSON.
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)}"; [ -d "$REPO/NNs" ] || REPO="$(git rev-parse --show-toplevel 2>/dev/null)"; [ -d "$REPO/NNs" ] || { echo "set REPO=<checkout>" >&2; exit 1; }; S="$REPO/NNs/vit_rewrite/_scratch"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"
T="$REPO/NNs/transformer_rewrite"; export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"
mark() { echo "$1 twoword_evalresume $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
mark START; "$PY" -u deept_gauge.py eval --name sst_bert_small_6 --k_words 2 --pairs_per_sent 7 --gauge "gauges/deept_small6_2w_seed0.pt,gauges/deept_small6_seed0.pt" --max_len 12 --n_sent 40 --seed 0 --eps_list "0.0047,0.0094,0.0141" --hi 0.1 --iters 10 --save_json "$T/results/deept_small6_2w_eval_short_seed0.json" >> "$S/deept_eval_short_small6_2w_resume.log" 2>&1; RC=$?; mark "DONE rc=$RC"; exit $RC
