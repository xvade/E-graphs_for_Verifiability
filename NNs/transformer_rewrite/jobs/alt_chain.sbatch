#!/bin/bash
# alternating alpha-CROWN / gauge learner on sst_bert_small_6, then the alpha-tier paired eval of the result.
# Restart-safe for the preemptible ckpt partition: the learner resumes from <out>.ckpt (every 5 steps) and is skipped once the
# final gauge (partial=False) exists; the eval resumes per instance from <json>.part_* files and is skipped once the JSON exists.
#   sbatch alt_chain.sbatch smoke | full
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd "$T"; W="$1"
mark() { echo "$1 alt_$W $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case $W in
  smoke) LEARN="--max_len 6 --n_sent 3 --pos_per_sent 2 --steps 2 --accum 2 --alpha_iters 3 --n_eval 2 --log_every 1 --lr 0.005 --ckpt_every 1"; OUT="$T/gauges/deept_small6_alt_smoke.pt"; EVAL="--n_sent 2 --max_len 6 --eps_list 0.02" ; EJ="$T/results/deept_small6_alt_smoke_eval_alpha.json" ;;
  full)  LEARN="--max_len 6 --n_sent 60 --pos_per_sent 3 --steps 60 --accum 4 --alpha_iters 10 --n_eval 16 --log_every 20 --lr 0.005 --ckpt_every 5"; OUT="$T/gauges/deept_small6_alt_seed0.pt"; EVAL="--n_sent 15 --max_len 6 --eps_list 0.02,0.03 --skip_stock 1"; EJ="$T/results/deept_small6_alt_eval_alpha_seed0.json" ;;
esac
mark "START (restart $SLURM_RESTART_COUNT)"
done_learn=$("$PY" -c "import torch,os,sys; p=sys.argv[1]; print(int(os.path.exists(p) and not torch.load(p).get('partial', True)))" "$OUT")
if [ "$done_learn" = "1" ]; then mark "LEARN already complete, skipped"; else
  "$PY" -u deept_gauge.py learn --name sst_bert_small_6 --split dev $LEARN --resume 1 --cond_pen 0.0001 --clip 1.0 --hi 0.1 --radius_iters 8 --seed 0 --gauge gauges/deept_small6_seed0.pt --out "$OUT" >> "$S/deept_learn_small6_alt_$W.log" 2>&1; RC=$?
  mark "LEARN DONE rc=$RC"; [ $RC -ne 0 ] && exit $RC
fi
if [ -f "$EJ" ]; then mark "EVAL already complete, skipped"; RC=0; else
  "$PY" -u deept_gauge.py eval_alpha --name sst_bert_small_6 $EVAL --seed 0 --hi 0.1 --gauge "$OUT" --save_json "$EJ" >> "$S/deept_eval_alpha_small6_alt_$W.log" 2>&1; RC=$?
fi
mark "DONE rc=$RC"; exit $RC
