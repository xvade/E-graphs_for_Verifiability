#!/bin/bash
#SBATCH -p ckpt-all -A ckpt-amath --qos=ckpt-gpu --gres=gpu:a100:1 -c 4 --mem=40G -t 1:00:00 -J formula_cost
REPO="${REPO:-/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability}"; PY="$REPO/alpha-beta-CROWN/.venv/bin/python"; T="$REPO/NNs/transformer_rewrite"; G=gauges; TMP=$REPO/NNs/transformer_rewrite/jobs
export OMP_NUM_THREADS=4; cd "$T"
run() { tag=$1; shift; echo "== $tag $(date)"; "$PY" -u gauge_formula.py validate --cost_only 1 --l1_steps 400 --l1_lr 0.02 "$@" > "$TMP/cost_$tag.log" 2>&1; echo "== $tag rc=$? $(date)"; }
Y=formula_yelp_bert_small_3; B=formula_sst_bert_big_3; S=formula_sst_bert_small_6
run big3   --name sst_bert_big_3   --n_sent 12 --pos 2 --seed 0 --probe_eps 0.0088 --gauges $G/deept_big3_seed0.pt,$G/${B}_svd_jacN_all_r5.pt,$G/${B}_l1N_r5.pt,$G/${B}_l1N_qk-av0_r5.pt --l1_init $G/deept_big3_seed0.pt --out results/formula_big3_cost.json --tag _cost
run yelp3  --name yelp_bert_small_3 --n_sent 48 --pos 2 --seed 0 --probe_eps 0.0127 --gauges $G/deept_yelp3_seed0.pt,$G/${Y}_svd_jacN_all_r5.pt,$G/${Y}_l1N_r5.pt,$G/${Y}_l1N_qk-av0_r5.pt,$G/${Y}_svd_jacN_all_r5s1.pt,$G/${Y}_l1N_r5s1.pt,$G/${Y}_l1N_qk-av0_r5s1.pt --l1_init $G/deept_yelp3_seed0.pt --out results/formula_yelp3_cost.json --tag _cost
#run yelp3s1 --name yelp_bert_small_3 --n_sent 48 --pos 2 --seed 1 --probe_eps 0.0127 --gauges $G/deept_yelp3_seed0.pt,$G/${Y}_svd_jacN_all_r5.pt,$G/${Y}_l1N_r5.pt,$G/${Y}_l1N_qk-av0_r5.pt,$G/${Y}_svd_jacN_all_r5s1.pt,$G/${Y}_l1N_r5s1.pt,$G/${Y}_l1N_qk-av0_r5s1.pt --out results/formula_yelp3_cost_s1.json --tag _costs1
run small6 --name sst_bert_small_6 --n_sent 12 --pos 2 --seed 0 --probe_eps 0.0106 --gauges $G/deept_small6_seed0.pt,$G/${S}_svd_jacN_all_r5.pt,$G/${S}_l1N_r5.pt,$G/${S}_l1N_qk-av0_r5.pt,$G/${S}_l1N_qk_av0_r4.pt --l1_init $G/deept_small6_seed0.pt --out results/formula_small6_cost.json --tag _cost
echo "ALL DONE $(date)"
