#!/bin/bash
# split attention-slack attribution on one DeepT model: run_attrib_split.sh <name> <tag>   (12 test sentences x 2 positions, seed 3,
# eps = 0.5 / 1 / 1.5 x stock radius; widths with frozen attention and with ONE of QK / softmax / AV linearised at the centre, and all three)
REPO="/mmfs1/gscratch/scrubbed/sgvtc/E-graphs for Verifiability"; S="$REPO/NNs/vit_rewrite/_scratch"; T="$REPO/NNs/transformer_rewrite"
echo "START attrib_split_$2 $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"
"$T/run_deept_cmd.sh" attrib --name $1 --split_attrib 1 --factors 0.5,1,1.5 --max_len 12 --n_sent 12 --pos_per_sent 2 --seed 3 --save_json "$T/results/deept_$2_attrib_split.json" > "$S/deept_attrib_split_$2.log" 2>&1
echo "DONE rc=$? attrib_split_$2 $(date) job=${SLURM_JOB_ID:-none}" >> "$S/official_sequence.log"
