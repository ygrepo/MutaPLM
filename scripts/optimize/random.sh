#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
datasets=("AAV" "AMIE" "avGFP" "E4B" "LGK" "UBE2I")
ncandidates=20
nrounds=10

rm ./outputs/score_random.txt

for dataset in "${datasets[@]}";
do
python eval.py \
--fitness_optimize \
--num_rounds $nrounds \
--num_candidates $ncandidates \
--dataset_name $dataset \
--dataset_path ./data/fitness/$dataset \
--surrogate_path ./ckpts/landscape_params/esm1b_landscape/$dataset/decoder.pt \
--model_name random \
--model_config_path ./configs/random.yaml \
--score_save_path ./outputs/score_random.txt \
--batch_size 64 \
--device 0
done