#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
levels=("easy" "medium" "hard")

for level in "${levels[@]}";
do 
python eval.py \
--dataset_name mutadescribe \
--dataset_path ./data/mutadescribe/test_$level.csv \
--muta_engineer \
--model_name mutaplm \
--model_config_path ./configs/mutaplm_inference.yaml \
--model_checkpoint ./ckpts/mutaplm/model_checkpoint1.pth \
--pred_save_path ./outputs/mutaplm.txt \
--batch_size 4 \
--device 0
done