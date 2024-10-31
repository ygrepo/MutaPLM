#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 2345 train.py \
--dataset_name literature \
--dataset_path ./data/pubs \
--model_name mutadescribe \
--model_config_path ./configs/mutaplm_pt.yaml \
--epochs 10 \
--save_epochs 5 \
--warmup_steps 1000 \
--batch_size 2 \
--gradient_accumulation_steps 4 \
--lr 1e-4 \
--distributed \
--save_path ./ckpts/pretrain