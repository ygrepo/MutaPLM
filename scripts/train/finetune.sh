#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 2345 train.py \
--dataset_name mutadescribe \
--dataset_path ./data/mutadescribe/train.csv \
--model_name mutaplm \
--model_config_path ./configs/mutaplm_ft.yaml \
--model_checkpoint ./ckpts/pretrain/checkpoint_9.pth \
--epochs 20 \
--save_epochs 5 \
--warmup_steps 1000 \
--batch_size 1 \
--grad_accu_steps 6 \
--lr 1e-4 \
--distributed \
--save_path ./ckpts/mutaplm